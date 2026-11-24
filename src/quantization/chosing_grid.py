import numpy as np
import torch
import torch.nn.functional as F
import transformers
from packaging import version
import math
import numpy as np

try:
    import torch
except ImportError:  # the NumPy path and the self-test still work without torch
    torch = None

from inference_lib.triton.PO2_quant_dequant import fake_quant_dequant_N_triton

try:
    import triton
    import triton.language as tl
except ImportError:  # swap_losses falls back to the chunked torch path
    triton = None

if triton is not None:
    @triton.jit
    def _swap_losses_kernel(x_ptr, base_ptr, imp_ptr, c_ptr, out_ptr, G, C,
                            GS: tl.constexpr, BLOCK_G: tl.constexpr, HAS_IMP: tl.constexpr):
        """out[g, c] = sum_i imp[g, i] * min(base[g, i], (x[g, i] - cands[c])^2), all float64.
        Each program loads its BLOCK_G groups once and scores every candidate on them, so
        none of the (G, GS, C) intermediates of the torch version are written to memory."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_G + tl.arange(0, BLOCK_G)
        cols = tl.arange(0, GS)
        rmask = rows < G
        offs = rows[:, None] * GS + cols[None, :]
        x = tl.load(x_ptr + offs, mask=rmask[:, None], other=0.0)
        base = tl.load(base_ptr + offs, mask=rmask[:, None], other=0.0)
        if HAS_IMP:
            imp = tl.load(imp_ptr + offs, mask=rmask[:, None], other=0.0)
        for ci in range(C):
            c = tl.load(c_ptr + ci)
            d = x - c
            e = tl.minimum(base, d * d)
            if HAS_IMP:
                e = e * imp
            tl.store(out_ptr + rows * C + ci, tl.sum(e, axis=1), mask=rmask)


# relative margin of the fp32 screening in _Groups.swap_best (fp32 error is ~2e-6)
_SWAP_SCREEN_TOL = 1e-4


def _swap_losses_triton(x, base, imp, cands):
    G, gs = x.shape
    C = cands.numel()
    out = torch.empty(G, C, dtype=torch.float64, device=x.device)
    BLOCK_G = 64
    _swap_losses_kernel[(triton.cdiv(G, BLOCK_G),)](
        x, base, imp if imp is not None else x, cands, out, G, C,
        GS=gs, BLOCK_G=BLOCK_G, HAS_IMP=imp is not None)
    return out


def generate_fp_codebook(scheme="fp4_e2m1fn"):
    schemes = {
        "fp4_e3m0fn": (3, 0, 3),
        "fp4_e2m1fn": (2, 1, 1),
        "fp4_e1m2fn": (1, 2, 0),
        "fp6_e3m2fn": (3, 2, 3),
        "fp6_e2m3fn": (2, 3, 1),
        "fp8_e4m3fn": (4, 3, 7),
        "fp5_e2m2fn": (2, 2, 1),
    }
    if scheme == "nf4":
        return torch.tensor(
            [
                -1.0,
                -0.6961928009986877,
                -0.5250730514526367,
                -0.39491748809814453,
                -0.28444138169288635,
                -0.18477343022823334,
                -0.09105003625154495,
                0.0,
                0.07958029955625534,
                0.16093020141124725,
                0.24611230194568634,
                0.33791524171829224,
                0.44070982933044434,
                0.5626170039176941,
                0.7229568362236023,
                1.0,
            ]
        )
    n_e, n_m, bias = schemes.get(scheme, (None, None, None))
    #     print(n_e, n_m, bias)
    assert n_e is not None, f"Invalid scheme: {scheme}"
    signs = torch.tensor([1.0, -1.0])
    exponents = torch.arange(0, 2**n_e)
    mantissas = torch.arange(0, 2**n_m)

    codebook = []
    for sign in signs:
        for e in exponents:
            if e == 0:
                E = 1 - bias
                for m in mantissas:
                    if m == 0:
                        value = 0.0
                    else:
                        significand = m / math.pow(2.0, n_m)
                        value = sign * math.pow(2.0, E) * significand
                    codebook.append(value)
            else:
                E = e - bias
                for m in mantissas:
                    significand = 1.0 + m / pow(2.0, n_m)
                    value = sign * math.pow(2.0, E) * significand
                    codebook.append(value)

    codebook = torch.tensor(sorted((codebook)))
    return codebook


def _is_torch(x):
    return torch is not None and isinstance(x, torch.Tensor)


def _np(x):
    return x.detach().cpu().double().numpy() if _is_torch(x) else np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------------------
# exact single-codebook solver on binned statistics (same DP as select_levels_torch.py)
# ---------------------------------------------------------------------------------------
def _breakpoints(candidates):
    p = np.unique(_np(candidates))                    # sorted, deduplicated
    shift = 0.5 * (p[0] + p[-1])                      # centre for numerical stability
    q = p - shift
    mid = 0.5 * (q[:, None] + q[None, :])             # all pairwise midpoints
    B = np.unique(np.concatenate([q, mid[np.triu_indices(len(q), 1)]]))
    return p, q, mid, B, shift


def _dp(q, mid, B, stats, k):
    """stats[:, b] = [sum imp, sum imp*x, sum imp*x^2] of bin b (bin b holds B[b-1] <= x < B[b])."""
    M = len(q)
    P = np.concatenate([np.zeros((3, 1)), np.cumsum(stats, axis=1)], axis=1)

    def seg(lo, hi, c):
        return (P[2, hi] - P[2, lo]) - 2.0 * c * (P[1, hi] - P[1, lo]) + c * c * (P[0, hi] - P[0, lo])

    pidx = np.searchsorted(B, q) + 1
    midx = np.searchsorted(B, mid) + 1
    end = P.shape[1] - 1
    a = np.broadcast_to(pidx[:, None], (M, M))
    b = np.broadcast_to(pidx[None, :], (M, M))
    trans = seg(a, midx, q[:, None]) + seg(midx, b, q[None, :])
    trans[np.tril_indices(M)] = np.inf
    dp = seg(np.zeros(M, dtype=int), pidx, q)
    back = []
    for _ in range(k - 1):
        tot = dp[:, None] + trans
        back.append(np.argmin(tot, axis=0))
        dp = tot.min(axis=0)
    total = dp + seg(pidx, np.full(M, end), q)
    i = int(np.argmin(total))
    chosen = [i]
    for bp in reversed(back):
        i = int(bp[i])
        chosen.append(i)
    return chosen[::-1]


# ---------------------------------------------------------------------------------------
# the only code that touches the weights: per-group losses and per-bin statistics
# ---------------------------------------------------------------------------------------
class _Groups:
    def __init__(self, W, group_size, importance, B, shift):
        self.shift, self.nb, self.torch = shift, len(B) + 1, _is_torch(W)
        if self.torch:
            self.dev = W.device if W.device.type != "mps" else torch.device("cpu")
            x = W.detach().reshape(-1)
            assert x.numel() % group_size == 0, "number of weights must be a multiple of group_size"
            self.x = x.to(self.dev, torch.float64).reshape(-1, group_size)
            self.imp = None if importance is None else \
                importance.detach().reshape(-1, group_size).to(self.dev, torch.float64)
            Bt = torch.as_tensor(B, dtype=torch.float64, device=self.dev)
            self.bins = torch.bucketize(self.x - shift, Bt, right=True)
            # fp32 copies for the swap screening pass (fp64 is ~40x slower on B300)
            self.fast = triton is not None and self.dev.type == "cuda" and group_size & (group_size - 1) == 0
            if self.fast:
                self.x32 = self.x.float().contiguous()
                self.imp32 = None if self.imp is None else self.imp.float().contiguous()
        else:
            x = np.asarray(W, dtype=np.float64).ravel()
            assert x.size % group_size == 0, "number of weights must be a multiple of group_size"
            self.x = x.reshape(-1, group_size)
            self.imp = None if importance is None else \
                np.asarray(importance, dtype=np.float64).reshape(-1, group_size)
            self.bins = np.searchsorted(B, self.x - shift, side="right")
        self.G = self.x.shape[0]

    def _err(self, levels):
        """(G, gs) unweighted squared distance of every weight to its nearest level."""
        lv = np.sort(_np(levels))
        if self.torch:
            if len(lv) == 0:
                return torch.full_like(self.x, float("inf"))
            lv_t = torch.as_tensor(lv, dtype=torch.float64, device=self.dev)
            if len(lv) == 1:
                q = lv_t[0]
            else:
                mids = torch.as_tensor(0.5 * (lv[1:] + lv[:-1]), dtype=torch.float64, device=self.dev)
                q = lv_t[torch.bucketize(self.x, mids)]
            return (self.x - q) ** 2
        if len(lv) == 0:
            return np.full_like(self.x, np.inf)
        mids = 0.5 * (lv[1:] + lv[:-1])
        return (self.x - lv[np.searchsorted(mids, self.x, side="left")]) ** 2

    def loss(self, levels):
        """(G,) squared error of every group when rounded to its nearest level in `levels`."""
        err = self._err(levels)
        if self.imp is not None:
            err = err * self.imp
        return err.sum(dim=1).cpu().numpy() if self.torch else err.sum(axis=1)

    def swap_losses(self, levels, pos, cands, budget=1 << 25, as_tensor=False):
        """(G, C) loss of every group when levels[pos] is replaced by each value in `cands`.
        All C replacements are scored in one vectorised pass (chunked over groups).
        as_tensor: torch path only, keep the result on the device (no host copy)."""
        rest = np.delete(np.sort(_np(levels)), pos)
        base = self._err(rest)                         # error without the removed level
        G, gs = self.x.shape
        C = len(cands)
        rows = max(1, budget // (gs * C))
        if self.torch:
            c = torch.as_tensor(cands, dtype=torch.float64, device=self.dev)
            if triton is not None and self.dev.type == "cuda" and gs & (gs - 1) == 0:
                out = _swap_losses_triton(self.x.contiguous(), base.contiguous(),
                                          None if self.imp is None else self.imp.contiguous(), c)
                return out if as_tensor else out.cpu().numpy()
            out = torch.empty(G, C, dtype=torch.float64, device=self.dev)
            for s in range(0, G, rows):
                e = torch.minimum(base[s:s + rows, :, None], (self.x[s:s + rows, :, None] - c) ** 2)
                if self.imp is not None:
                    e = e * self.imp[s:s + rows, :, None]
                out[s:s + rows] = e.sum(dim=1)
            return out if as_tensor else out.cpu().numpy()
        c = np.asarray(cands, dtype=np.float64)
        out = np.empty((G, C))
        for s in range(0, G, rows):
            e = np.minimum(base[s:s + rows, :, None], (self.x[s:s + rows, :, None] - c) ** 2)
            if self.imp is not None:
                e = e * self.imp[s:s + rows, :, None]
            out[s:s + rows] = e.sum(axis=1)
        return out

    def swap_best(self, levels, pos, cands, other, threshold):
        """Exact (fp64) best replacement for levels[pos] among `cands`, given `other` (G,) =
        each group's loss under the other codebooks. Same decision as scoring every candidate
        in fp64 (swap_losses + min + sum), but only near-best candidates are scored in fp64:
          1) score all candidates in fp32 (sums of non-negative terms: relative error of every
             total is < ~(group_size + 2) * 2^-24 ~ 2e-6, far below _SWAP_SCREEN_TOL);
          2) if no candidate can reach `threshold` even allowing for that error, return None;
          3) otherwise rescore in fp64 every candidate within the tolerance of the fp32 best
             and pick among those (first index on ties, as np.argmin over all candidates).
        Returns None or (c index into cands, exact total, exact (G,) loss column)."""
        rest = np.delete(np.sort(_np(levels)), pos)
        base = self._err(rest).contiguous()                       # fp64 (G, gs)
        c64 = torch.as_tensor(cands, dtype=torch.float64, device=self.dev)
        Lc32 = _swap_losses_triton(self.x32, base.float(), self.imp32, c64.float())
        v32 = torch.minimum(Lc32, other[:, None]).sum(dim=0)
        best32 = v32.min()
        if bool(best32 * (1 - _SWAP_SCREEN_TOL) >= threshold):
            return None                                           # nothing can improve
        idx = torch.nonzero(v32 <= best32 * (1 + _SWAP_SCREEN_TOL)).squeeze(1)
        Lc = _swap_losses_triton(self.x, base, self.imp, c64[idx])  # exact, (G, |idx|)
        vals = torch.minimum(Lc, other[:, None]).sum(dim=0).cpu().numpy()
        i = int(np.argmin(vals))
        return int(idx[i]), vals[i], Lc[:, i]

    def stats(self, group_mask):
        """(3, nb) bin statistics of the groups selected by the boolean (G,) mask."""
        if self.torch:
            m = torch.as_tensor(group_mask, device=self.dev)
            xs = (self.x[m] - self.shift).reshape(-1)
            b = self.bins[m].reshape(-1)
            w = torch.ones_like(xs) if self.imp is None else self.imp[m].reshape(-1)
            acc = torch.zeros(3, self.nb, dtype=torch.float64, device=self.dev)
            acc[0].index_add_(0, b, w)
            acc[1].index_add_(0, b, w * xs)
            acc[2].index_add_(0, b, w * xs * xs)
            return acc.cpu().numpy()
        xs = (self.x[group_mask] - self.shift).ravel()
        b = self.bins[group_mask].ravel()
        w = np.ones_like(xs) if self.imp is None else self.imp[group_mask].ravel()
        return np.stack([np.bincount(b, w, self.nb), np.bincount(b, w * xs, self.nb),
                         np.bincount(b, w * xs * xs, self.nb)])


def select_codebooks(candidates, weights, k=16, n_books=2, group_size=32, importance=None,
                     n_init=4, swaps=True, n_refine=1, max_iter=100, seed=0, verbose=False):
    """
    candidates : the M allowed values, e.g. a 64-point grid
    weights    : tensor/ndarray of any shape; groups = consecutive `group_size` elements
                 of the flattened (row-major) tensor, i.e. along the last dim for a Linear weight
    k          : values per codebook
    n_books    : number of codebooks (n_books=1 reduces to the exact single-codebook solver)
    importance : optional non-negative per-weight weights, same shape as `weights`
    n_init     : number of starting points for the alternation (energy split, greedy, random...)
    swaps      : refine with single-value swaps scored on the true objective (slower, better)
    n_refine   : how many of the best alternation results get the swap refinement

    Returns (books, assign, loss, info):
      books  : (n_books, k) chosen values, each row sorted (torch tensor if weights are torch)
      assign : (G,) index of the codebook each group uses
      loss   : total (importance-weighted) squared error over all weights
      info   : list of (start name, loss after alternation, loss after swaps or None)
    """
    p, q, mid, B, shift = _breakpoints(candidates)
    k = min(k, len(p))
    grp = _Groups(weights, group_size, importance, B, shift)
    G = grp.G
    rng = np.random.default_rng(seed)

    def solve(mask):                                   # exact best codebook for these groups
        return p[_dp(q, mid, B, grp.stats(mask), k)]

    def assign_step(books):
        L = np.stack([grp.loss(b) for b in books], axis=1)            # (G, n_books)
        a = L.argmin(axis=1)
        return a, L[np.arange(G), a], L

    def alternate(assign, cur):
        """Exact update / exact assign until nothing changes. Never increases the loss."""
        books = update_step(assign, cur)
        assign, cur, L = assign_step(books)
        loss = cur.sum()
        for _ in range(max_iter):
            new_books = update_step(assign, cur)
            new_assign, new_cur, new_L = assign_step(new_books)
            new_loss = new_cur.sum()
            if not new_loss < loss:                    # no further progress
                break
            unchanged = np.array_equal(new_assign, assign)
            books, assign, cur, L, loss = new_books, new_assign, new_cur, new_L, new_loss
            if unchanged:                              # same assignment -> same books next time
                break
        return books, assign, cur, L, loss

    def swap_sweep(books, L, loss):
        """For each codebook and each of its slots: try every unused candidate in that slot,
        scored on the true objective (every group re-picks its better codebook). Keep the
        best replacement if it lowers the loss."""
        books, changed = [b.copy() for b in books], False
        # torch path: keep the (G, n_books) / (G, C) loss matrices on the device and only
        # bring the (C,) totals to the host (the full matrices made this host-transfer bound)
        L = torch.as_tensor(L, device=grp.dev) if grp.torch else L.copy()
        for j in range(n_books):
            for pos in range(k):
                cands = p[~np.isin(p, books[j])]
                if len(cands) == 0:
                    continue
                if grp.torch and grp.fast:
                    other = torch.cat([L[:, :j], L[:, j + 1:]], dim=1).amin(dim=1)
                    res = grp.swap_best(books[j], pos, cands, other, loss * (1 - 1e-12))
                    if res is not None:
                        c, val, col = res
                        if val < loss * (1 - 1e-12):
                            books[j] = np.sort(np.append(np.delete(books[j], pos), cands[c]))
                            L[:, j] = col
                            loss, changed = val, True
                    continue
                if grp.torch:
                    Lc = grp.swap_losses(books[j], pos, cands, as_tensor=True)   # (G, C) on device
                    other = torch.cat([L[:, :j], L[:, j + 1:]], dim=1).amin(dim=1)
                    vals = torch.minimum(Lc, other[:, None]).sum(dim=0).cpu().numpy()
                else:
                    Lc = grp.swap_losses(books[j], pos, cands)                   # (G, C)
                    other = np.delete(L, j, axis=1).min(axis=1, initial=np.inf)
                    vals = np.minimum(Lc, other[:, None]).sum(axis=0)
                c = int(np.argmin(vals))
                if vals[c] < loss * (1 - 1e-12):
                    books[j] = np.sort(np.append(np.delete(books[j], pos), cands[c]))
                    L[:, j] = Lc[:, c]
                    loss, changed = vals[c], True
        if grp.torch:
            L = L.cpu().numpy()
        return books, L, loss, changed

    def update_step(assign, cur):
        books = []
        for j in range(n_books):
            mask = assign == j
            if not mask.any():                         # empty codebook: refit it on the worst-served groups
                mask = cur >= np.quantile(cur, 1 - 1 / n_books)
            books.append(solve(mask))
        return books

    def starts():
        E = grp.loss(np.zeros(1))                      # per-group energy  sum imp * x^2
        ranks = np.argsort(np.argsort(E, kind="stable"), kind="stable")
        yield "energy split", ranks * n_books // G     # narrow groups vs wide groups
        books = [solve(np.ones(G, dtype=bool))]        # greedy: next codebook fits the worst-served groups
        for _ in range(1, n_books):
            cur = np.stack([grp.loss(b) for b in books], axis=1).min(axis=1)
            books.append(solve(cur >= np.quantile(cur, 1 - 1 / n_books)))
        yield "greedy", assign_step(books)[0]
        while True:
            yield "random", rng.integers(0, n_books, G)

    # 1) cheap: alternation from several starting points
    runs = []
    gen = starts()
    for _ in range(n_init if n_books > 1 else 1):
        name, assign = next(gen)
        runs.append([name, *alternate(assign, np.zeros(G))])
    runs.sort(key=lambda r: r[5])
    info = [[r[0], float(r[5]), None] for r in runs]

    # 2) expensive: swap refinement of the best few, alternating again after every sweep
    if swaps and n_books > 1:
        for r in range(min(n_refine, len(runs))):
            _, books, assign, cur, L, loss = runs[r]
            while True:
                books, L, loss, changed = swap_sweep(books, L, loss)
                if not changed:
                    break
                assign = L.argmin(axis=1)
                books, assign, cur, L, loss = alternate(assign, L[np.arange(G), assign])
            runs[r][1:] = [books, assign, cur, L, loss]
            info[r][2] = float(loss)

    for name, a, b in info:
        if verbose:
            print(f"  start {name:12s}: alternation {a:.6g}" + (f" -> swaps {b:.6g}" if b is not None else ""))
    _, books, assign, _, _, loss = min(runs, key=lambda r: r[5])
    loss = float(loss)
    books = np.stack(books)
    if _is_torch(weights):
        books, assign = torch.from_numpy(books), torch.from_numpy(assign)
    return books, assign, loss, info


def _nearest(x, levels):
    lv = np.sort(_np(levels))
    mids = 0.5 * (lv[1:] + lv[:-1])
    if _is_torch(x):
        m = torch.as_tensor(mids, dtype=torch.float32, device=x.device)
        idx = torch.bucketize(x.float(), m)
        return torch.as_tensor(lv, dtype=torch.float32, device=x.device)[idx].to(x.dtype), idx.to(torch.uint8)
    idx = np.searchsorted(mids, np.asarray(x, dtype=np.float64), side="left")
    return lv[idx].astype(np.asarray(x).dtype), idx.astype(np.uint8)


def quantize_groups(W, books, assign, group_size=32):
    """Round every weight to the nearest value of its group's codebook.
    Returns (dequantized weights with W's shape/dtype/device, uint8 codes into the sorted codebook)."""
    if _is_torch(W):
        x = W.detach().reshape(-1, group_size)
        a = torch.as_tensor(assign, device=x.device)
        out = torch.empty_like(x)
        codes = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
        for j in range(len(books)):
            m = a == j
            if bool(m.any()):
                out[m], codes[m] = _nearest(x[m], books[j])
        return out.reshape(W.shape), codes.reshape(W.shape)
    x = np.asarray(W).reshape(-1, group_size)
    a = np.asarray(assign)
    out = np.empty_like(x)
    codes = np.empty(x.shape, dtype=np.uint8)
    for j in range(len(books)):
        m = a == j
        out[m], codes[m] = _nearest(x[m], books[j])
    return out.reshape(np.shape(W)), codes.reshape(np.shape(W))

def e8m0_scales_for_M(x, M, block_size=32):
    blocks = x.float().reshape(*x.shape[:-1], -1, block_size)
    amax = blocks.abs().amax(dim=-1)
    scales = torch.exp2(torch.ceil(torch.log2((amax / M).clamp(min=2.0**-126))))
    # same guard as the kernel: log2 can round down just above a power of two
    scales = torch.where(scales * M < amax, scales * 2, scales)
    scales = torch.where(amax == 0, torch.ones_like(scales), scales)
    return blocks, scales

def search_M(x, candidates, Ms=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, ), k=16, gs=32, n_books=2):
    xc = x.cuda().float()
    results = {}
    for M in Ms:
        blocks, scales = e8m0_scales_for_M(xc, M, gs)       # blocks: (rows, n_groups, gs)
        x_scaled = blocks / scales.unsqueeze(-1)            # every block max lands in (M/2, M]
        books, assign, loss, info = select_codebooks(candidates, x_scaled, k=k,
                                                     n_books=n_books, group_size=gs)

        # one 1-D grid per book; per-group book choice from `assign`
        grids = [books[j].float().cuda() for j in range(n_books)]

        w_hat = fake_quant_dequant_N_triton(
            x=xc, grids=grids, gs=gs, scale_quant="e8m0"
            
        )
        mse = (w_hat.float() - xc).pow(2).mean().item()

        # reference: pure-torch quantization in the scaled domain, then rescale
        w_ref, _ = quantize_groups(x_scaled, books, assign, group_size=gs)
        w_ref = (w_ref * scales.unsqueeze(-1)).reshape(xc.shape)
        mse_ref = (w_ref.float() - xc).pow(2).mean().item()

        frac = [(assign == j).float().mean().item() for j in range(n_books)]
        results[M] = (mse, books, assign)
        # print(f"M={M:4.1f}  mse={mse:.4e}  ref={mse_ref:.4e}  usage={[f'{f:.2f}' for f in frac]}")
        # for j in range(n_books):
        #     print(f"    book {j}: {books[j].tolist()}")
    best_M = min(results, key=lambda m: results[m][0])
    print(f"best M = {best_M}, mse = {results[best_M][0]:.4e}")
    return best_M, results

def h_err(w_hat, w, H):
    d = w_hat.float() - w.float()
    return ((d @ H) * d).sum().item() / ((w.float() @ H) * w.float()).sum().item()


def search_M_hessian(x, H, H_eval, candidates, Ms=(4.0, 5.0, 6.0, 7.5), k=16, gs=32, n_books=2):
    xc = x.cuda().float()
    dH = torch.diagonal(H).to(xc.device)
    results = {}
    for M in Ms:
        blocks, scales = e8m0_scales_for_M(xc, M, gs)              # (out, in//gs, gs)
        x_scaled = blocks / scales.unsqueeze(-1)
        imp = (scales.unsqueeze(-1) ** 2 * dH.reshape(1, -1, gs)).contiguous()
        books, assign, loss, info = select_codebooks(candidates, x_scaled, k=k, n_books=n_books,
                                                     group_size=gs, importance=imp)
        grids = [books[j].float().cuda() for j in range(n_books)]
        grid_ids = assign.reshape(scales.shape).to(device=xc.device, dtype=torch.uint8).contiguous()
        # same scales + importance-weighted assignment (the kernel's own pick is unweighted MSE)
        w_hat = fake_quant_dequant_N_triton(x=xc, grids=grids, gs=gs,
                                            scales=scales.contiguous(), grid_ids=grid_ids)
        results[M] = dict(calib=h_err(w_hat, xc, H), hold=h_err(w_hat, xc, H_eval),
                          books=books, assign=assign)
    best_M = min(results, key=lambda m: results[m]["calib"])
    return best_M, results

