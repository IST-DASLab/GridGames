import torch
import torch.nn.functional as F
from tqdm.auto import trange


@torch.no_grad()
def compute_kl_divergence(
    quantized_model,
    teacher_model,
    data,
    batch_size: int = 1,
    pad_token_id: int | None = None,
    temperature: float = 1.0,
):
    """
    Computes average token-level KL divergence: KL(p_teacher || p_student)
    over next-token distributions (shifted by 1), matching your PPL code.

    Args:
      quantized_model: student model (e.g., quantized) returning .logits [B, T, V]
      teacher_model: teacher model returning .logits [B, T, V]
      data: list of tensors shaped [1, T] or [B, T] that can be torch.cat'ed along dim 0
      batch_size: number of samples to concatenate per step
      pad_token_id: if set, tokens equal to pad_token_id are ignored in the average
      temperature: optional softening. For pure evaluation keep 1.0

    Returns:
      avg_kl_nats: float, average KL per (non-pad) token in nats
    """
    quantized_model.eval()
    teacher_model.eval()

    num_samples = len(data)

    student_device = next(quantized_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device

    kl_running = 0.0
    tokens_processed = 0

    for i in trange(0, num_samples, batch_size, desc="Computing KL", leave=False):
        j = min(i + batch_size, num_samples)

        # Match your PPL batching behavior
        inputs = torch.cat(data[i:j])

        # Forward on each model's device
        s_inp = inputs.to(student_device)
        t_inp = inputs.to(teacher_device)

        student_logits = quantized_model(s_inp).logits  # [B, T, V]
        teacher_logits = teacher_model(t_inp).logits    # [B, T, V]

        # Shift for next-token prediction
        student_shift = student_logits[:, :-1, :]
        teacher_shift = teacher_logits[:, :-1, :]

        # Bring teacher logits to student device for KL computation
        teacher_shift = teacher_shift.to(student_device)

        if student_shift.size(-1) != teacher_shift.size(-1):
            raise ValueError(
                f"Vocab mismatch: student V={student_shift.size(-1)} "
                f"teacher V={teacher_shift.size(-1)}"
            )

        # Temperature scaling (optional)
        if temperature != 1.0:
            student_shift = student_shift / temperature
            teacher_shift = teacher_shift / temperature

        # Compute KL(p_teacher || p_student) per token:
        # KL = sum_v p_t(v) * (log p_t(v) - log p_s(v))
        # Upcast logits to fp32 BEFORE softmax so low-probability tokens
        # are not crushed by bf16/fp16 mantissa precision.
        student_log_probs = F.log_softmax(student_shift.float(), dim=-1)
        teacher_probs = F.softmax(teacher_shift.float(), dim=-1)

        # kl_div expects input = log-probs, target = probs when log_target=False
        # reduction='none' keeps per-element; then sum over vocab -> per-token KL
        kl_per_token = F.kl_div(
            student_log_probs, teacher_probs, reduction="none", log_target=False
        ).sum(dim=-1).clamp(min=0)  # [B, T-1]

        # Optional padding mask based on shifted labels
        if pad_token_id is not None:
            shift_labels = s_inp[:, 1:]  # [B, T-1]
            mask = (shift_labels != pad_token_id)
            kl_sum = (kl_per_token * mask).sum()
            tok_count = mask.sum()
        else:
            kl_sum = kl_per_token.sum()
            tok_count = torch.tensor(kl_per_token.numel(), device=student_device)

        tok_count_int = int(tok_count.item())
        if tok_count_int == 0:
            continue

        batch_avg_kl = (kl_sum / tok_count).item()

        # Running token-weighted average (same style as your PPL code)
        a = tok_count_int / (tokens_processed + tok_count_int)
        b = tokens_processed / (tokens_processed + tok_count_int)
        kl_running = a * batch_avg_kl + b * kl_running
        tokens_processed += tok_count_int

    return float(kl_running)



