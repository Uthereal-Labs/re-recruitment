from dataclasses import dataclass
from typing import List
import re
import torch
import torch.nn.functional as F
from torch import nn, optim


# ---------------------------
# Minimal utilities
# ---------------------------


def token_span_masks(prompt_lens: List[int], seq_lens: List[int], device: str):
    """
    Make a boolean mask [N, T_max] that is True ONLY over the generated span for each row.
    prompt_lens[i] is the prompt length; seq_lens[i] is the full length (prompt+gen).
    """
    assert len(prompt_lens) == len(seq_lens)
    N = len(prompt_lens)
    T = int(max(seq_lens))
    m = torch.zeros((N, T), dtype=torch.bool, device=device)
    for i, (p, s) in enumerate(zip(prompt_lens, seq_lens)):
        s_clamped = min(s, T)
        if s_clamped > p:
            m[i, p:s_clamped] = True
    return m


def _extract_first_int(text: str):
    m = re.search(r"(-?\d+)", text)
    return int(m.group(1)) if m else None


def _accuracy_reward(completions: List[str], answers: List[str]) -> List[float]:
    out = []
    for txt, gt in zip(completions, answers):
        g = _extract_first_int(txt)
        out.append(1.0 if g is not None and str(g).strip() == str(gt).strip() else 0.0)
    return out


def _format_reward(completions: List[str]) -> List[float]:
    out = []
    for txt in completions:
        ok = bool(re.search(r"<answer>\s*-?\d+\s*</answer>", txt.strip(), flags=re.I))
        out.append(1.0 if ok else 0.0)
    return out


def combined_reward(
    completions: List[str], answers: List[str], wa: float = 0.9, wf: float = 0.1
) -> List[float]:
    """
    Simple shaped reward = 0.9 * accuracy + 0.1 * formatting.
    """
    acc = _accuracy_reward(completions, answers)
    fmt = _format_reward(completions)
    return [wa * a + wf * f for a, f in zip(acc, fmt)]


def gather_logprobs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: str
) -> torch.Tensor:
    """
    Per-token log-probabilities for NEXT token (aligned to input ids length).
    Returns [N, T] tensor where last position is padded with 0 for alignment.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [N, T, V]
    logp = F.log_softmax(logits, dim=-1)
    target = input_ids[:, 1:].contiguous()  # next-token ids
    logp_next = logp[:, :-1, :].contiguous()  # shift
    tok_logp = torch.gather(logp_next, dim=-1, index=target.unsqueeze(-1)).squeeze(
        -1
    )  # [N, T-1]
    pad = torch.zeros(
        (tok_logp.size(0), 1), device=tok_logp.device, dtype=tok_logp.dtype
    )
    tok_logp = torch.cat([tok_logp, pad], dim=1)  # [N, T]
    return tok_logp


# ---------------------------
# GRPO
# ---------------------------


@dataclass
class GRPOConfig:
    group_size: int = 4  # K completions per prompt
    eps_clip: float = 0.2  # PPO clip on importance ratio
    beta_kl: float = 0.02  # weight for KL to reference policy
    lr: float = 5e-5  # optimizer LR


class GRPOTrainer:
    """
    Minimal GRPO trainer operating on already-sampled sequences.
    You must pass:
    - policy: trainable causal LM (transformers-compatible forward returning logits)
    - ref: frozen reference LM (same tokenizer space)
    - tokenizer: with .pad_token_id
    - cfg: GRPOConfig
    - answers: list of ground-truth answers (aligned by prompt index)

    step(...) expects:
    - prompt_input_ids: tokenized prompts (unused for loss, kept for compatibility)
    - outputs: token ids of prompt+generated sequences [N, T]
    - texts: decoded strings for each output (used by reward function)
    - prompt_lens: list[int] length N
    - seq_lens: list[int] length N
    """

    def __init__(self, policy, ref, tokenizer, cfg: GRPOConfig, answers: List[str]):
        self.policy = policy
        self.ref = ref
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.answers = answers
        self.opt = optim.AdamW(self.policy.parameters(), lr=cfg.lr)

    def _advantages(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        """
        Group-relative, zero-mean, z-normalized advantages.
        rewards is [N] in prompt-major order (i.e., K samples per prompt, contiguous).
        """
        N = rewards.numel()
        assert N % group_size == 0, "N must be divisible by group_size"
        B = N // group_size
        r = rewards.view(B, group_size)
        mean = r.mean(dim=1, keepdim=True)
        std = r.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        A = (r - mean) / std
        return A.view(N)

    def _clip_surrogate(
        self, ratio: torch.Tensor, adv: torch.Tensor, eps: float
    ) -> torch.Tensor:
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
        return torch.minimum(unclipped, clipped)

    def step(
        self,
        prompt_input_ids,
        outputs: torch.Tensor,
        texts: List[str],
        prompt_lens: List[int],
        seq_lens: List[int],
    ):
        device = outputs.device
        N, T = outputs.shape

        # Attention over full sequences (prompt + generated)
        attn = (outputs != self.tokenizer.pad_token_id).to(outputs.dtype)

        # "Old" log-probs at sample time. Here we re-run the same policy and detach to emulate
        # storing old log-probs when the samples were created.
        with torch.no_grad():
            old_logp = gather_logprobs(
                self.policy, outputs, attn, device=device
            ).detach()

        # Rewards per sequence (align answers by prompt index, repeating per group)
        answers_repeated = [self.answers[i // self.cfg.group_size] for i in range(N)]
        rewards = torch.tensor(
            combined_reward(texts, answers_repeated), device=device, dtype=torch.float32
        )

        # Group-relative advantages
        adv = self._advantages(rewards, self.cfg.group_size)  # [N]

        # Mask only the generated span (exclude prompt/pad tokens)
        gen_mask = token_span_masks(prompt_lens, seq_lens, device=device)  # [N, T]
        tok_count = gen_mask.sum().clamp_min(1)

        # Current and reference log-probs
        cur_logp = gather_logprobs(self.policy, outputs, attn, device=device)  # [N, T]
        ref_logp = gather_logprobs(
            self.ref, outputs, attn, device=device
        ).detach()  # [N, T]

        # Importance ratio and PPO-style clipped surrogate on generated tokens
        ratio = torch.exp(cur_logp - old_logp)  # [N, T]
        adv_tok = adv.view(N, 1).expand_as(ratio)
        surr = self._clip_surrogate(ratio, adv_tok, self.cfg.eps_clip)
        policy_loss = -(surr * gen_mask).sum() / tok_count

        # Simple per-token KL to the frozen reference (on sampled tokens)
        kl_tok = cur_logp - ref_logp
        kl_loss = self.cfg.beta_kl * (kl_tok.abs() * gen_mask).sum() / tok_count

        loss = policy_loss + kl_loss

        # Optimize
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "mean_reward": float(rewards.mean().cpu()),
        }
