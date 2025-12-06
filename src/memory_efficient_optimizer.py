"""Memory-efficient AdamW optimizer with CPU offloading for optimizer states."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import AdamW


class MemoryEfficientAdamW(AdamW):
    """
    AdamW optimizer that stores optimizer states (exp_avg, exp_avg_sq) on CPU
    with pinned memory instead of GPU. This saves ~4x the model size in GPU memory.

    When enabled=False, behaves exactly like standard AdamW.
    When enabled=True:
      - Initializes optimizer states on CPU with pin_memory=True and dtype=float32
      - Transfers states to GPU using non_blocking=True for computation
      - Transfers states back to CPU after update

    Parameters
    ----------
    params : Iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float
        Learning rate. Default: 1e-3.
    betas : Tuple[float, float]
        Coefficients for computing running averages of gradient and its square.
        Default: (0.9, 0.999).
    eps : float
        Term added to denominator for numerical stability. Default: 1e-8.
    weight_decay : float
        Decoupled weight decay coefficient. Default: 1e-2.
    amsgrad : bool
        Whether to use AMSGrad variant. Default: False.
    pin_memory : bool
        Whether to use pinned memory for CPU tensors. Default: True.
    enabled : bool
        Whether to enable CPU offloading. If False, behaves like standard AdamW.
        Default: True.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        pin_memory: bool = True,
        enabled: bool = True,
    ) -> None:
        self.pin_memory = pin_memory
        self.enabled = enabled
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Parameters
        ----------
        closure : Callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        Optional[float]
            The loss if a closure is provided, otherwise None.
        """
        if not self.enabled:
            return super().step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            amsgrad = group.get("amsgrad", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "MemoryEfficientAdamW does not support sparse gradients"
                    )

                param_device = p.device

                state = self.state[p]

                # State initialization (on CPU with pinned memory)
                if len(state) == 0:
                    state["step"] = torch.tensor(0, dtype=torch.long)
                    # Create states on CPU with pinned memory
                    state["exp_avg"] = torch.zeros_like(
                        p.data,
                        device="cpu",
                        dtype=torch.float32,
                    )
                    if self.pin_memory and torch.cuda.is_available():
                        state["exp_avg"] = state["exp_avg"].pin_memory()

                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data,
                        device="cpu",
                        dtype=torch.float32,
                    )
                    if self.pin_memory and torch.cuda.is_available():
                        state["exp_avg_sq"] = state["exp_avg_sq"].pin_memory()

                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p.data,
                            device="cpu",
                            dtype=torch.float32,
                        )
                        if self.pin_memory and torch.cuda.is_available():
                            state["max_exp_avg_sq"] = state["max_exp_avg_sq"].pin_memory()

                # Increment step
                state["step"] += 1
                step = state["step"].item()

                # Transfer states to GPU
                exp_avg = state["exp_avg"].to(param_device, non_blocking=True)
                exp_avg_sq = state["exp_avg_sq"].to(param_device, non_blocking=True)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"].to(
                        param_device, non_blocking=True
                    )

                # Bias correction factors
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Decoupled weight decay (applied directly to parameters)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # Maintain the maximum of all 2nd moment running averages
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max for normalizing running avg of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        eps
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Transfer states back to CPU
                state["exp_avg"].copy_(exp_avg, non_blocking=True)
                state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)

                if amsgrad:
                    state["max_exp_avg_sq"].copy_(max_exp_avg_sq, non_blocking=True)

        return loss
