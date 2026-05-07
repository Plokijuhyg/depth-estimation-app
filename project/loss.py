"""
loss.py
-------
Loss Functions for Monocular Depth Estimation
==============================================

WHY SCALE AMBIGUITY MATTERS
────────────────────────────
A single image contains NO absolute scale information.  A photo of a room
could equally be a tiny model or a full-sized space — the pixel patterns
are identical when only relative geometry is viewed.

Monocular depth estimation therefore suffers from **scale ambiguity**: the
network can only learn the *relative* depth structure from training examples,
and the overall scale can drift.

Two families of losses address this differently:

1.  **L1 / L2 / Berhu** – penalise absolute differences between predicted and
    ground-truth depth in metric space.  These work when the training set is
    large and diverse enough for the network to learn an absolute scale prior
    (NYU Depth V2 is all indoor, giving a strong prior).

2.  **Scale-Invariant Loss (SILog)** – explicitly removes the effect of a
    global scale shift between prediction and GT.  Introduced by Eigen et al.
    (NeurIPS 2014).  It measures the *distribution of log-depth differences*
    rather than absolute values, so a prediction that is uniformly 2× too
    large is penalised minimally.

    SILog(d) = 1/n Σ(dᵢ²) − λ/n² (Σdᵢ)²
    where dᵢ = log(pred_i) − log(gt_i)

We train with a combined loss:
    L_total = α * L1 + β * SILog
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


MIN_DEPTH = 1e-3   # clip to avoid log(0)


# ─────────────────────────────────────────────
# 1. L1 Loss (masked)
# ─────────────────────────────────────────────

class MaskedL1Loss(nn.Module):
    """
    Pixel-wise L1 loss, ignoring pixels where the ground-truth depth
    is zero or below MIN_DEPTH (invalid sensor readings).
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : (B, 1, H, W)  predicted depth
        gt   : (B, 1, H, W)  ground-truth depth

        Returns
        -------
        scalar loss
        """
        mask  = (gt > MIN_DEPTH).detach()
        diff  = torch.abs(pred[mask] - gt[mask])
        return diff.mean()


# ─────────────────────────────────────────────
# 2. Scale-Invariant Logarithmic Loss (SILog)
# ─────────────────────────────────────────────

class ScaleInvariantLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss.

    Reference: Eigen et al., "Depth Map Prediction from a Single Image
    using a Multi-Scale Deep Network", NeurIPS 2014.

    Formula
    -------
    dᵢ = log(pred_i) − log(gt_i)
    L  = (1/n) Σ dᵢ²  −  (λ / n²) (Σ dᵢ)²

    λ = 0.5 gives the published SILog.
    λ = 0   reduces to mean of squared log-differences (no scale invariance).
    λ = 1   maximises scale invariance.
    """

    def __init__(self, lam: float = 0.5):
        super().__init__()
        self.lam = lam

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mask = (gt > MIN_DEPTH).detach()

        pred_log = torch.log(torch.clamp(pred[mask], min=MIN_DEPTH))
        gt_log   = torch.log(torch.clamp(gt[mask],   min=MIN_DEPTH))

        d  = pred_log - gt_log
        n  = d.numel()

        loss = (d ** 2).mean() - self.lam * (d.mean() ** 2)
        return loss


# ─────────────────────────────────────────────
# 3. Gradient Loss (edge sharpness)
# ─────────────────────────────────────────────

class GradientLoss(nn.Module):
    """
    Encourages sharp depth boundaries by penalising differences in
    spatial gradients between prediction and ground truth.

    ∇x d = d[x+1] − d[x]
    ∇y d = d[y+1] − d[y]
    L_grad = mean(|∇x pred − ∇x gt| + |∇y pred − ∇y gt|)
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mask = (gt > MIN_DEPTH).detach().float()

        # Horizontal gradient
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_dx   = gt  [:, :, :, 1:] - gt  [:, :, :, :-1]
        mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

        # Vertical gradient
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_dy   = gt  [:, :, 1:, :] - gt  [:, :, :-1, :]
        mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        loss_x = (torch.abs(pred_dx - gt_dx) * mask_dx).mean()
        loss_y = (torch.abs(pred_dy - gt_dy) * mask_dy).mean()

        return loss_x + loss_y


# ─────────────────────────────────────────────
# 4. Combined Loss
# ─────────────────────────────────────────────

class DepthLoss(nn.Module):
    """
    Combined depth loss:
        L = α * L1  +  β * SILog  +  γ * Gradient

    Default weights produce a balance between absolute accuracy (L1),
    scale-invariant structural fidelity (SILog), and sharpness (Gradient).

    Parameters
    ----------
    alpha : weight for L1 loss
    beta  : weight for SILog loss
    gamma : weight for gradient loss
    lam   : scale-invariant lambda (0.5 = published SILog)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta:  float = 1.0,
        gamma: float = 0.5,
        lam:   float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.l1       = MaskedL1Loss()
        self.silog    = ScaleInvariantLoss(lam=lam)
        self.gradient = GradientLoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        Returns
        -------
        total   : combined scalar loss
        details : dict of individual loss values (for logging)
        """
        l1_val   = self.l1(pred, gt)
        si_val   = self.silog(pred, gt)
        grad_val = self.gradient(pred, gt)

        total = (self.alpha * l1_val
                 + self.beta  * si_val
                 + self.gamma * grad_val)

        details = {
            "loss/total":    total.item(),
            "loss/l1":       l1_val.item(),
            "loss/silog":    si_val.item(),
            "loss/gradient": grad_val.item(),
        }

        return total, details


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pred = torch.rand(4, 1, 240, 320) * 10.0
    gt   = torch.rand(4, 1, 240, 320) * 10.0

    criterion = DepthLoss()
    loss, details = criterion(pred, gt)

    print("\nLoss components:")
    for k, v in details.items():
        print(f"  {k:20s} : {v:.6f}")
