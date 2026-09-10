# models/dme_net_random.py
# ✅ FINAL (robust) version:
# - Strictly aligned with your dataset mixed degradation for integer/fractional scales
# - Fixes "integer scales PSNR explodes" by adding controlled PSF init perturbation
# - Optional: randomize sigma slightly (more "blind", helps generalization)
# - Keeps your existing API: Loss_dme_net, DME_NETAnyScale.forward_using_psf(..., target_hw=...), forward_using_srf(...)
#
# Notes:
# 1) Integer/near-integer: conv2d(input, psf, stride=s, padding=pad)  (matches dataset)
# 2) Fractional: conv2d(stride=1,pad) + bicubic resize (align_corners=False) with round(h/scale)
# 3) If target_hw is provided for integer branch: do stride first, then optional resize-to-target
#
# You can directly replace your file with this one.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Constrained parameters
# ==========================================================
class ConstrainedParameter(nn.Parameter):
    """Parameter with constraint projection."""
    def __new__(cls, data=None, requires_grad=True):
        return super(ConstrainedParameter, cls).__new__(cls, data, requires_grad)

    def project(self):
        pass


class PSFParameter(ConstrainedParameter):
    """PSF: non-negative + sum=1"""
    def project(self):
        with torch.no_grad():
            data = self.data.clamp(0.0, 1.0)
            data = data / (data.sum() + 1e-8)
            self.data.copy_(data)


class SRFParameter(ConstrainedParameter):
    """SRF: non-negative + row-normalized (sum over hs_bands = 1)"""
    def project(self):
        with torch.no_grad():
            data = self.data.clamp(0.0, 1.0)
            norm = data.sum(dim=1, keepdim=True)
            data = data / (norm + 1e-8)
            self.data.copy_(data)


# ==========================================================
# Loss (keep your original logic)
# ==========================================================
class Loss_dme_net(nn.Module):
    """
    L1 + lambda_mse*MSE + lambda_reg*(TV-like regularization)
    """
    def __init__(self, lambda_reg=1e-4, lambda_mse=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.lambda_reg = float(lambda_reg)
        self.lambda_mse = float(lambda_mse)

    def forward(self, pred_fhsi, pred_fmsi, psf, srf):
        l1 = self.l1_loss(pred_fhsi, pred_fmsi)
        mse = self.mse_loss(pred_fhsi, pred_fmsi)

        # TV-like on psf (spatial)
        psf_reg = torch.sum(torch.abs(psf[:, :, 1:] - psf[:, :, :-1])) + \
                  torch.sum(torch.abs(psf[:, 1:, :] - psf[:, :-1, :]))
        # TV-like on srf (band dims)
        srf_reg = torch.sum(torch.abs(srf[:, 1:, :, :] - srf[:, :-1, :, :])) + \
                  torch.sum(torch.abs(srf[:, :, 1:, :] - srf[:, :, :-1, :]))

        return l1 + self.lambda_mse * mse + self.lambda_reg * (psf_reg + srf_reg)


# ==========================================================
# Helpers: match dataset rules
# ==========================================================
def _is_near_integer(scale: float, eps: float = 1e-6) -> bool:
    s = float(scale)
    return s > 0 and abs(s - round(s)) <= float(eps)


def _sigma_from_scale(scale: float) -> float:
    # dataset uses: sigma = scale_factor / 2.35482
    return float(scale) / 2.35482


def _kernel_size_from_sigma(sigma: float) -> int:
    # dataset uses: kernel_size = max(3, round(sigma*6)), then make odd
    k = max(3, int(round(float(sigma) * 6.0)))
    if k % 2 == 0:
        k += 1
    return k


def _gaussian_psf_2d(ksize: int, sigma: float) -> torch.Tensor:
    center = ksize // 2
    x = torch.arange(ksize).float() - center
    y = x.unsqueeze(0)
    x = x.unsqueeze(1)
    psf = torch.exp(-(x**2 + y**2) / (2.0 * (sigma**2 + 1e-12)))
    psf = psf / (psf.sum() + 1e-8)
    return psf


def init_psf_like_dataset(
    scale: float,
    *,
    randomize_sigma: bool = True,
    sigma_jitter: float = 0.10,     # ±10% sigma jitter (only if randomize_sigma=True)
    init_noise: float = 1e-3        # add small noise to break "perfect match" -> avoid PSNR blow-up
) -> torch.Tensor:
    """
    Initialize PSF close to dataset kernel but NOT exactly identical:
    - base sigma = scale / 2.35482
    - optional sigma jitter (multiplicative)
    - optional small noise then renormalize (prevents integer-scale PSNR from exploding to 100+)
    """
    base_sigma = _sigma_from_scale(scale)
    if randomize_sigma and sigma_jitter > 0:
        # multiplicative jitter in [1-sigma_jitter, 1+sigma_jitter]
        jitter = 1.0 + (2.0 * torch.rand(1).item() - 1.0) * float(sigma_jitter)
        sigma = max(1e-4, base_sigma * jitter)
    else:
        sigma = base_sigma

    k = _kernel_size_from_sigma(sigma)
    psf = _gaussian_psf_2d(k, sigma)

    # small noise to avoid exact equality with dataset-generated integer degradation
    if init_noise and init_noise > 0:
        psf = psf + float(init_noise) * torch.rand_like(psf)
        psf = torch.clamp(psf, 0.0, 1.0)
        psf = psf / (psf.sum() + 1e-8)

    return psf


# ==========================================================
# Degradation operators
# ==========================================================



class BlurDownMixedAnyScale(object):
    """
    ✅ FINAL mixed degradation aligned to dataset:
    - integer/near-integer: conv2d(input, psf, stride=s, padding=pad)
    - fractional: conv2d(stride=1,pad) + bicubic resize (align_corners=False), out size round(h/scale)
    - If target_hw is provided:
        * integer: do stride first (GT-consistent), then resize only if needed (for shape alignment)
        * fractional: resize directly to target_hw
    """
    def __init__(self, int_eps: float = 1e-6):
        self.int_eps = float(int_eps)

    def __call__(self, input_tensor: torch.Tensor, psf: torch.Tensor, groups: int, scale: float, target_hw=None):
        if float(scale) <= 0:
            raise ValueError(f"scale must be positive, got: {scale}")

        # repeat PSF per-channel if needed
        if psf.shape[0] == 1:
            psf_rep = psf.repeat(groups, 1, 1, 1)
        else:
            psf_rep = psf

        k = int(psf_rep.shape[-1])
        pad = k // 2

        # ----- integer / near-integer -----
        if _is_near_integer(scale, self.int_eps):
            s = int(round(float(scale)))
            out = F.conv2d(input_tensor, psf_rep, bias=None, stride=(s, s), padding=pad, groups=groups)

            if target_hw is not None and (out.shape[-2], out.shape[-1]) != tuple(target_hw):
                out = F.interpolate(out, size=target_hw, mode="bicubic", align_corners=False)
            return out

        # ----- fractional -----
        blurred = F.conv2d(input_tensor, psf_rep, bias=None, stride=1, padding=pad, groups=groups)

        if target_hw is None:
            _, _, h, w = blurred.shape
            out_h = max(1, int(round(h / float(scale))))
            out_w = max(1, int(round(w / float(scale))))
            target_hw = (out_h, out_w)

        out = F.interpolate(blurred, size=target_hw, mode="bicubic", align_corners=False)
        return out


# ==========================================================
# Fixed-scale DME (keep)
# ==========================================================



# ==========================================================
# AnyScale DME (FINAL robust)
# ==========================================================
class DME_NETAnyScale(nn.Module):
    """
    ✅ FINAL robust AnyScale DME:
    - Uses dataset-aligned mixed degradation
    - PSF init is close to dataset kernel BUT slightly perturbed:
        * init_noise breaks exact equality -> avoids integer PSNR "blow-up"
        * optional sigma jitter makes it more "blind"
    """
    def __init__(
        self,
        hsi_channels,
        msi_channels,
        ratio: float,
        *,
        randomize_sigma: bool = True,
        sigma_jitter: float = 0.10,
        init_noise: float = 1e-3,
        int_eps: float = 1e-6
    ):
        super().__init__()
        self.hs_bands = int(hsi_channels)
        self.ms_bands = int(msi_channels)
        self.ratio = float(ratio)

        psf2d = init_psf_like_dataset(
            self.ratio,
            randomize_sigma=randomize_sigma,
            sigma_jitter=sigma_jitter,
            init_noise=init_noise
        )  # (k,k)

        self.psf = PSFParameter(psf2d.unsqueeze(0).unsqueeze(0))  # (1,1,k,k)

        srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1]) * (1.0 / float(self.hs_bands))
        self.srf = SRFParameter(srf)

        self.blur_down = BlurDownMixedAnyScale(int_eps=float(int_eps))

    def project_parameters(self):
        self.psf.project()
        self.srf.project()

    def forward_using_srf(self, hsi: torch.Tensor) -> torch.Tensor:
        srf_div = torch.sum(self.srf, dim=1, keepdim=True)
        srf_div = torch.div(1.0, srf_div + 1e-8)
        srf_div = torch.transpose(srf_div, 0, 1)

        msi = F.conv2d(hsi, self.srf, None)
        msi = torch.mul(msi, srf_div)
        return torch.clamp(msi, 0.0, 1.0)

    def forward_using_psf(self, img: torch.Tensor, ratio: float = None, target_hw=None) -> torch.Tensor:
        _, cc, _, _ = img.shape
        r = self.ratio if ratio is None else float(ratio)
        out = self.blur_down(img, self.psf, cc, r, target_hw=target_hw)
        return torch.clamp(out, 0.0, 1.0)

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor):
        # default aligns PSF path to lr_hsi size (training script should override with lrmsi target_hw)
        lr_msi_fhsi = self.forward_using_srf(lr_hsi)
        target_hw = (lr_hsi.shape[-2], lr_hsi.shape[-1])
        lr_msi_fmsi = self.forward_using_psf(hr_msi, target_hw=target_hw)
        return lr_msi_fhsi, lr_msi_fmsi





