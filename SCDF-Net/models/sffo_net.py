import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# SFFO-Net: Spectral Fourier Fusion Operator Network
# 任意尺度（整数/小数）HSI-MSI 融合（无监督友好）
# forward(lr_hsi, hr_msi) -> hr_hsi_hat
# -----------------------------

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class SpectralTokenMixer(nn.Module):
    """HSI encoder that mixes spectral information without using any advisor-specific blocks."""
    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, 1, 1, 0)
        # depthwise spatial mixing
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()
        self.rb1 = ResBlock(dim)
        self.rb2 = ResBlock(dim)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.pw(self.act(self.dw(x)))
        x = self.rb2(self.rb1(x))
        return x


class MSIEncoder(nn.Module):
    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        self.body = nn.Sequential(ResBlock(dim), ResBlock(dim), ResBlock(dim))

    def forward(self, x):
        return self.body(self.stem(x))


class ScaleEmbed(nn.Module):
    """Embed continuous scale (float) into a feature vector."""
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, scale_h: torch.Tensor, scale_w: torch.Tensor):
        # scale_h/scale_w: (B,)
        s = torch.stack([scale_h, scale_w], dim=-1)  # (B,2)
        return self.mlp(s)  # (B,dim)


class FFTDetailBranch(nn.Module):
    """Extract and gate MSI frequency details with a learnable mask conditioned on scale."""
    def __init__(self, dim: int):
        super().__init__()
        # generate gate from magnitude (1ch) + scale embedding (dim) -> gate (1ch)
        self.mag_conv = nn.Sequential(
            nn.Conv2d(1, dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
        )
        self.scale_to_map = nn.Linear(dim, dim)
        self.gate_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, feat_m: torch.Tensor, scale_vec: torch.Tensor):
        """
        feat_m: (B, D, H, W) real
        scale_vec: (B, D) real
        returns: gated MSI fft (B, D, H, W) complex
        """
        B, D, H, W = feat_m.shape
        m_f = torch.fft.fft2(feat_m, norm='ortho')  # complex64/complex32

        mag = torch.log1p(torch.abs(m_f))  # (B,D,H,W)
        mag = mag.mean(dim=1, keepdim=True)  # (B,1,H,W)

        base = self.mag_conv(mag)  # (B,D,H,W)
        s = self.scale_to_map(scale_vec).view(B, D, 1, 1).expand(-1, -1, H, W)
        gate = self.gate_head(base + s)  # (B,1,H,W)

        # stronger emphasis on high-frequency: apply gate in frequency domain
        # broadcast to D channels
        gate_d = gate.expand(-1, D, -1, -1)
        return m_f * gate_d


class ComplexFusionOperator(nn.Module):
    """Fuse HSI and MSI in complex frequency space via a learnable operator Φ."""
    def __init__(self, dim: int):
        super().__init__()
        # input channels: Re/Im of H_f and M_f -> 4*D
        self.phi = nn.Sequential(
            nn.Conv2d(4 * dim, 2 * dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(2 * dim, 2 * dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(2 * dim, 2 * dim, 1, 1, 0),
        )
        # a small residual stabilizer (bounded)
        self._alpha_logit = nn.Parameter(torch.tensor(0.4))

    def forward(self, h_f: torch.Tensor, m_f_gated: torch.Tensor):
        """
        h_f: (B,D,H,W) complex
        m_f_gated: (B,D,H,W) complex
        returns y_f: (B,D,H,W) complex
        """
        x = torch.cat([h_f.real, h_f.imag, m_f_gated.real, m_f_gated.imag], dim=1)
        delta = self.phi(x)  # (B,2D,H,W)
        d_re, d_im = torch.chunk(delta, 2, dim=1)
        d = torch.complex(d_re, d_im)
        # fuse: keep base HSI spectrum, inject MSI details through operator and a direct residual
        alpha = torch.sigmoid(self._alpha_logit)
        y_f = h_f + alpha * d + (1.0 - alpha) * m_f_gated
        return y_f




class SpatialDetailBranch(nn.Module):
    """Extract and gate spatial detail from HR-MSI features (real domain)."""
    def __init__(self, dim: int):
        super().__init__()
        # edge proxy from MSI feature
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()
        self.edge_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.scale_to_map = nn.Linear(dim, dim)

    def forward(self, feat_m: torch.Tensor, scale_vec: torch.Tensor):
        # feat_m: (B,D,H,W), scale_vec: (B,D)
        B, D, H, W = feat_m.shape
        x = self.pw(self.act(self.dw(feat_m)))
        s = self.scale_to_map(scale_vec).view(B, D, 1, 1)
        x = x + s
        gate = self.edge_head(x)  # (B,1,H,W)
        return feat_m * gate  # (B,D,H,W)


class RefineUNetLite(nn.Module):
    """A tiny UNet-like refiner (2-level) to sharpen spatial structures."""
    def __init__(self, dim: int):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            ResBlock(dim),
        )
        self.down = nn.Conv2d(dim, dim, 3, 2, 1)
        self.enc2 = nn.Sequential(
            nn.GELU(),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1, 1, 0),
            nn.GELU(),
            ResBlock(dim),
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down(x1))
        x2u = self.up(x2)
        if x2u.shape[-2:] != x1.shape[-2:]:
            x2u = F.interpolate(x2u, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        y = self.fuse(torch.cat([x1, x2u], dim=1))
        return y


class SFFO_NET_DC(nn.Module):
    """DualConsistent-SFFO: SFFO_NET + Spatial Detail Branch + UNet-lite refine."""
    def __init__(self, hsi_channels: int, msi_channels: int, dim: int = 96, out_refine_blocks: int = 3):
        super().__init__()
        self.dim = dim
        self.hsi_enc = SpectralTokenMixer(hsi_channels, dim)
        self.msi_enc = MSIEncoder(msi_channels, dim)
        self.scale_emb = ScaleEmbed(dim)

        self.fft_detail = FFTDetailBranch(dim)
        self.spa_detail = SpatialDetailBranch(dim)
        self.cfo = ComplexFusionOperator(dim)

        self.unet_refine = RefineUNetLite(dim)
        self.refine = nn.Sequential(*[ResBlock(dim) for _ in range(out_refine_blocks)])

        self.out = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, hsi_channels, 1, 1, 0)
        )

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor):
        B, C, h, w = lr_hsi.shape
        _, _, H, W = hr_msi.shape

        # continuous scale from shapes (supports non-integer)
        scale_h = (torch.ones(B, device=lr_hsi.device, dtype=lr_hsi.dtype) * (H / float(h)))
        scale_w = (torch.ones(B, device=lr_hsi.device, dtype=lr_hsi.dtype) * (W / float(w)))
        svec = self.scale_emb(scale_h, scale_w)  # (B,dim)

        # HSI branch
        f_h = self.hsi_enc(lr_hsi)  # (B,dim,h,w)
        f_h_up = F.interpolate(f_h, size=(H, W), mode='bicubic', align_corners=False)

        # MSI branch
        f_m = self.msi_enc(hr_msi)  # (B,dim,H,W)

        # dual-domain detail from MSI
        m_f_gated = self.fft_detail(f_m, svec)          # complex (B,dim,H,W)
        m_spa = self.spa_detail(f_m, svec)              # real   (B,dim,H,W)

        # frequency fusion
        h_f = torch.fft.fft2(f_h_up, norm='ortho')
        y_f = self.cfo(h_f, m_f_gated)
        y = torch.fft.ifft2(y_f, norm='ortho').real

        # inject spatial detail + refine
        y = y + m_spa
        y = self.unet_refine(y)
        y = self.refine(y)

        rec = self.out(y)
        return rec
