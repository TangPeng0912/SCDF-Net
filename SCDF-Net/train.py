# train_anyscale_fsii_up_merged1_anyscale_unsup_eval_2026dc_v3.py
# 2026DC-v3: SFFO_NET_DC + Dual-Consistency (scale + spectral) + Frequency Consistency (FFT high-frequency)
import os
import sys
import time
import random
import copy
import re
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HSI_DatasetAnyScale
from models.dme_net_random import Loss_dme_net, DME_NETAnyScale


from models.sffo_net import SFFO_NET_DC

from utils import save_checkpoint, save_results, calculate_metrics, frequency_analyse, plot_metrics
from utils_plot import plot_metrics_with_epochs
from options_anyscale import get_options


# -------------------------
# CLI overrides (will be stripped from sys.argv before get_options())
# -------------------------
def _parse_cli_overrides(argv):
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--mixed_scales", type=str, default=None,
                        help="第二阶段混合倍率，逗号分隔，例如: 2.7,3.0,3.5,4.0")
    parser.add_argument("--eval_scale", type=float, default=None, help="第二阶段评估倍率。")

    parser.add_argument("--scale_min", type=float, default=None, help="连续倍率下界（与 scale_max/scale_step 配合使用）。")
    parser.add_argument("--scale_max", type=float, default=None, help="连续倍率上界（与 scale_min/scale_step 配合使用）。")
    parser.add_argument("--scale_step", type=float, default=None, help="连续倍率步长（例如 0.2）。提供后会自动生成 mixed_scales 列表。")

    parser.add_argument("--psf_mismatch_mode", type=str, default="resize",
                        help="PSF 尺寸不一致时处理方式：resize | init | keep_ckpt")

    # ---- 2026DC weights & warmup ----
    parser.add_argument("--sc_w", type=float, default=None, help="scale-consistency weight（Dual-Consistency）")
    parser.add_argument("--spec_w", type=float, default=None, help="spectral-consistency weight（Dual-Consistency, MSI-space SAM）")
    parser.add_argument("--scspec_warmup_epochs", type=int, default=None,
                        help="Warmup epochs for sc/spec weights (linear 0->1).")
    parser.add_argument("--freq_w", type=float, default=None, help="frequency-consistency weight (FFT high-frequency L1, MSI-space).")
    parser.add_argument("--freq_keep", type=float, default=None,
                        help="High-frequency keep ratio for FFT mask (e.g., 0.15).")

    # ---- v3: HSI-space SAM + clamp for stability ----
    parser.add_argument("--sam_hsi_w", type=float, default=None,
                        help="HSI-space SAM self-supervised weight: SAM(D_s(rec_hsi), lrhsi).")
    parser.add_argument("--clamp_01", action="store_true",
                        help="If set, clamp key tensors to [0,1] before SAM/FFT losses for stability.")

    args, remaining = parser.parse_known_args(argv[1:])

    mixed_scales = None
    if args.mixed_scales:
        try:
            mixed_scales = [float(x.strip()) for x in args.mixed_scales.split(",") if x.strip()]
        except ValueError as e:
            raise ValueError(f"--mixed_scales 解析失败: {args.mixed_scales}") from e
        if len(mixed_scales) == 0:
            raise ValueError("--mixed_scales 不能为空")

    scale_range = None
    if args.scale_min is not None or args.scale_max is not None or args.scale_step is not None:
        if args.scale_min is None or args.scale_max is None or args.scale_step is None:
            raise ValueError("--scale_min/--scale_max/--scale_step 必须同时提供")
        if args.scale_min <= 0 or args.scale_max <= 0:
            raise ValueError(f"scale_min/scale_max 必须为正数: {args.scale_min}, {args.scale_max}")
        if args.scale_max < args.scale_min:
            raise ValueError(f"scale_max 必须 >= scale_min: {args.scale_max} < {args.scale_min}")
        if args.scale_step <= 0:
            raise ValueError(f"scale_step 必须为正数: {args.scale_step}")
        scale_range = (float(args.scale_min), float(args.scale_max), float(args.scale_step))

    return {
        "mixed_scales": mixed_scales,
        "scale_range": scale_range,
        "eval_scale": args.eval_scale,
        "psf_mismatch_mode": args.psf_mismatch_mode,
        "sc_w": args.sc_w,
        "spec_w": args.spec_w,
        "scspec_warmup_epochs": args.scspec_warmup_epochs,
        "freq_w": args.freq_w,
        "freq_keep": args.freq_keep,
        "sam_hsi_w": args.sam_hsi_w,
        "clamp_01": args.clamp_01,
    }, [argv[0]] + remaining


CLI_OVERRIDES, _REMAINING_ARGV = _parse_cli_overrides(sys.argv)
sys.argv = _REMAINING_ARGV


# -------------------------
# Utils
# -------------------------
def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _format_scale_tag(scale_factor: float) -> str:
    s = float(scale_factor)
    if s.is_integer():
        scale_str = f"{int(s)}.0"
    else:
        scale_str = f"{s:g}"
    return scale_str.replace('.', 'p')


def _round_scale_value(scale: float, ndigits: int = 6) -> float:
    return float(round(float(scale), int(ndigits)))


def _round_scale_list(scales, ndigits: int = 6):
    return [_round_scale_value(s, ndigits=ndigits) for s in scales]


def _make_run_suffix(opt) -> str:
    scale_tag = _format_scale_tag(opt.scale_factor)
    scale_suffix = f"x{scale_tag}"
    model_tag = ""
    if hasattr(opt, "second_stage_model") and opt.second_stage_model == "FSII_UP_V2":
        model_tag = "fsii_up_v2"
    if hasattr(opt, "second_stage_model") and opt.second_stage_model == "FSII_BIGREV":
        model_tag = "fsii_bigrev"

    if getattr(opt, "suffix", ""):
        parts = [opt.suffix]
        if model_tag and model_tag not in opt.suffix:
            parts.append(model_tag)
        if scale_suffix not in opt.suffix:
            parts.append(scale_suffix)
        return "_".join(parts)

    if model_tag:
        return f"{model_tag}_{scale_suffix}"
    return scale_suffix


def _strip_scale_suffix(suffix: str) -> str:
    if not suffix:
        return suffix
    parts = suffix.split('_')
    cleaned = [p for p in parts if not re.match(r"^x\d+(?:p\d+)?$", p)]
    return "_".join(cleaned)


def _contains_scale(scales, value, eps=1e-8):
    return any(abs(float(s) - float(value)) <= eps for s in scales)


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if len(state_dict) == 0:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _resize_psf_to_shape(psf, target_shape):
    if psf.ndim != 4 or len(target_shape) != 4:
        raise ValueError(f"PSF 维度异常: got {tuple(psf.shape)}, target {tuple(target_shape)}")

    _, _, th, tw = target_shape
    _, _, h, w = psf.shape
    if (h, w) == (th, tw):
        out = psf
    else:
        out = F.interpolate(psf, size=(th, tw), mode="bicubic", align_corners=False)

    out = torch.clamp(out, min=0.0)
    out = out / (out.sum() + 1e-8)
    return out


def _load_blind_net_state_dict_with_psf_fallback(model, checkpoint, device, scale, mismatch_mode="resize"):
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = _normalize_state_dict_keys(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as e:
        msg = str(e)
        if "size mismatch for psf" not in msg:
            raise
        if "psf" not in state_dict:
            raise

        ckpt_psf = state_dict["psf"].detach().to(device)
        if ckpt_psf.ndim != 4:
            raise RuntimeError(f"checkpoint['psf'] 维度异常: {tuple(ckpt_psf.shape)}")

        model_psf_shape = tuple(model.psf.shape)
        ckpt_psf_shape = tuple(ckpt_psf.shape)
        mode = str(mismatch_mode).lower()

        if mode == "resize":
            resized_psf = _resize_psf_to_shape(ckpt_psf, model_psf_shape)
            state_dict["psf"] = resized_psf
            print(
                f"[Stage2-Mixed][WARN] scale={scale} 的 PSF 尺寸不一致："
                f"model={model_psf_shape} vs ckpt={ckpt_psf_shape}。"
                f"已将 ckpt PSF 重采样到 model 尺寸后加载。"
            )
            model.load_state_dict(state_dict, strict=True)
            return

        if mode == "init":
            state_dict = dict(state_dict)
            state_dict.pop("psf", None)
            print(
                f"[Stage2-Mixed][WARN] scale={scale} 的 PSF 尺寸不一致："
                f"model={model_psf_shape} vs ckpt={ckpt_psf_shape}。"
                f"将保留当前初始化 PSF，仅加载其余参数。"
            )
            model.load_state_dict(state_dict, strict=False)
            return

        if mode == "keep_ckpt":
            from models.dme_net_random import PSFParameter
            state_dict = dict(state_dict)
            state_dict.pop("psf", None)
            print(
                f"[Stage2-Mixed][WARN] scale={scale} 的 PSF 尺寸不一致："
                f"model={model_psf_shape} vs ckpt={ckpt_psf_shape}。"
                f"将使用 ckpt PSF 形状并加载其余参数。"
            )
            model.load_state_dict(state_dict, strict=False)
            model.psf = PSFParameter(ckpt_psf.detach().clone())
            return

        raise ValueError(f"未知 psf mismatch mode: {mismatch_mode}，支持: init | resize | keep_ckpt")


def _build_stage1_suffix(opt, scale_tag: str) -> str:
    suffix = getattr(opt, "suffix", "") or ""
    if "${SCALE_TAG}" in suffix:
        return suffix.replace("${SCALE_TAG}", scale_tag)
    if "{SCALE_TAG}" in suffix:
        return suffix.replace("{SCALE_TAG}", scale_tag)

    if re.search(r"x\d+(?:p\d+)?", suffix):
        return re.sub(r"x\d+(?:p\d+)?", f"x{scale_tag}", suffix, count=1)

    base = _strip_scale_suffix(suffix)
    if base:
        return f"{base}_x{scale_tag}"
    return f"stage1_dme_x{scale_tag}"


def analyse_frequency(opt):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    train_dataset = HSI_DatasetAnyScale(opt, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    batch = next(iter(train_loader))
    lrhsi = batch['lrhsi'].to(device)
    target = batch['hsi'].to(device)

    run_suffix = _make_run_suffix(opt)
    opt.run_suffix = run_suffix
    save_dir = os.path.join(opt.results_dir, f"frequency_analysis_{opt.dataset_name}_{run_suffix}")
    os.makedirs(save_dir, exist_ok=True)

    results = frequency_analyse(lrhsi, target, scale_factor=opt.scale_factor, save_dir=save_dir)
    print(f"频率分析完成，结果保存在 {save_dir}")
    return results


def physical_constraint_loss(model):
    # 非负 + 归一化约束
    psf_loss = torch.abs(torch.sum(model.psf) - 1.0)
    psf_loss += torch.mean(torch.relu(-model.psf))
    srf_loss = torch.mean(torch.relu(-model.srf))
    return psf_loss + srf_loss


# -------------------------
# Stage-1
# -------------------------
def train_first_stage():
    opt = get_options()
    set_random_seed(getattr(opt, 'seed', 42))

    run_suffix = _make_run_suffix(opt)
    opt.run_suffix = run_suffix
    checkpoints_dir = os.path.join(opt.checkpoints_dir, f"{opt.dataset_name}_{run_suffix}")
    results_dir = os.path.join(opt.results_dir, f"{opt.dataset_name}_{run_suffix}")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    train_dataset = HSI_DatasetAnyScale(opt, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    model = DME_NETAnyScale(
        hsi_channels=train_dataset.lrhsi_list[0].shape[2],
        msi_channels=train_dataset.hrmsi_list[0].shape[2],
        ratio=float(opt.scale_factor),
        randomize_sigma=False,
        sigma_jitter=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_stage1, betas=(opt.beta1, opt.beta2))
    criterion = Loss_dme_net().to(device)

    total_steps = opt.num_epochs_first_stage * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=opt.lr_stage1,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1e4
    )

    eval_batch = next(iter(train_loader))
    eval_lrhsi = eval_batch['lrhsi'].to(device)
    eval_hrmsi = eval_batch['hrmsi'].to(device)
    eval_lrmsi = eval_batch['lrmsi'].to(device)

    metrics_history = {
        'psnr_hsi': [], 'sam_hsi': [], 'rmse_hsi': [], 'ergas_hsi': [], 'cc_hsi': [],
        'psnr_msi': [], 'sam_msi': [], 'rmse_msi': [], 'ergas_msi': [], 'cc_msi': []
    }

    warmup_epochs = getattr(opt, "constraint_warmup_epochs", max(1, int(0.2 * opt.num_epochs_first_stage)))
    constraint_w_min = getattr(opt, "constraint_w_min", 0.02)
    constraint_w_max = getattr(opt, "constraint_w_max", 0.12)

    for epoch in range(opt.num_epochs_first_stage):
        model.train()
        epoch_loss = 0.0
        num_iter = 0
        start_time = time.time()

        last_loss = None
        last_constraint_loss = None
        last_total_loss = None

        for batch in train_loader:
            lrhsi = batch['lrhsi'].to(device)
            hrmsi = batch['hrmsi'].to(device)
            lrmsi = batch['lrmsi'].to(device)

            target_hw = (lrmsi.shape[-2], lrmsi.shape[-1])

            lr_msi_fhsi = model.forward_using_srf(lrhsi)
            if (lr_msi_fhsi.shape[-2], lr_msi_fhsi.shape[-1]) != target_hw:
                lr_msi_fhsi = F.interpolate(lr_msi_fhsi, size=target_hw, mode='bicubic', align_corners=False)

            lr_msi_fmsi = model.forward_using_psf(hrmsi, target_hw=target_hw)

            loss = criterion(lr_msi_fhsi, lr_msi_fmsi, model.psf, model.srf)

            if warmup_epochs > 0:
                progress = min(1.0, float(epoch + 1) / float(warmup_epochs))
            else:
                progress = 1.0
            constraint_w = constraint_w_min + (constraint_w_max - constraint_w_min) * progress

            constraint_loss = physical_constraint_loss(model)
            total_loss = loss + constraint_w * constraint_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.project_parameters()
            scheduler.step()

            epoch_loss += total_loss.item()
            num_iter += 1
            last_loss = loss
            last_constraint_loss = constraint_loss
            last_total_loss = total_loss

        avg_loss = epoch_loss / max(num_iter, 1)
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 100 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                target_hw_eval = (eval_lrmsi.shape[-2], eval_lrmsi.shape[-1])

                lr_msi_fhsi_eval = model.forward_using_srf(eval_lrhsi)
                if (lr_msi_fhsi_eval.shape[-2], lr_msi_fhsi_eval.shape[-1]) != target_hw_eval:
                    lr_msi_fhsi_eval = F.interpolate(lr_msi_fhsi_eval, size=target_hw_eval,
                                                     mode='bicubic', align_corners=False)

                lr_msi_fmsi_eval = model.forward_using_psf(eval_hrmsi, target_hw=target_hw_eval)

                metrics_1 = calculate_metrics(eval_lrmsi, lr_msi_fhsi_eval, scale_factor=opt.scale_factor)
                metrics_2 = calculate_metrics(eval_lrmsi, lr_msi_fmsi_eval, scale_factor=opt.scale_factor)

            metrics_history['psnr_hsi'].append(metrics_1["psnr"])
            metrics_history['sam_hsi'].append(metrics_1["sam"])
            metrics_history['rmse_hsi'].append(metrics_1["rmse"])
            metrics_history['ergas_hsi'].append(metrics_1["ergas"])
            metrics_history['cc_hsi'].append(metrics_1["cc"])
            metrics_history['psnr_msi'].append(metrics_2["psnr"])
            metrics_history['sam_msi'].append(metrics_2["sam"])
            metrics_history['rmse_msi'].append(metrics_2["rmse"])
            metrics_history['ergas_msi'].append(metrics_2["ergas"])
            metrics_history['cc_msi'].append(metrics_2["cc"])

            if last_loss is not None:
                print(
                    f'current epoch: {epoch+1}, current loss: {last_loss.item():.4f}, '
                    f'constraint loss: {last_constraint_loss.item():.4f}, total loss: {last_total_loss.item():.4f}, '
                    f'avg loss: {avg_loss:.4f}, lr: {current_lr:.6f}, time: {time.time() - start_time:.2f}s'
                )
            print(f'PSNR from hsi: {metrics_1["psnr"]:.2f} | SAM from hsi: {metrics_1["sam"]:.4f} | RMSE from hsi: {metrics_1["rmse"]:.4f} | ERGAS from hsi: {metrics_1["ergas"]:.4f} | CC from hsi: {metrics_1["cc"]:.4f}')
            print(f'PSNR from msi: {metrics_2["psnr"]:.2f} | SAM from msi: {metrics_2["sam"]:.4f} | RMSE from msi: {metrics_2["rmse"]:.4f} | ERGAS from msi: {metrics_2["ergas"]:.4f} | CC from msi: {metrics_2["cc"]:.4f}')

    save_checkpoint(model, optimizer, opt, 'checkpoint_first_stage_final.pth')

    hsi_metrics = {'psnr': metrics_history['psnr_hsi'], 'sam': metrics_history['sam_hsi'], 'rmse': metrics_history['rmse_hsi'], 'ergas': metrics_history['ergas_hsi'], 'cc': metrics_history['cc_hsi']}
    msi_metrics = {'psnr': metrics_history['psnr_msi'], 'sam': metrics_history['sam_msi'], 'rmse': metrics_history['rmse_msi'], 'ergas': metrics_history['ergas_msi'], 'cc': metrics_history['cc_msi']}
    plot_metrics(hsi_metrics, results_dir, 1, opt.dataset_name, 'from_hsi')
    plot_metrics(msi_metrics, results_dir, 1, opt.dataset_name, 'from_msi')


# -------------------------
# Stage-2 model factory
# -------------------------
def create_model(model_name, hsi_channels, msi_channels, device, scale_factor):
    

    if model_name == 'SFFO_NET_DC':
        # 2026DC model
        return SFFO_NET_DC(hsi_channels=hsi_channels, msi_channels=msi_channels,
                           dim=96, out_refine_blocks=3).to(device)


    raise ValueError(f"不支持的模型名称: {model_name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# -------------------------
# Stage-2 (Unsupervised)
# -------------------------
def train_second_stage():
    opt = get_options()
    set_random_seed(getattr(opt, 'seed', 42))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    run_suffix = _make_run_suffix(opt)
    checkpoints_dir = os.path.join(opt.checkpoints_dir, f"{opt.dataset_name}_{run_suffix}")
    results_dir = os.path.join(opt.results_dir, f"{opt.dataset_name}_{run_suffix}")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(checkpoints_dir, 'options.txt'), 'w') as f:
        for k, v in vars(opt).items():
            f.write(f"{k}: {v}\n")
        for k, v in CLI_OVERRIDES.items():
            f.write(f"CLI_OVERRIDES.{k}: {v}\n")

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    # mixed scales
    if CLI_OVERRIDES["mixed_scales"] is not None:
        mixed_scales = _round_scale_list(CLI_OVERRIDES["mixed_scales"])
    elif CLI_OVERRIDES["scale_range"] is not None:
        smin, smax, sstep = CLI_OVERRIDES["scale_range"]
        n_steps = int(np.floor((smax - smin) / sstep + 1e-8)) + 1
        mixed_scales = [float(smin + i * sstep) for i in range(n_steps)]
        if mixed_scales[-1] < smax - 1e-8:
            mixed_scales.append(float(smax))
        mixed_scales = _round_scale_list(mixed_scales)
    else:
        mixed_scales = _round_scale_list([2.7, 3.0, 3.5, 4.0])

    def _opt_for_scale(base_opt, s):
        opt_s = copy.deepcopy(base_opt)
        opt_s.scale_factor = float(s)
        return opt_s

    loaders = {}
    iters = {}
    metas = {}

    for s in mixed_scales:
        opt_s = _opt_for_scale(opt, s)
        ds = HSI_DatasetAnyScale(opt_s, mode='train')
        dl = DataLoader(ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        loaders[s] = dl
        iters[s] = iter(dl)
        metas[s] = (ds.lrhsi_list[0].shape[2], ds.hrmsi_list[0].shape[2])

    # load stage1 blind nets
    blind_nets = {}
    psf_mismatch_mode = CLI_OVERRIDES["psf_mismatch_mode"]
    for s in mixed_scales:
        hsi_c, msi_c = metas[s]
        bn = DME_NETAnyScale(
            hsi_channels=hsi_c,
            msi_channels=msi_c,
            ratio=float(s),
            randomize_sigma=False,
            sigma_jitter=0.0,
        ).to(device)
        bn.requires_grad_(False)

        scale_tag = _format_scale_tag(s)
        run_suffix_s = _build_stage1_suffix(opt, scale_tag)
        ckpt_dir_s = os.path.join(opt.checkpoints_dir, f"{opt.dataset_name}_{run_suffix_s}")
        ckpt_path_s = os.path.join(ckpt_dir_s, 'checkpoint_first_stage_final.pth')

        if not os.path.exists(ckpt_path_s):
            print(f"\n[ERROR] 未找到倍率={s} 的第一阶段 checkpoint：{ckpt_path_s}")
            print("请先分别训练好这些倍率的第一阶段，并确保目录命名/文件名与当前代码一致。")
            sys.exit(0)

        print(f"[Stage2-Mixed] 加载第一阶段 ckpt | scale={s} | {ckpt_path_s}")
        checkpoint = torch.load(ckpt_path_s, map_location=device)
        _load_blind_net_state_dict_with_psf_fallback(
            bn, checkpoint, device, s, mismatch_mode=psf_mismatch_mode
        )
        blind_nets[s] = bn

    # main fusion model
    base_hsi_c, base_msi_c = metas[mixed_scales[0]]
    model = create_model(
        model_name=opt.second_stage_model,
        hsi_channels=base_hsi_c,
        msi_channels=base_msi_c,
        device=device,
        scale_factor=float(opt.scale_factor),
    )

    total_params = count_parameters(model)
    print(f'模型的总参数量: {total_params:,} 参数')
    print(f'使用的模型: {opt.second_stage_model}')
    print(f'混合训练倍率: {mixed_scales}')
    if CLI_OVERRIDES["scale_range"] is not None and CLI_OVERRIDES["mixed_scales"] is None:
        smin, smax, sstep = CLI_OVERRIDES["scale_range"]
        print(f'倍率范围模式: [{smin}, {smax}] step={sstep} (需准备对应每个倍率的 stage1 ckpt)')
    print(f'use_supervised(opt): {getattr(opt, "use_supervised", 0)} (本脚本强制无监督训练)')

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_stage2, betas=(opt.beta1, opt.beta2))
    criterion_l1 = nn.L1Loss().to(device)
    criterion_mse = nn.MSELoss().to(device)

    # ---- weights (supervised block stays off by default) ----
    grad_w = getattr(opt, "grad_w", 0.02)
    mse_w = getattr(opt, "mse_w", 0.08)
    charb_w = getattr(opt, "charb_w", 0.05)
    charb_eps = getattr(opt, "charb_eps", 1e-3)
    ms_w = getattr(opt, "ms_w", 0.03)
    sam_w = getattr(opt, "sam_w", 0.02)

    # ---- 2026DC weights (can be overridden by CLI_OVERRIDES) ----
    sc_w = CLI_OVERRIDES["sc_w"] if CLI_OVERRIDES["sc_w"] is not None else getattr(opt, "sc_w", 0.03)
    spec_w = CLI_OVERRIDES["spec_w"] if CLI_OVERRIDES["spec_w"] is not None else getattr(opt, "spec_w", 0.02)
    scspec_warm = CLI_OVERRIDES["scspec_warmup_epochs"] if CLI_OVERRIDES["scspec_warmup_epochs"] is not None else getattr(opt, "scspec_warmup_epochs", 300)
    freq_w = CLI_OVERRIDES["freq_w"] if CLI_OVERRIDES["freq_w"] is not None else getattr(opt, "freq_w", 0.02)
    freq_keep = CLI_OVERRIDES["freq_keep"] if CLI_OVERRIDES["freq_keep"] is not None else getattr(opt, "freq_keep", 0.15)

    sam_hsi_w = CLI_OVERRIDES.get("sam_hsi_w", None)
    sam_hsi_w = sam_hsi_w if sam_hsi_w is not None else getattr(opt, "sam_hsi_w", 0.02)

    clamp_01 = bool(CLI_OVERRIDES.get("clamp_01", False) or getattr(opt, "clamp_01", False))

    use_supervised = False
    if getattr(opt, "use_supervised", 0):
        print("[WARN] use_supervised=1，但当前脚本已强制无监督训练（合成GT仅用于评估）。")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.num_epochs_second_stage,
        eta_min=opt.lr_stage2 * 0.01
    )

    # eval
    eval_scale = CLI_OVERRIDES["eval_scale"]
    if eval_scale is None:
        eval_scale = float(opt.scale_factor)

    if _contains_scale(mixed_scales, eval_scale):
        eval_scale = next(s for s in mixed_scales if abs(float(s) - float(eval_scale)) <= 1e-8)
        eval_batch = next(iter(loaders[eval_scale]))
    else:
        print(
            f"[WARN] eval_scale={eval_scale} 不在 mixed_scales={mixed_scales} 中，将单独构建 eval 数据集。"
        )
        opt_eval = copy.deepcopy(opt)
        opt_eval.scale_factor = float(eval_scale)
        eval_ds = HSI_DatasetAnyScale(opt_eval, mode='train')
        eval_dl = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=opt.num_workers)
        eval_batch = next(iter(eval_dl))

    print(f'评估倍率: {eval_scale}')
    eval_lrhsi = eval_batch['lrhsi'].to(device)
    eval_hrmsi = eval_batch['hrmsi'].to(device)
    eval_target = eval_batch['hsi'].to(device)

    metrics_history = {'psnr': [], 'sam': [], 'rmse': [], 'ergas': [], 'cc': [], 'epochs': []}

    # -------------------------
    # Loss helpers

    def _maybe_clamp01(x):
        if not clamp_01:
            return x
        return torch.clamp(x, 0.0, 1.0)

    # -------------------------
    # Loss helpers
    # -------------------------
    def _sobel_kernels(dtype, device_):
        kx = torch.tensor([[1.0, 0.0, -1.0],
                           [2.0, 0.0, -2.0],
                           [1.0, 0.0, -1.0]], dtype=dtype, device=device_)
        ky = torch.tensor([[1.0, 2.0, 1.0],
                           [0.0, 0.0, 0.0],
                           [-1.0, -2.0, -1.0]], dtype=dtype, device=device_)
        return kx, ky

    def _gradient_loss(pred, target):
        _, c, _, _ = pred.shape
        kx, ky = _sobel_kernels(pred.dtype, pred.device)
        kx = kx.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        ky = ky.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        pred_gx = F.conv2d(pred, kx, padding=1, groups=c)
        pred_gy = F.conv2d(pred, ky, padding=1, groups=c)
        tgt_gx = F.conv2d(target, kx, padding=1, groups=c)
        tgt_gy = F.conv2d(target, ky, padding=1, groups=c)
        return criterion_l1(pred_gx, tgt_gx) + criterion_l1(pred_gy, tgt_gy)

    def _charbonnier_loss(pred, target, eps):
        return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

    def _multiscale_loss(pred, target, scales=(0.5, 0.25)):
        loss = 0.0
        for s in scales:
            pred_s = F.interpolate(pred, scale_factor=s, mode='bicubic', align_corners=False)
            tgt_s = F.interpolate(target, scale_factor=s, mode='bicubic', align_corners=False)
            loss = loss + criterion_l1(pred_s, tgt_s)
        return loss / float(len(scales))

    def _spectral_angle_loss(pred, target, eps=1e-8):
        b, c, h, w = pred.shape
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, c)
        tgt_flat = target.permute(0, 2, 3, 1).reshape(-1, c)
        pred_norm = torch.norm(pred_flat, dim=1, keepdim=True)
        tgt_norm = torch.norm(tgt_flat, dim=1, keepdim=True)
        cos = (pred_flat * tgt_flat).sum(dim=1, keepdim=True) / (pred_norm * tgt_norm + eps)
        cos = torch.clamp(cos, -1.0, 1.0)
        return torch.mean(1.0 - cos)

    def _fft_highfreq_l1(pred, target, keep=0.15, eps=1e-8):
        """FFT high-frequency amplitude L1. pred/target: [B,C,H,W]"""
        B, C, H, W = pred.shape
        fp = torch.fft.fftshift(torch.fft.fft2(pred, norm="ortho"), dim=(-2, -1))
        ft = torch.fft.fftshift(torch.fft.fft2(target, norm="ortho"), dim=(-2, -1))

        mp = torch.sqrt(fp.real ** 2 + fp.imag ** 2 + eps)
        mt = torch.sqrt(ft.real ** 2 + ft.imag ** 2 + eps)

        cy, cx = H // 2, W // 2
        ry = int((1.0 - keep) * H / 2)
        rx = int((1.0 - keep) * W / 2)

        mask = torch.ones((H, W), device=pred.device, dtype=pred.dtype)
        mask[cy - ry: cy + ry, cx - rx: cx + rx] = 0.0
        mask = mask.view(1, 1, H, W)

        return torch.mean(torch.abs(mp * mask - mt * mask))

    iters_per_epoch = int(np.mean([len(dl) for dl in loaders.values()]))

    for epoch in range(opt.num_epochs_second_stage):
        model.train()
        epoch_loss = 0.0
        num_iter = 0
        start_time = time.time()

        last_lv = last_hv = last_sc = last_spec = last_sam_hsi = last_freq = None

        # warmup factor for sc/spec
        if scspec_warm and scspec_warm > 0:
            w_scale = min(1.0, float(epoch + 1) / float(scspec_warm))
        else:
            w_scale = 1.0

        for _ in range(iters_per_epoch):
            scale = random.choice(mixed_scales)
            bn = blind_nets[scale]

            try:
                batch = next(iters[scale])
            except StopIteration:
                iters[scale] = iter(loaders[scale])
                batch = next(iters[scale])

            lrhsi = batch['lrhsi'].to(device)
            hrmsi = batch['hrmsi'].to(device)
            if use_supervised:
                target = batch['hsi'].to(device)

            target_hw = (lrhsi.shape[-2], lrhsi.shape[-1])
            lrlrhsi_target_hw = None
            if 'lrlrhsi' in batch:
                lrlrhsi_target_hw = (batch['lrlrhsi'].shape[-2], batch['lrlrhsi'].shape[-1])

            with torch.no_grad():
                lrmsi = bn.forward_using_psf(hrmsi, target_hw=target_hw)
                lrlrhsi = bn.forward_using_psf(lrhsi, target_hw=lrlrhsi_target_hw)

            # low-variance cycle: (lrlrhsi, lrmsi) -> lrhsi
            rec_lrhsi_lv = model(lrlrhsi, lrmsi)
            loss_lv = criterion_l1(rec_lrhsi_lv, lrhsi)

            # high-variance cycle (lrhsi, hrmsi)
            rec_hsi = model(lrhsi, hrmsi)
            rec_lrhsi_hv = bn.forward_using_psf(rec_hsi, target_hw=target_hw)
            rec_hrmsi_hv = bn.forward_using_srf(rec_hsi)
            loss_hv = criterion_l1(rec_lrhsi_hv, lrhsi) + criterion_l1(rec_hrmsi_hv, hrmsi)

            total_loss = loss_lv + loss_hv

            # -----------------------------
            # Dual-Consistency regularizers
            # 1) Scale-consistency: rec_hsi -> degrade -> re-fuse should be consistent
            rec_hsi_sc = model(rec_lrhsi_hv, hrmsi)
            loss_sc = criterion_l1(rec_hsi_sc, rec_hsi.detach())

            # 2) Spectral-consistency in MSI space (SAM): SRF(rec_hsi) should align with hrmsi
            lrhsi_c = _maybe_clamp01(lrhsi)
            rec_lrhsi_hv_c = _maybe_clamp01(rec_lrhsi_hv)
            rec_hrmsi_hv_c = _maybe_clamp01(rec_hrmsi_hv)
            hrmsi_c = _maybe_clamp01(hrmsi)

            loss_spec = _spectral_angle_loss(rec_hrmsi_hv_c, hrmsi_c)

            # 2b) HSI-space SAM self-supervised: SAM(D_s(rec_hsi), lrhsi)
            loss_sam_hsi = _spectral_angle_loss(rec_lrhsi_hv_c, lrhsi_c)

            # 3) Frequency-consistency in MSI space: high-frequency amplitude alignment
            loss_freq = _fft_highfreq_l1(rec_hrmsi_hv_c, hrmsi_c, keep=float(freq_keep))

            total_loss = (total_loss
                          + (w_scale * sc_w) * loss_sc
                          + (w_scale * spec_w) * loss_spec
                          + (w_scale * float(sam_hsi_w)) * loss_sam_hsi
                          + float(freq_w) * loss_freq)

            last_lv = loss_lv.detach()
            last_hv = loss_hv.detach()
            last_sc = loss_sc.detach()
            last_spec = loss_spec.detach()
            last_sam_hsi = loss_sam_hsi.detach()
            last_freq = loss_freq.detach()

            # ---- optional supervised terms (off by default) ----
            if use_supervised:
                loss_mse = criterion_mse(rec_hsi, target)
                loss_grad = _gradient_loss(rec_hsi, target)
                loss_charb = _charbonnier_loss(rec_hsi, target, charb_eps)
                loss_ms = _multiscale_loss(rec_hsi, target)
                loss_sam = _spectral_angle_loss(rec_hsi, target)
                total_loss = total_loss + (
                    mse_w * loss_mse
                    + grad_w * loss_grad
                    + charb_w * loss_charb
                    + ms_w * loss_ms
                    + sam_w * loss_sam
                )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            num_iter += 1

        avg_loss = epoch_loss / max(num_iter, 1)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f'第二阶段(2026DC-v3) Epoch [{epoch+1}/{opt.num_epochs_second_stage}] '
            f'Loss: {avg_loss:.4f} LR: {current_lr:.6f} w_scale: {w_scale:.3f} '
            f'lv:{(last_lv.item() if last_lv is not None else float("nan")):.4f} '
            f'hv:{(last_hv.item() if last_hv is not None else float("nan")):.4f} '
            f'sc:{(last_sc.item() if last_sc is not None else float("nan")):.4f} '
            f'spec:{(last_spec.item() if last_spec is not None else float("nan")):.4f} '
            f'samH:{(last_sam_hsi.item() if last_sam_hsi is not None else float("nan")):.4f} '
            f'freq:{(last_freq.item() if last_freq is not None else float("nan")):.4f} '
            f'Time: {time.time() - start_time:.2f}s'
        )

        if (epoch + 1) % 100 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                rec_hsi_eval = model(eval_lrhsi, eval_hrmsi)
                metrics_1 = calculate_metrics(eval_target, rec_hsi_eval, scale_factor=eval_scale)

            metrics_history['psnr'].append(metrics_1["psnr"])
            metrics_history['sam'].append(metrics_1["sam"])
            metrics_history['rmse'].append(metrics_1["rmse"])
            metrics_history['ergas'].append(metrics_1["ergas"])
            metrics_history['cc'].append(metrics_1["cc"])
            metrics_history['epochs'].append(epoch + 1)

            print(f'[Eval@scale={eval_scale}] PSNR: {metrics_1["psnr"]:.2f} | SAM: {metrics_1["sam"]:.4f} | RMSE: {metrics_1["rmse"]:.4f} | ERGAS: {metrics_1["ergas"]:.4f} | CC: {metrics_1["cc"]:.4f}')
            save_results(rec_hsi_eval, eval_target, opt, epoch, stage=2)

    save_checkpoint(model, optimizer, opt, 'checkpoint_second_stage_final.pth')
    save_checkpoint(model, optimizer, opt, f'checkpoint_{opt.second_stage_model}_final.pth')
    plot_metrics_with_epochs(metrics_history, results_dir, 2, opt.dataset_name)


# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    opt = get_options()
    stage = opt.stage if hasattr(opt, 'stage') else 1

    if hasattr(opt, 'analyse') and opt.analyse:
        analyse_frequency(opt)
    elif stage == 1:
        print("开始第一阶段训练...")
        train_first_stage()
    else:
        print("开始第二阶段训练...")
        train_second_stage()
