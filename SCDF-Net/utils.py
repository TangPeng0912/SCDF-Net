import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
import random


def calculate_metrics(pred, target, scale_factor=1.0):
    """计算评估指标"""
    psnr_value = calculate_psnr(target, pred)
    sam_value = calculate_sam(target, pred)
    rmse_value = calculate_rmse(target, pred)
    ergas_value = calculate_ergas(pred, target, scale_factor=scale_factor)
    cc_value = calculate_cc(target, pred)
    
    return {
        'psnr': psnr_value,
        'sam': sam_value,
        'rmse': rmse_value,
        'ergas': ergas_value,
        'cc': cc_value
    }


def calculate_psnr(target, pred):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    与原有代码一致，计算每个波段PSNR的平均
    """
    # 获取波段数和 batch size
    num_bands = pred.shape[1]
    batch_size = pred.shape[0]
    
    # 存储每个样本每个波段的 PSNR
    all_band_psnrs = []
    
    # 遍历 batch 中的每个样本
    for b in range(batch_size):
        # 存储当前样本每个波段的 PSNR
        batch_band_psnrs = []
        
        # 遍历每个波段并计算 PSNR
        for band in range(num_bands):
            rec_band = pred[b, band, :, :]
            gt_band = target[b, band, :, :]
            
            # 计算均方误差
            mse = torch.mean((rec_band - gt_band) ** 2)
            
            # 计算 PSNR（假设数据范围[0,1]）
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
            
            batch_band_psnrs.append(psnr.item())
        
        # 计算当前样本所有波段的平均 PSNR
        all_band_psnrs.append(np.mean(batch_band_psnrs))
    
    # 返回所有样本波段 PSNR 的平均值
    return np.mean(all_band_psnrs)


def calculate_sam(target, pred):
    """
    计算SAM (Spectral Angle Mapper)
    与原有代码一致
    """
    # 确保输入是torch.Tensor
    if not isinstance(target, torch.Tensor) or not isinstance(pred, torch.Tensor):
        target = torch.tensor(target)
        pred = torch.tensor(pred)
    
    # 获取batch size
    batch_size = target.shape[0]
    
    # 存储每个样本的SAM
    all_sam = []
    
    # 遍历batch中的每个样本
    for b in range(batch_size):
        # 获取当前样本
        target_sample = target[b]  # [c,h,w]
        pred_sample = pred[b]      # [c,h,w]
        
        # 将[c,h,w]重塑为[c,h*w]，每个像素点作为一个向量
        target_reshaped = target_sample.view(target_sample.shape[0], -1)  # [c,h*w]
        pred_reshaped = pred_sample.view(pred_sample.shape[0], -1)        # [c,h*w]
        
        # 计算点积
        dot_product = torch.sum(target_reshaped * pred_reshaped, dim=0)  # [h*w]
        
        # 计算范数
        target_norm = torch.sqrt(torch.sum(target_reshaped ** 2, dim=0))  # [h*w]
        pred_norm = torch.sqrt(torch.sum(pred_reshaped ** 2, dim=0))      # [h*w]
        
        # 计算余弦角度
        cos_angle = dot_product / (target_norm * pred_norm + 1e-6)
        cos_angle = torch.clamp(cos_angle, -1, 1)
        
        # 计算角度（弧度）
        angle = torch.acos(cos_angle)
        
        # 转换为角度并取平均
        sam = torch.mean(angle) * 180 / torch.pi
        all_sam.append(sam.item())
    
    # 返回所有样本SAM的平均值
    return sum(all_sam) / len(all_sam)


def calculate_rmse(target, pred):
    """
    计算RMSE (Root Mean Square Error)
    与原有代码一致
    """
    # 确保输入是torch.Tensor
    if not isinstance(target, torch.Tensor) or not isinstance(pred, torch.Tensor):
        target = torch.tensor(target)
        pred = torch.tensor(pred)
    
    # 计算每个样本的RMSE
    mse = torch.mean((target - pred) ** 2, dim=[1, 2, 3])  # [b]
    rmse = torch.sqrt(mse)  # [b]
    
    # 返回所有样本RMSE的平均值
    return torch.mean(rmse).item()


def calculate_ergas(target, pred, scale_factor=1.0):
    """
    计算ERGAS (Erreur Relative Globale Adimensionnelle de Synthese)
    ERGAS = 100 / scale_factor * sqrt(mean_bands((RMSE_band / mean_target_band)^2))
    """
    if not isinstance(target, torch.Tensor) or not isinstance(pred, torch.Tensor):
        target = torch.tensor(target)
        pred = torch.tensor(pred)

    scale_factor = float(scale_factor)
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive for ERGAS, got {scale_factor}")

    target = target.float()
    pred = pred.float()
    eps = 1e-10

    mse_per_band = torch.mean((target - pred) ** 2, dim=[2, 3])
    rmse_per_band = torch.sqrt(mse_per_band + eps)
    mean_per_band = torch.mean(target, dim=[2, 3]).abs()

    relative_error = rmse_per_band / (mean_per_band + eps)
    ergas_per_sample = (100.0 / scale_factor) * torch.sqrt(torch.mean(relative_error ** 2, dim=1))
    return torch.mean(ergas_per_sample).item()


def calculate_cc(target, pred):
    """
    计算CC (Correlation Coefficient)，按每个样本每个波段计算后取平均。
    """
    if not isinstance(target, torch.Tensor) or not isinstance(pred, torch.Tensor):
        target = torch.tensor(target)
        pred = torch.tensor(pred)

    target = target.float()
    pred = pred.float()
    eps = 1e-10

    target_centered = target - torch.mean(target, dim=[2, 3], keepdim=True)
    pred_centered = pred - torch.mean(pred, dim=[2, 3], keepdim=True)

    numerator = torch.sum(target_centered * pred_centered, dim=[2, 3])
    denominator = torch.sqrt(
        torch.sum(target_centered ** 2, dim=[2, 3]) *
        torch.sum(pred_centered ** 2, dim=[2, 3]) +
        eps
    )

    cc = numerator / denominator
    cc = torch.clamp(cc, -1.0, 1.0)
    return torch.mean(cc).item()


def save_checkpoint(model, optimizer, opt, filename, scheduler=None):
    """
    保存检查点（兼容原有代码）
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'opt': opt
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 创建包含数据集名称和后缀的目录
    dataset_dir = os.path.join(opt.checkpoints_dir, f"{opt.dataset_name}_{opt.suffix}")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 保存到数据集特定目录
    save_path = os.path.join(dataset_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"检查点已保存到: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, scheduler=None):
    """
    加载检查点（兼容原有代码）
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint['opt']


def save_results(pred, target, opt, epoch, stage=None, result_type=None):
    """
    保存结果（兼容原有代码）
    """
    # 创建包含数据集名称和后缀的目录
    save_dir = os.path.join(opt.results_dir, f"{opt.dataset_name}_{opt.suffix}")
    
    # 如果指定了训练阶段，则创建对应的子目录
    if stage is not None:
        save_dir = os.path.join(save_dir, f"stage_{stage}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算评估指标
    metrics = calculate_metrics(target, pred, scale_factor=getattr(opt, "scale_factor", 1.0))
    
    # 保存所有epoch的metrics到同一个文件
    if result_type is not None:
        metrics_path = os.path.join(save_dir, f'metrics_{result_type}.txt')
        with open(metrics_path, 'a') as f:
            f.write(f'Epoch {epoch}:\n')
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
            f.write('\n')
    else:
        # 如果没有指定result_type，则使用默认的metrics文件
        metrics_path = os.path.join(save_dir, 'metrics.txt')
        with open(metrics_path, 'a') as f:
            f.write(f'Epoch {epoch}:\n')
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
            f.write('\n')
    
    # 只保存最新的结果，覆盖之前的结果
    if result_type is not None:
        # 保存预测结果
        pred_path = os.path.join(save_dir, f'{result_type}_latest.npy')
        np.save(pred_path, pred.cpu().numpy())
        
        # 保存目标结果
        target_path = os.path.join(save_dir, f'{result_type}_target_latest.npy')
        np.save(target_path, target.cpu().numpy())


def plot_metrics(metrics_history, save_dir, stage, dataset_name, result_type=None):
    """
    绘制评价指标折线图（兼容原有代码）
    """
    # 创建保存目录
    if stage is not None:
        save_dir = os.path.join(save_dir, f"stage_{stage}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    # Prefer SimHei for Chinese, fall back to commonly available fonts.
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    metric_styles = {
        'psnr': ('b-o', 'PSNR (dB)'),
        'sam': ('r-o', 'SAM'),
        'rmse': ('g-o', 'RMSE'),
        'ergas': ('m-o', 'ERGAS'),
        'cc': ('c-o', 'CC'),
    }
    metric_names = [name for name in metric_styles if name in metrics_history]
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4 * len(metric_names)))
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metric_names):
        style, ylabel = metric_styles[metric_name]
        epochs = list(range(0, len(metrics_history[metric_name]) * 100, 100))
        ax.plot(epochs, metrics_history[metric_name], style, linewidth=2, markersize=6)
        ax.set_title(f'{metric_name.upper()} / epoch', fontsize=14)
        ax.set_xlabel('epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图像
    if result_type:
        save_path = os.path.join(save_dir, f'metrics_stage{stage}_{result_type}.png')
    else:
        save_path = os.path.join(save_dir, f'metrics_stage{stage}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"评价指标折线图已保存至: {save_path}")
    
    # 关闭图像，释放内存
    plt.close(fig)


def frequency_analyse(lrhsi, target, scale_factor=4, save_dir=None):
    """
    频率分析（兼容原有代码）
    """
    from scipy import fftpack
    
    # 确保输入是torch.Tensor
    if not isinstance(lrhsi, torch.Tensor) or not isinstance(target, torch.Tensor):
        lrhsi = torch.tensor(lrhsi)
        target = torch.tensor(target)
    
    # 将张量转换为numpy数组
    lrhsi_np = lrhsi.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # 获取图像尺寸
    b, c, h, w = lrhsi_np.shape
    _, _, h_target, w_target = target_np.shape
    int_scale = max(1, int(round(scale_factor)))
    
    # 创建结果字典
    results = {}
    
    # 对每个样本进行分析
    for batch_idx in range(b):
        # 选择当前样本
        lrhsi_sample = lrhsi_np[batch_idx]  # [c,h,w]
        target_sample = target_np[batch_idx]  # [c,h,w]
        
        # 计算lrhsi上采样后的图像
        lrhsi_upsampled = np.zeros((c, h_target, w_target))
        for i in range(h):
            for j in range(w):
                lrhsi_upsampled[:, i*int_scale:(i+1)*int_scale,
                                j*int_scale:(j+1)*int_scale] = lrhsi_sample[:, i, j, np.newaxis, np.newaxis]
        
        # 计算差异图像
        diff_image = target_sample - lrhsi_upsampled
        
        # 计算频谱
        lrhsi_spectrum = np.zeros((c, h, w))
        target_spectrum = np.zeros((c, h_target, w_target))
        diff_spectrum = np.zeros((c, h_target, w_target))
        
        # 对每个波段计算频谱
        for band in range(c):
            # 计算lrhsi的频谱
            lrhsi_fft = fftpack.fft2(lrhsi_sample[band])
            lrhsi_fft_shift = fftpack.fftshift(lrhsi_fft)
            lrhsi_spectrum[band] = np.abs(lrhsi_fft_shift)
            
            # 计算target的频谱
            target_fft = fftpack.fft2(target_sample[band])
            target_fft_shift = fftpack.fftshift(target_fft)
            target_spectrum[band] = np.abs(target_fft_shift)
            
            # 计算差异图像的频谱
            diff_fft = fftpack.fft2(diff_image[band])
            diff_fft_shift = fftpack.fftshift(diff_fft)
            diff_spectrum[band] = np.abs(diff_fft_shift)
        
        # 保存结果
        results[f'sample_{batch_idx}'] = {
            'lrhsi_spectrum': lrhsi_spectrum,
            'target_spectrum': target_spectrum,
            'diff_spectrum': diff_spectrum,
        }
        
        # 如果提供了保存目录，则保存图像
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存频谱图
            plt.figure(figsize=(15, 5))
            
            # 选择中间波段进行可视化
            mid_band = c // 2
            
            plt.subplot(131)
            plt.imshow(np.log10(lrhsi_spectrum[mid_band] + 1), cmap='viridis')
            plt.colorbar()
            plt.title('LRHSI Spectrum (log scale)')
            
            plt.subplot(132)
            plt.imshow(np.log10(target_spectrum[mid_band] + 1), cmap='viridis')
            plt.colorbar()
            plt.title('Target Spectrum (log scale)')
            
            plt.subplot(133)
            plt.imshow(np.log10(diff_spectrum[mid_band] + 1), cmap='viridis')
            plt.colorbar()
            plt.title('Difference Spectrum (log scale)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'spectrum_sample_{batch_idx}.png'))
            plt.close()
    
    return results




def set_random_seed(seed):
    """
    设置随机种子（兼容原有代码）
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
