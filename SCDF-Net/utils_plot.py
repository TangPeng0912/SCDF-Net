import os
import matplotlib.pyplot as plt


def plot_metrics_with_epochs(metrics_history, save_dir, stage, dataset_name, result_type=None):
    """
    绘制评价指标随迭代次数变化的折线图（使用真实记录的 epochs）

    Args:
        metrics_history (dict): 包含各个指标历史的字典，需包含 'epochs'
        save_dir (str): 保存图像的目录
        stage (int): 训练阶段（1或2）
        dataset_name (str): 数据集名称
        result_type (str, optional): 结果类型，用于第一阶段训练（'from_hsi'或'from_msi'）
    """
    if stage is not None:
        save_dir = os.path.join(save_dir, f"stage_{stage}")
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = metrics_history.get('epochs')
    if not epochs:
        # 回退到与旧实现一致的轴
        epochs = list(range(0, len(metrics_history['psnr']) * 100, 100))

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
        ax.plot(epochs, metrics_history[metric_name], style, linewidth=2, markersize=6)
        ax.set_title(f'{metric_name.upper()} / epoch', fontsize=14)
        ax.set_xlabel('epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if result_type:
        save_path = os.path.join(save_dir, f'metrics_stage{stage}_{result_type}.png')
    else:
        save_path = os.path.join(save_dir, f'metrics_stage{stage}.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"评价指标折线图已保存至: {save_path}")
    plt.close(fig)
