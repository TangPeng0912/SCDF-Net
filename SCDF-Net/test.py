import os
import sys
import torch
from torch.utils.data import DataLoader
from scipy.io import savemat

from dataset import HSI_DatasetAnyScale
from utils import calculate_metrics
from options_anyscale import get_options
from train import (
    create_model,
    set_random_seed,
    _make_run_suffix,
    _normalize_state_dict_keys,
)


def test():
    # 获取参数（对齐 test.py 框架）
    opt = get_options()
    if getattr(opt, "random_scale", False) and opt.batch_size != 1:
        raise ValueError("启用随机缩放时请设置 batch_size=1，以避免不同分辨率样本无法拼接。")

    # 设置随机种子
    set_random_seed(opt.seed if hasattr(opt, 'seed') else 42)

    # 创建保存目录
    run_suffix = _make_run_suffix(opt)
    results_dir = os.path.join(opt.results_dir, f"{opt.dataset_name}_{run_suffix}")
    os.makedirs(results_dir, exist_ok=True)

    # 设置设备
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    test_dataset = HSI_DatasetAnyScale(opt, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    # 创建模型
    model = create_model(
        model_name=opt.second_stage_model,
        hsi_channels=test_dataset.lrhsi_list[0].shape[2],
        msi_channels=test_dataset.hrmsi_list[0].shape[2],
        device=device,
        scale_factor=opt.scale_factor,
    )

    # 加载模型权重
    checkpoint_dir = os.path.join(opt.checkpoints_dir, f"{opt.dataset_name}_{run_suffix}")
    candidate_paths = [
        os.path.join(checkpoint_dir, "checkpoint_second_stage_final.pth"),
        os.path.join(checkpoint_dir, f"checkpoint_{opt.second_stage_model}_final.pth"),
        os.path.join(opt.checkpoints_dir, f"{opt.dataset_name}_{opt.suffix}", "checkpoint_second_stage_final.pth"),
        os.path.join(opt.checkpoints_dir, f"{opt.dataset_name}_{opt.suffix}", f"checkpoint_{opt.second_stage_model}_final.pth"),
    ]
    checkpoint_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if checkpoint_path is None:
        print("错误: 未找到模型检查点，候选路径为：")
        for p in candidate_paths:
            print(f"- {p}")
        sys.exit(1)

    print(f"加载模型检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = _normalize_state_dict_keys(state_dict)
    # Backward-compat for old SFFO_NET alpha parameter
    if "cfo.alpha" in state_dict and "cfo._alpha_logit" not in state_dict:
        alpha = state_dict.pop("cfo.alpha")
        alpha_clamped = torch.clamp(alpha, 1e-4, 1.0 - 1e-4)
        state_dict["cfo._alpha_logit"] = torch.log(alpha_clamped / (1.0 - alpha_clamped))
    model.load_state_dict(state_dict, strict=True)

    # 测试
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 获取数据
            lrhsi = batch['lrhsi'].to(device)
            hrmsi = batch['hrmsi'].to(device)
            target = batch['hsi'].to(device)

            # 前向传播
            pred = model(lrhsi, hrmsi)
            metrics = calculate_metrics(pred, target)

            # 打印评估结果
            print(f"[{batch_idx + 1}/{len(test_loader)}] 测试结果:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

            # 保存为mat格式（对齐 test.py）
            save_dir = os.path.join(results_dir, 'test')
            os.makedirs(save_dir, exist_ok=True)

            pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)      # [B, H, W, C]
            target_np = target.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
            lrhsi_np = lrhsi.cpu().numpy().transpose(0, 2, 3, 1)    # [B, H, W, C]
            hrmsi_np = hrmsi.cpu().numpy().transpose(0, 2, 3, 1)    # [B, H, W, C]

            savemat(
                os.path.join(save_dir, 'test_result_all_in_one.mat'),
                {
                    'rec_hrhsi': pred_np[0],
                    'hsi': target_np[0],
                    'lrhsi': lrhsi_np[0],
                    'hrmsi': hrmsi_np[0],
                    'metrics': metrics
                }
            )

            mat_path = os.path.join(save_dir, 'test_result_only_rec_hrhsi.mat')
            savemat(mat_path, {'rec_hrhsi': pred_np[0]})
            print(f"结果已保存到: {mat_path}")


if __name__ == '__main__':
    test()
