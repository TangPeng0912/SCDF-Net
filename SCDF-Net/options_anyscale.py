import argparse

def get_options():
    parser = argparse.ArgumentParser()
    
    # 数据集参数
    parser.add_argument('--dataset_name', type=str, default='paviau', 
                      choices=['Chikusei', 'houston18', 'paviau', 'wadc'],
                      help='数据集名称')
    parser.add_argument('--srf_name', type=str, default='landsat',
                      help='SRF文件名')
    parser.add_argument('--scale_factor', type=float, default=4.0,
                      help='下采样因子（支持非整数倍率）')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1,
                      help='批次大小')
    parser.add_argument('--num_epochs_first_stage', type=int, default=100,
                      help='第一阶段训练轮数')
    parser.add_argument('--num_epochs_second_stage', type=int, default=100,
                      help='第二阶段训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='默认学习率')
    parser.add_argument('--lr_stage1', type=float, default=None,
                      help='第一阶段学习率，如果未指定则使用默认学习率')
    parser.add_argument('--lr_stage2', type=float, default=None,
                      help='第二阶段学习率，如果未指定则使用默认学习率')
    parser.add_argument('--beta1', type=float, default=0.5,
                      help='Adam优化器参数beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                      help='Adam优化器参数beta2')
    parser.add_argument('--constraint_warmup_epochs', type=int, default=0,
                      help='物理约束权重warmup的epoch数，0表示自动取20%')
    parser.add_argument('--constraint_w_min', type=float, default=0.02,
                      help='物理约束最小权重')
    parser.add_argument('--constraint_w_max', type=float, default=0.12,
                      help='物理约束最大权重')
    parser.add_argument('--mse_w', type=float, default=0.08,
                      help='第二阶段MSE损失权重')
    parser.add_argument('--grad_w', type=float, default=0.02,
                      help='第二阶段梯度损失权重')
    parser.add_argument('--charb_w', type=float, default=0.05,
                      help='第二阶段Charbonnier损失权重')
    parser.add_argument('--charb_eps', type=float, default=1e-3,
                      help='第二阶段Charbonnier的epsilon')
    parser.add_argument('--ms_w', type=float, default=0.03,
                      help='第二阶段多尺度一致性损失权重')
    parser.add_argument('--use_supervised', type=int, default=0,
                      help='第二阶段是否启用监督损失: 1=启用, 0=关闭')
    
    parser.add_argument('--sam_w', type=float, default=0.02)
    parser.add_argument('--sc_w', type=float, default=0.03,
                      help='DualConsistent: scale-consistency weight (2026 upgrade)')
    parser.add_argument('--spec_w', type=float, default=0.02,
                      help='DualConsistent: spectral-consistency (MSI-space SAM) weight (2026 upgrade)')
    
    # 模型参数
    parser.add_argument('--second_stage_model', type=str, default='SFFO_NET_DC',
                      choices=[ 'SFFO_NET_DC'],
                      help='第二阶段使用的模型')
    
    # 路径参数
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                      help='模型保存路径')
    parser.add_argument('--results_dir', type=str, default='./results',
                      help='结果保存路径')
    parser.add_argument('--suffix', type=str, default='',
                      help='保存路径的后缀名')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                      help='数据加载线程数')
    parser.add_argument('--device', type=str, default='cuda',
                      help='训练设备')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                      help='训练阶段: 1=第一阶段, 2=第二阶段')
    parser.add_argument('--analyse', action='store_true',
                      help='是否进行频率分析')
    
    opt = parser.parse_args()

    # 不再自动追加 fsii_up 后缀；仅当未指定时提供默认
    if not opt.suffix:
        opt.suffix = 'fsii_up'
    
    # 如果未指定阶段特定学习率，则使用默认学习率
    if opt.lr_stage1 is None:
        opt.lr_stage1 = opt.lr
    if opt.lr_stage2 is None:
        opt.lr_stage2 = opt.lr
        
    return opt
