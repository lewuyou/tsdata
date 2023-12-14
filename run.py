import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')  # TimesNet 描述

    # 基本配置
    # 任务类型
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='任务名称，选项：[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    # 设置为训练模式
    parser.add_argument('--is_training', type=int, required=True, default=1, help='状态')
    # 模型名称
    parser.add_argument('--model_id', type=str, required=True, default='test', help='模型 id')
    # 选择模型
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='模型名称，选项：[Autoformer, Transformer, TimesNet]')

    # 数据加载器
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='数据集类型')
    # 数据文件所在文件夹
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='数据文件的根路径')
    # 数据文件全称
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件')
    # 时间特征处理方式
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务，选项：[M, S, MS]; M:多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量')
    # 目标列列名
    parser.add_argument('--target', type=str, default='OT', help='S或MS任务中的目标特征')
    # 时间采集粒度
    parser.add_argument('--freq', type=str, default='d',
                        help='时间特征编码的频率，选项：[s:秒, t:分钟, h:小时, d:天, b:工作日, w:周, m:月], 也可以使用更详细的频率如15min或3h')
    # 模型权重保存文件夹
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点的位置')

    # 预测任务
    # 回顾窗口
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    # 先验序列长度
    parser.add_argument('--label_len', type=int, default=48, help='开始标记长度')
    # 预测窗口长度
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')
    # 季节模式（针对M4数据集）
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='M4数据集的子集')
    parser.add_argument('--inverse', action='store_true', help='反转输出数据', default=False)

    # 插补任务
    # 插补任务中数据丢失率
    parser.add_argument('--mask_rate', type=float, default=0.25, help='掩码比率')

    # 异常检测任务
    # 异常检测中异常点占比
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='先验异常比例 (%)')

    # 模型定义
    # TimesBlock 中傅里叶变换,频率排名前k个周期
    parser.add_argument('--top_k', type=int, default=5, help='用于 TimesBlock')
    # Inception 中卷积核个数
    parser.add_argument('--num_kernels', type=int, default=6, help='用于 Inception')
    # encoder 输入特征数
    parser.add_argument('--enc_in', type=int, default=7, help='编码器输入大小')
    # decoder 输入特征数
    parser.add_argument('--dec_in', type=int, default=7, help='解码器输入大小')
    # 输出通道数
    parser.add_argument('--c_out', type=int, default=7, help='输出大小')
    # 线性层隐含神经元个数
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    # 多头注意力机制
    parser.add_argument('--n_heads', type=int, default=8, help='头数')
    # encoder 层数
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    # decoder 层数
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    # FFN 层隐含神经元个数
    parser.add_argument('--d_ff', type=int, default=2048, help='全连接层维度')
    # 滑动窗口长度
    parser.add_argument('--moving_avg', type=int, default=25, help='移动平均窗口大小')
    # 对 Q 进行采样，对 Q 采样的因子数
    parser.add_argument('--factor', type=int, default=1, help='注意力因子')
    # 是否下采样操作 pooling
    parser.add_argument('--distil', action='store_false',
                        help='是否在编码器中使用蒸馏，使用此参数意味着不使用蒸馏',
                        default=True)
    # dropout 率
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # 时间特征嵌入方式
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码，选项：[timeF, fixed, learned]')
    # 激活函数类型
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    # 是否输出 attention
    parser.add_argument('--output_attention', action='store_true', help='是否在编码器中输出注意力')

    # 优化
    # 并行核心数
    parser.add_argument('--num_workers', type=int, default=10, help='数据加载器的工作数')
    # 实验轮数
    parser.add_argument('--itr', type=int, default=1, help='实验次数')
    # 训练迭代次数
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮数')
    # batch size 大小
    parser.add_argument('--batch_size', type=int, default=32, help='训练输入数据的批量大小')
    # early stopping 机制容忍次数
    parser.add_argument('--patience', type=int, default=3, help='早停耐心')
    # 学习率
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='优化器学习率')
    parser.add_argument('--des', type=str, default='test', help='实验描述')
    # 损失函数
    parser.add_argument('--loss', type=str, default='MSE', help='损失函数')
    # 学习率下降策略
    parser.add_argument('--lradj', type=str, default='type1', help='调整学习率')
    # 使用混合精度训练
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='使用 gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='使用多个 gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多 gpu 的设备 id')

    # 去平稳化投影仪参数
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='投影仪的隐藏层维度（列表）')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='投影仪中的隐藏层数')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('实验中的参数：')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # 实验记录设置
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # 设置实验
            print('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.use_gpu:
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # 设置实验
        print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            torch.cuda.empty_cache()
