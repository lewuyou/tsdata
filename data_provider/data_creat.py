# create_data.py 创建数据集
import akshare as ak
import datetime
from data_provider.func_util import *
from data_provider.func_stock import *
from data_provider.func_RankGauss import *

from pyti.chaikin_money_flow import chaikin_money_flow as cmf
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.relative_strength_index import  relative_strength_index as rsi
from pyti.stochastic import percent_k as kdj_k
from pyti.stochastic import percent_d as kdj_d

def add_data(raw_data, args):
    '''创建模型合适的数据集train_loader, valid_loader, test_loader'''

    # 提取列数据为NumPy数组
    open_data = raw_data['Open'].values
    close_data = raw_data['Close'].values
    high_data = raw_data['High'].values
    low_data = raw_data['Low'].values
    volume_data = raw_data['Volume'].values

    # 指标计算
    raw_data['macd'] = macd(close_data, 12, 26)
    raw_data['rsi'] = rsi(close_data, 14)
    raw_data['kdj_k'] = kdj_k(close_data, 9)
    raw_data['kdj_d'] = kdj_d(close_data, 9)
    raw_data['twf_feat'] = twf_feat(close_data, high_data, low_data, volume_data, 21)
    raw_data['K_volume'] = volume_data / 1000  # 注意列名修正为 'K_volume'
    print('\n指标计算完成正在添加指标到data....')
    return raw_data

def sub_data(raw_data,args):
    '''
    添加预测目标lable列,并将所有的nan值填充为0'''
    lable_data = add_lable(raw_data,args.zhangfu,args.lable_n, args.lable_ch)
    '''
    删除第一列date列,删除前26行因为macd和cmf是空值,
    删除最后config['feat_n']行因为未来lable的n天的数据是空值,影响训练'''
    lable_data = lable_data.iloc[26:-args.lable_n, 1:]
    '''
    手动选择拼接前原始数据列'''
    final_data = lable_data[args.final_data_feat].copy() 
    print('拼接前data形状: ',final_data.shape)
    print('拼接前数据有: ',args.final_data_feat)
    '''
    如果标签为True,那么就对final_data的change,twf列进行归一化'''
    if args.feat_normalize == 1:
        for i in args.feat_nor_lise:
            final_data[i] = normalize(final_data[i])
        print('拼接前对数据进行归一化处理')
    elif args.feat_normalize == 2:
        for i in args.feat_nor_lise:
            final_data[i] = Standardization(final_data[i])
        print('拼接前对数据进行标准化处理')
    elif args.feat_normalize == 3:
        for i in args.feat_nor_lise:
            final_data[i] = QTF(final_data[i])
        print('拼接前对数据进行均匀分布转换QuantileTransformer处理')
    elif args.feat_normalize == 4:
        for i in args.feat_nor_lise:
            rg_data = final_data[i].values
            Rank = RankGauss_norm()
            final_data[i] = Rank.fit_transform(rg_data)
        print('拼接前对数据进行高斯标准化Rank Gaussian Normalization处理')
    elif args.feat_normalize == 5:
        for i in args.feat_nor_lise:
            final_data[i] = PowerTransformer(final_data[i])
        print('拼接前对数据进行高斯分布转换PowerTransformer处理')
    elif args.feat_normalize == 6:
        for i in args.feat_nor_lise:
            final_data[i] = RobustScaler(final_data[i])
        print('拼接前对数据进行鲁棒缩放RobustScaler处理')

def add_label(data, args):
    '''原始数据计算并添加预测标签'''
    data['Tom_Chg'] = (data['Close'].shift(-args.label_n) - data['Close'])/ data['Close']# 计算第n天到今天收益率
    if args.label_ch:
        # 初始化OT列为0
        data['OT'] = 0
        # 如果第n天到今天收益率data['Tom_Chg']大于0.5，那么label就等于1
        data.loc[data['Tom_Chg'] >= args.zhangfu, 'OT'] = 1
    else:
        data['OT'] = data['Tom_Chg']
    data = data.fillna(0)#将数据中的nan替换为0
    return data

def download_data(num,args):
    '''下载指定时间段数据'''
    if args.start_date == '0':
        args.num_start_data = ak.stock_individual_info_em(symbol=num).iat[3,1]    # 开始时间选取公司信息里面的上市时间
    else:
        args.num_start_data = args.start_date
    if args.end_date == '0':
        args.num_end_data = (datetime.datetime.now()).strftime("%Y%m%d")   # 结束选择今天
    elif args.end_date == '-1':
        args.num_end_data = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%Y%m%d")    # 结束选择昨天
    else:
        args.num_end_data = args.end_date
    df_colum = ak.stock_zh_a_hist(symbol=num, period=args.period, start_date= args.num_start_data, end_date= args.num_end_data, adjust=args.fuquan)
    df_resault = df_colum[['日期', '开盘', '收盘', '最高', '最低', '成交量', '振幅', '涨跌幅', '换手率']].copy()    # 只获取其中需要的数据
    df_resault.columns = ['date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Amplitude', 'Change', 'Turnover']    # 重命名表头
    print(f'获取数据时间为： {args.num_start_data} - {args.num_end_data}')
    print(f'数据形状： {df_resault.shape}')
    return df_resault
