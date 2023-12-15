# create_data.py 创建数据集
import datetime
from data_provider.func_util import *
from data_provider.func_stock import *
from data_provider.func_RankGauss import *

from pyti.chaikin_money_flow import chaikin_money_flow as cmf
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.relative_strength_index import  relative_strength_index as rsi
from pyti.stochastic import percent_k as kdj_k
from pyti.stochastic import percent_d as kdj_d

def creat_data(stock_num,config):
    '''创建模型合适的数据集train_loader, valid_loader,test_loader'''
    NUM = stock_num
    time_now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_') 

    '''拼接前原始数据处理'''
    print(f'股票代码为{NUM}的数据正在下载中....')
    gp_data = get_data(NUM, config['fuquan'], config['start_date'], config['end_date'] ) # 获取原始数据
    print('原始数据下载完成,形状为:',gp_data.shape)
    close_data = gp_data['Close'].values
    open_data = gp_data['Open'].values
    high_data = gp_data['High'].values
    low_data = gp_data['Low'].values
    volume_data = gp_data['Volume'].values
    '''
    计算指标'''
    macd_data = macd(close_data,12,26)
    # cmf_data = cmf(close_data, high_data, low_data, volume_data, 20)
    rsi_data = rsi(close_data, 14)
    kdj_k_data = kdj_k(close_data, 9)
    kdj_d_data = kdj_d(close_data, 9)
    twf_feat_data = twf_feat(close_data, high_data, low_data, volume_data, 21)
    '''
    将指标添加到原始数据中'''
    gp_data['macd'] = macd_data
    # gp_data['cmf'] = cmf_data
    gp_data['rsi'] = rsi_data
    gp_data['kdj_k'] = kdj_k_data - kdj_d_data
    gp_data['kdj_d'] = kdj_d_data
    gp_data['twf_feat'] = twf_feat_data
    gp_data['K_valume'] = volume_data/1000
    print('\n指标计算完成正在添加指标到data....')
    '''
    添加预测目标lable列,并将所有的nan值填充为0'''
    lable_data = add_lable(gp_data,config['zhangfu'],  config['lable_n'], config['lable_ch'])
    '''
    删除第一列date列,删除前26行因为macd和cmf是空值,
    删除最后config['feat_n']行因为未来lable的n天的数据是空值,影响训练'''
    lable_data = lable_data.iloc[26:-config['lable_n'], 1:]
    '''
    手动选择拼接前原始数据列'''
    final_data = lable_data[config['final_data_feat']].copy() 
    print('拼接前data形状: ',final_data.shape)
    print('拼接前数据有: ',config['final_data_feat'])
    '''
    如果标签为True,那么就对final_data的change,twf列进行归一化'''
    if config['feat_normalize'] == 1:
        for i in config['feat_nor_lise']:
            final_data[i] = normalize(final_data[i])
        print('拼接前对数据进行归一化处理')
    elif config['feat_normalize'] == 2:
        for i in config['feat_nor_lise']:
            final_data[i] = Standardization(final_data[i])
        print('拼接前对数据进行标准化处理')
    elif config['feat_normalize'] == 3:
        for i in config['feat_nor_lise']:
            final_data[i] = QTF(final_data[i])
        print('拼接前对数据进行均匀分布转换QuantileTransformer处理')
    elif config['feat_normalize'] == 4:
        for i in config['feat_nor_lise']:
            rg_data = final_data[i].values
            Rank = RankGauss_norm()
            final_data[i] = Rank.fit_transform(rg_data)
        print('拼接前对数据进行高斯标准化Rank Gaussian Normalization处理')
    elif config['feat_normalize'] == 5:
        for i in config['feat_nor_lise']:
            final_data[i] = PowerTransformer(final_data[i])
        print('拼接前对数据进行高斯分布转换PowerTransformer处理')
    elif config['feat_normalize'] == 6:
        for i in config['feat_nor_lise']:
            final_data[i] = RobustScaler(final_data[i])
        print('拼接前对数据进行鲁棒缩放RobustScaler处理')
    
    return final_data

def creat_test_data(num):
    # 创建数据
    data_len = num
    t = np.linspace(0, 12*np.pi, data_len) # 创建等差数列
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    # 查看sin_t里面数据的类型
    print(type(sin_t[0]))
    
    dataset = np.zeros((data_len, 7)) # 创建一个200*2的零矩阵
    dataset[:,0] = sin_t
    dataset[:,1] = cos_t
    df11 = np.roll(dataset[:,0],-1) 
    df12 = np.roll(dataset[:,0],-2)
    df21 = np.roll(dataset[:,1],-1)
    df22 = np.roll(dataset[:,1],-2)
    dataset[:,2] = df11*2 + df21*3
    dataset[:,3] = df22*3 + df12*2
    dataset[:,4] = dataset[:,2]*dataset[:,3]
    dataset[:,5] = dataset[:,0]*dataset[:,1]
    dataset[:,6] = dataset[:,0]*2+dataset[:,1]*3+dataset[:,2]*0.5+dataset[:,3]*0.4+dataset[:,4]*2 + dataset[:,5]*3

    # dataset[:,4] = df5
    #删除最后一行和第一行
    dataset = np.delete(dataset, 0, axis=0)
    dataset = np.delete(dataset, -1, axis=0)
    # 打印dataset的类型
    print(type(dataset))
    dataset = pd.DataFrame(dataset)
    return dataset

def creat_test_data2(dataframe, periods=[365.25, 182.625, 91.3125, 45.65625, 22.828125]):
    import random
    clip_val = random.uniform(0.3, 1)

    period = random.choice(periods)

    phase = random.randint(-1000, 1000)

    dataframe["views"] = dataframe.apply(
        lambda x: np.clip(
            np.cos(x["index"] * 2 * np.pi / period + phase), -clip_val, clip_val
        )
        * x["amplitude"]
        + x["offset"],
        axis=1,
    ) + np.random.normal(
        0, dataframe["amplitude"].abs().max() / 10, size=(dataframe.shape[0],)
    )

    return dataframe