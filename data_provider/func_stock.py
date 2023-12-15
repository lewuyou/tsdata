'''股票相关数据处理函数'''
import datetime
import akshare as ak
import pandas as pd

def get_data(num, fuquan, start_date = 0, end_date = 0, time ='daily'):
    '''下载指定时间段数据'''
    if start_date == '0':
        num_start_data = ak.stock_individual_info_em(symbol=num).iat[3,1]    # 开始时间选取公司信息里面的上市时间
    else:
        num_start_data = start_date
    if end_date == '0':
        num_end_data = (datetime.datetime.now()).strftime("%Y%m%d")   # 结束选择今天
    elif end_date == '-1':
        num_end_data = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%Y%m%d")    # 结束选择昨天
    else:
        num_end_data = end_date
    df_colum = ak.stock_zh_a_hist(symbol=num, period=time, start_date= num_start_data, end_date= num_end_data, adjust=fuquan)
    df_resault = df_colum[['日期', '开盘', '收盘', '最高', '最低', '成交量', '振幅', '涨跌幅', '换手率']].copy()    # 只获取其中需要的数据
    df_resault.columns = ['date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Amplitude', 'Change', 'Turnover']    # 重命名表头
    print(f'获取数据时间为： {num_start_data} - {num_end_data}')
    return df_resault

def add_lable(data, zhangfu, lable_n, lable_ch=False):
    '''原始数据计算并添加twf参数和预测标签'''
    data['Tom_Chg'] = (data['Close'].shift(-lable_n) - data['Close'])/ data['Close']# 计算第n天到今天收益率
    if lable_ch:
        data['lable'] = 0
        # 如果第n天到今天收益率data['Tom_Chg']大于0.5，那么lable就等于1
        data.loc[data['Tom_Chg'] >= zhangfu, 'feat'] = 1
    else:
        data['lable'] = data['Tom_Chg']*100
    data = data.fillna(0)#将数据中的nan替换为0
    return data

def data_to_df(data,w,n):
    '''拼接final_data的数据*n'''
    data = pd.DataFrame(data[i*w:(i+n)*w] for i in range(int(len(data)/w)-n+1)) #将一维数组转换成dataframe,每行的数据是data的第i*w个数据到第(i+n)*w个数据
    return data

def del_col(data,w,n):
    '''清理不需要的feat标签列'''
    for i in range(1,n):
        data = data.drop([(i)*w-1],axis=1) # 删除第(i)*w-1列，直到删除完第n列
    return data

def split_xy(data):
    '''将数据拆分出x和y'''
    y =  data[:,-1] # 最后一列是y
    x = data[:,:-1] # 剩下的是x
    return x, y

def twf_feat(close, high, low, valume,period=21):
    ''' 计算twf'''
    data = pd.DataFrame()
    data['Close'] = close
    data['High'] = high
    data['Low'] = low
    data['Volume'] = valume
    data['Close_yday'] = data['Close'].shift(1) # 昨天收盘价
    data['tr_h_max'] = data[['High', 'Close_yday']].max(axis=1) # 计算今天最高价和昨天收盘价的最大值
    data['tr_l_min'] = data[['Low', 'Close_yday']].min(axis=1) # 计算今天最低价和昨天收盘价的最小值
    data['tr_c'] = data['tr_h_max'] - data['tr_l_min'] # 计算真实波动幅度
    data['tr_tmp1'] = data['Close'] - data['tr_l_min'] # 计算Close减去tr_l_min的值
    data['tr_tmp2'] = data['tr_h_max'] - data['Close'] # 计算tr_h_max减去Close的值
    data['adv'] = data['Volume'] * (data['tr_tmp1'] - data['tr_tmp2']) / data['tr_c'].replace(0, 99999999) # 计算Volume*(tr_tmp1-tr_tmp2)/tr_c的值,如果tr_c为0，那就用99999999代替
    data['vol_ma'] = 0
    data['adv_ma'] = 0
    for i in range(1, len(data)):
        data.loc[[i],['vol_ma']] = data['Volume'] + data['vol_ma'].shift(1)*(period-1)/period
    for i in range(1, len(data)):
        data.loc[[i],['adv_ma']] = data['adv'] + data['adv_ma'].shift(1)*(period-1)/period
    data['twf'] = data['adv_ma'] / data['vol_ma'] # 计算twf
    return data['twf'].values