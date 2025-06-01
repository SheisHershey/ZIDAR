#import self as self

from pts.model.deepar import DeepAREstimator
from pts import Trainer
from pts.modules.distribution_output import NormalOutput,NormalMixtureOutput
from torch.distributions import Categorical
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.common import ListDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score
import spec
from gluonts.evaluation.backtest import make_evaluation_predictions
from tqdm import tqdm
from gluonts.evaluation import Evaluator
import json
import os
from imblearn.over_sampling import SMOTE
import hmmlearn.hmm as hmm
from gluonts.dataset.repository.datasets import get_dataset
from scipy.stats import variation
from scipy import stats
import joblib
import torch
import mxnet as mx
import random
##设置随机数种子，保证每次运行代码，能得到相同的结果
r_seed = 95
random.seed(r_seed)
torch.manual_seed(r_seed) # pytorch为CPU设置随机种子
mx.random.seed(r_seed)
np.random.seed(r_seed)


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    """
    计算平均绝对比例误差（MASE）
    y_true: 测试集真实值
    y_pred: 测试集预测值
    y_train: 训练集真实值
    """
    n = y_true.shape[0]
    scaling_factor = np.mean(np.abs(np.diff(y_train)))
    errors = np.abs(y_true - y_pred) / scaling_factor
    return np.mean(errors)

def root_mean_square_scaled_error(y_true, y_pred, y_train):
    """
    计算平均绝对比例误差（RMSSE）
    y_true: 测试集真实值
    y_pred: 测试集预测值
    y_train: 训练集真实值
    """
    n = y_true.shape[0]
    scaling_factor = np.mean(np.square(np.diff(y_train)))
    errors = np.square(y_true - y_pred) / scaling_factor
    return np.sqrt(np.mean(errors))

def QuantileLoss(y, y_hat, rou):
    # return 2 * (np.sum(y_hat) - np.sum(y)) * (rou * (np.sum(y_hat)>np.sum(y)) - (1-rou) * (np.sum(y_hat)<=np.sum(y)))
    E = np.sum(y_hat) - np.sum(y)
    return max(rou * E, (1-rou) * (-E))
    #return 2 * np.sum(np.abs((y_hat - y) * ((y <= y_hat) - rou)))

def BiasAmount(y, y_hat):
    E = np.sum(y_hat) - np.sum(y)
    return E

def RMSSE(y, y_hat, x, h=1):
    fenzi = np.sum(np.square(y - y_hat))
    T = len(x)

    fenmu = 0.0
    for t in range(1, T):
        fenmu += (x[t] - x[t-1])**2
    fenmu = fenmu / (T-1)

    RMSSE = np.sqrt(fenzi / (fenmu * h))
    return RMSSE

def Segment_MAE(test_y, test_predict):
    test_y_0 = []
    test_predict_0 = []
    test_y_0_50 = []
    test_predict_0_50 = []
    test_y_50_100 = []
    test_predict_50_100 = []
    test_y_100_above = []
    test_predict_100_above = []

    for i in range(len(test_y)):
        if test_y[i] == 0:
            test_y_0.append(test_y[i])
            test_predict_0.append(test_predict[i])
        elif test_y[i] > 0 and test_y[i] <= 50:
            test_y_0_50.append(test_y[i])
            test_predict_0_50.append(test_predict[i])
        elif test_y[i] > 50 and test_y[i] <= 100:
            test_y_50_100.append(test_y[i])
            test_predict_50_100.append(test_predict[i])
        elif test_y[i] > 100:
            test_y_100_above.append(test_y[i])
            test_predict_100_above.append(test_predict[i])

    RMSE_0 = mean_absolute_error(test_y_0, test_predict_0)
    print('0RMSE:%f' % (RMSE_0))
    RMSE_0_50 = mean_absolute_error(test_y_0_50, test_predict_0_50)
    print('0-50RMSE值:%f' % (RMSE_0_50))
    RMSE_50_100 = mean_absolute_error(test_y_50_100, test_predict_50_100)
    print('50-100RMSE值:%f' % (RMSE_50_100))
    RMSE_100_above = mean_absolute_error(test_y_100_above, test_predict_100_above)
    print('>100RMSE值:%f' % (RMSE_100_above))

    ##保存各个分段的真实值和预测值
    # df_0 = pd.DataFrame({'test_y':test_y_0, 'test_predict':test_predict_0})
    # df_0.to_csv('../predict_result(半月数据InterSim聚类的477SKU)/DeepAR(混合分布2LSTM+2Normal+1NormalMixture)分段结果/DeepAR(混合分布)_0.csv',index=False)
    #
    # df_0_50 = pd.DataFrame({'test_y':test_y_0_50, 'test_predict':test_predict_0_50})
    # df_0_50.to_csv('../predict_result(半月数据InterSim聚类的477SKU)/DeepAR(混合分布2LSTM+2Normal+1NormalMixture)分段结果/DeepAR(混合分布)_0_50.csv',index=False)
    #
    # df_50_100 = pd.DataFrame({'test_y':test_y_50_100, 'test_predict':test_predict_50_100})
    # df_50_100.to_csv('../predict_result(半月数据InterSim聚类的477SKU)/DeepAR(混合分布2LSTM+2Normal+1NormalMixture)分段结果/DeepAR(混合分布)_50_100.csv',index=False)
    #
    # df_100_above = pd.DataFrame({'test_y':test_y_100_above, 'test_predict':test_predict_100_above})
    # df_100_above.to_csv('../predict_result(半月数据InterSim聚类的477SKU)/DeepAR(混合分布2LSTM+2Normal+1NormalMixture)分段结果/DeepAR(混合分布)_100_above.csv',index=False)

def Segment_RMSE(test_y, test_predict):
    test_y_0 = []
    test_predict_0 = []
    test_y_0_50 = []
    test_predict_0_50 = []
    test_y_50_100 = []
    test_predict_50_100 = []
    test_y_100_above = []
    test_predict_100_above = []

    for i in range(len(test_y)):
        if test_y[i] == 0:
            test_y_0.append(test_y[i])
            test_predict_0.append(test_predict[i])
        elif test_y[i] > 0 and test_y[i] <= 50:
            test_y_0_50.append(test_y[i])
            test_predict_0_50.append(test_predict[i])
        elif test_y[i] > 50 and test_y[i] <= 100:
            test_y_50_100.append(test_y[i])
            test_predict_50_100.append(test_predict[i])
        elif test_y[i] > 100:
            test_y_100_above.append(test_y[i])
            test_predict_100_above.append(test_predict[i])

    RMSE_0 = mean_squared_error(test_y_0, test_predict_0, squared=False)
    print('0RMSE:%f' % (RMSE_0))
    RMSE_0_50 = mean_squared_error(test_y_0_50, test_predict_0_50, squared=False)
    print('0-50RMSE值:%f' % (RMSE_0_50))
    RMSE_50_100 = mean_squared_error(test_y_50_100, test_predict_50_100, squared=False)
    print('50-100RMSE值:%f' % (RMSE_50_100))
    RMSE_100_above = mean_squared_error(test_y_100_above, test_predict_100_above, squared=False)
    print('>100RMSE值:%f' % (RMSE_100_above))

    ##保存各个分段的真实值和预测值
    # df_0 = pd.DataFrame({'test_y':test_y_0, 'test_predict':test_predict_0})
    # df_0.to_csv('../predict_result(半月数据InterSim聚类的477SKU)/DeepAR(混合分布2LSTM+2Normal+1NormalMixture)分段结果/DeepAR(混合分布)_0.csv',index=False)
    #
    # df_0_50 = pd.DataFrame({'test_y':test_y_0_50, 'test_predict':test_predict_0_50})
    # df_0_50.to_csv('../predict_result(半月数据InterSim聚类的477SKU)/DeepAR(混合分布2LSTM+2Normal+1NormalMixture)分段结果/DeepAR(混合分布)_0_50.csv',index=False)
    #
    # df_50_100 = pd.DataFrame({'test_y':test_y_50_100, 'test_predict':test_predict_50_100})
    # df_50_100.to_csv('../predict_result(半月数据InterSim聚类的477SKU)/DeepAR(混合分布2LSTM+2Normal+1NormalMixture)分段结果/DeepAR(混合分布)_50_100.csv',index=False)
    #
    # df_100_above = pd.DataFrame({'test_y':test_y_100_above, 'test_predict':test_predict_100_above})
    # df_100_above.to_csv('../predict_result(半月数据InterSim聚类的477SKU)/DeepAR(混合分布2LSTM+2Normal+1NormalMixture)分段结果/DeepAR(混合分布)_100_above.csv',index=False)



'''
读取数据集
'''
#df = pd.read_csv('../ZIP-DeepAR代码/data/carpartsdelete80.csv', header=0)
#df = pd.read_csv('E:/ZIP-DeepAR代码/data/carpartsdelete70.csv', header=0)
#df = pd.read_csv('E:/ZIP-DeepAR代码/data/carpartsdeletepick.csv', header=0)
df = pd.read_csv('E:\ZIP-DeepAR代码\data\InterSim层次聚类后的Q料202001-202302(halfmonth).csv', header=0)
#df = pd.read_csv('E:\ZIP-DeepAR代码\data\salestv_data.csv', header=0)
#df = pd.read_csv('E:\ZIP-DeepAR代码\data\salestv_data_filtered.csv', header=0)
print(df)
#id_1913 = df['id'].values.tolist()
#df = df.drop('id',axis=1)
sku_477 = df['sku'].values.tolist()
df = df.drop('sku',axis=1)
#Parts_2509 = df['Part'].values.tolist()
#df = df.drop('Part',axis=1)


#将列名转化为202001 ~ 202604月份，避免freq=15d的2-30日期问题
import calendar
from datetime import datetime, timedelta

date_str = '1998-01'
date_format = '%Y-%m'
num_months = 76

dates = []
current_date = datetime.strptime(date_str, date_format)

for i in range(num_months):
    dates.append(current_date.strftime(date_format))
    num_days = calendar.monthrange(current_date.year, current_date.month)[1]
    current_date += timedelta(days=num_days)

df.columns = dates

'''
构造训练集、测试集
'''
##提取test_y
test_y = []
for x in df.values:
    test_y.append(x[-4])
test_y = np.array(test_y)
test_y = test_y.reshape(-1)
print(test_y.shape)
print(test_y)
zero_mask = test_y == 0
non_zero_mask = ~zero_mask
test_y_non_zero = test_y[non_zero_mask]
print(test_y_non_zero)
train_series_list = []
for i in range(df.shape[0]):
    train_series = df.iloc[i, :-4].values  # 假设最后4个点是测试点
    train_series_list.append(train_series)
train_series_non_zero = [train_series_list[i] for i in range(len(test_y)) if non_zero_mask[i]]

train_list = []
prod_sales_max = []
index = 0


for i in range(df.shape[0]):  #构造训练集，每个产品的前72个半月的销量  （[第1个产品的前72个半月的销量，第2个产品的前72个半月的销量,...])
   dic = {'month':df.columns, 'sales':df.iloc[i].values}
   prod_data = pd.DataFrame(dic)
   prod_data.reset_index(inplace=True)

   #对数变换
   prod_data['sales'] = np.log(prod_data['sales']+1)

   # # 数据归一化
   # prod_data['sales'] = (prod_data['sales'] - np.min(prod_data['sales'])) / (np.max(prod_data['sales']) - np.min(prod_data['sales']))

   train_dic = {"start": prod_data.iloc[0]['month'], "target": prod_data.iloc[0:72]['sales']}
   train_list.append(train_dic)
training_data = ListDataset(train_list, freq="1m") #输入数据格式
print(len(training_data))
print()

###构造测试集，每个产品的第73~76个月的销量
### 为h个单步预测时间点，分别构建h个ListDataset测试集
test_data_list = []
for predict_day in range(73, 74): #需要单步预测的时间点（使用前t-1时间点的真实值，来预测第t个时间点的预测值)
    test_list = []
    for i in range(df.shape[0]):
        dic = {'month': df.columns, 'sales': df.iloc[i].values}
        prod_data = pd.DataFrame(dic)
        prod_data.reset_index(inplace=True)

        # 对数变换
        prod_data['sales'] = np.log(prod_data['sales'] + 1)
        test_dic = {"start": prod_data.iloc[0]['month'], "target": prod_data.iloc[0:predict_day]['sales']} ###dict修改成predict_length=1的形式

        # # 数据归一化
        # test_dic['target'] = (test_dic['target'] - np.min(test_dic['target'])) / (np.max(test_dic['target']) - np.min(test_dic['target']))
        # # 每条测试样本对应的最大最小值
        # max_values.append(np.max(test_dic['target']))
        # min_values.append(np.min(test_dic['target']))
        test_list.append(test_dic)
    test_data = ListDataset(test_list, freq="1m")
    print('第%d个单步预测时间点的测试集大小(样本数)'%len(test_data))

    test_data_list.append(test_data)



'''
模型训练
'''
# # # 定义混合分布的成分数
# # num_components = 2
# # 定义每个成分的概率分布类型
# distr_outputs = [GaussianOutput(), GaussianOutput()]
#
# # 定义混合分布输出层，，，，，
# distribution = MixtureDistributionOutput(distr_outputs = distr_outputs)

print('####模型训练####')
context_l = 24
trainer = Trainer(epochs=15,num_batches_per_epoch=50,batch_size=1024,learning_rate=1e-2,patience=3,minimum_learning_rate=1e-5,learning_rate_decay_factor=0.5,
                  weight_decay=0.01,# l2正则化项
                  clip_gradient=1, # !! 梯度裁剪(0.1,0.5,1,3,5)
                  loss_threshold=0.1)  #!! 增加loss_threshold

estimator = DeepAREstimator(freq="1m",context_length=context_l,input_size=17,#需要根据代码报错来设定，input.size(-1) must be equal to input_size
                            prediction_length=1, num_layers=2, num_cells=16, num_layers2=2, num_cells2=64, #numcell 32
                            dropout_rate=0.1, # 加入dropout防止过拟合
                            cell_type = "LSTM",
                            scaling=True, #是否对数据进行标准化
                            distr_output=NormalOutput(),distr_output2=NormalOutput(),distr_output3=NormalMixtureOutput(components=3), #全连接pdf网络层，需要通过神经网络来训练全连接层参数，通过输入数据才能输出分布参数。可以通过distribution方法来得到分布函数和参数
                            trainer=trainer)
                                     # context_length窗口长度，predict_length预测长度，num_layers=2,num_cells=40
                                     # 全连接输出层 使用2个正态分布的混合分布
predictor = estimator.train(training_data=training_data)
#输出DeepAR模型参数
#print(predictor.training_net.collect_params())
#print(predictor.prediction_net.collect_params())

'''
模型批量测试
'''
print('####模型批量测试####')
##预测阈值(小于阈值记为0)
predict_threshold = 1
zero_prob_threshold = 0.5

##设置随机种子
torch.manual_seed(r_seed)  # pytorch为CPU设置随机种子
mx.random.seed(r_seed)
np.random.seed(r_seed)

# 直接predict预测
# new_test_list = []
# for single_test_data in test_data:
#                                                                 # predict方法排除预测区间(单步预测的-1，多步预测的-h)
#     single_test_data['target'] = single_test_data['target'][:-1]# make_evaluation方法不需要排除，它是用在验证集上的，而不是测试集
#     new_test_list.append(single_test_data)                      # 可以考虑使用-(contextl+1):-1去截取
# new_test_data = ListDataset(new_test_list, freq="1m")
# forecast_it = predictor.predict(new_test_data) #forecast_it是一个迭代器，里面每个元素是prediction+distr_args
#                                                #prediction：1个时间序列样本的预测路径结果， distr_args：所有分布参数结果

# make_evaluation_predictions预测
## 分别对h个单步预测时间点各自的测试集进行预测
test_predict_alltime = []
test_predict_non_zero_alltime = []
test_predict_10_alltime = []
test_predict_50_alltime = []
test_predict_90_alltime = []

for test_data in test_data_list: #h个单步预测时间点，分别构建h个ListDataset测试集
    forecast_it, ts_it = make_evaluation_predictions( #forecast_it是一个迭代器，里面每个元素是prediction+distr_args
        dataset=test_data,
        predictor=predictor,
        num_samples=100
    )

    print("Obtaining time series conditioning values ...")
    tss = list(tqdm(ts_it, total=len(test_data)))

    print("Obtaining time series predictions ...")
    forecasts = list(tqdm(forecast_it, total=len(test_data)))

    #真实值(make_evaluation_predictions输出的ts_it，与原始数据集df和测试集test_data的预测样本顺序一致)
    # test_y = []
    # for i in range(len(tss)):
    #     # 对数变换还原(exp(x)-1会将极小接近于0的prediction_means，还原成0）
    #     tss[i] = np.exp(tss[i]) - 1
    #     test_y.append(tss[i].iloc[-1:].T.values)# 若为多步预测predict_length=h，则改为iloc[-h:]
    # test_y = np.array(test_y)
    # test_y = test_y.reshape(-1)


    #预测值+分布参数
    test_predict = []
    test_predict_non_zero = []
    test_predict_10 = []
    test_predict_50 = []
    test_predict_90 = []

    distr_args_list = []
    for i in range(len(forecasts)): #1个list
        prediction, distr_args = forecasts[i] #每个样本的预测输出结果: 预测值+分布参数
        mix_logits, loc, dist_scale = distr_args #每个元素，都是(batchsize * num_samples, predict_length, num_components)
                                                 #第一维，0~99表示第1个时间序列样本的100条预测路径, 100~199表示第2个时间序列样本的100条预测路径
                                #!! 由于预测时每次样本数也为batchsize大小，而非所有样本一起预测，因此distr_args只包含预测的最后一个batch内样本的分布参数
        probs = Categorical(logits=mix_logits).probs
        # print(probs)
        # print(probs.shape)
        # print(probs[(i*100):(i*100+100), 0, 0])

        '''
        抽样均值
        '''
        prediction_means = np.array(list(prediction.mean))[0]  # 100条抽样路径的均值
        # print('对数变换还原前：', prediction_means)
        # print(prediction.samples)
        '''
        抽样分位数
        '''
        prediction_array = np.array(list(prediction.samples))
        prediction_non_zero = np.mean(prediction_array[prediction_array != 0]) if np.any(prediction_array != 0) else 0.0
        prediction_10 = np.percentile(prediction_array, 10)
        prediction_50 = np.percentile(prediction_array, 50)
        prediction_90 = np.percentile(prediction_array, 90)

        # 对数变换还原(exp(x)-1会将极小接近于0的prediction_means，还原成0）
        prediction_means = np.exp(prediction_means) - 1
        prediction_non_zero = np.exp(prediction_non_zero) - 1
        prediction_10 = np.exp(prediction_10) - 1
        prediction_50 = np.exp(prediction_50) - 1
        prediction_90 = np.exp(prediction_90) - 1
        print('对数变换还原后：', prediction_means)
        print('对数变换还原后：', prediction_non_zero)
        print('对数变换还原后：', prediction_10)
        print('对数变换还原后：', prediction_50)
        print('对数变换还原后：', prediction_90)
        print()

        # '''
        # 如果预测值 < 阈值predict_threshold，记为0
        # '''
        if prediction_means < predict_threshold:
            prediction_means = 0.0

        # '''
        # 如果多条预测路径的平均零值概率 > 阈值zero_prob_threshold，记为0
        # '''
        #print(torch.mean(probs[(i*100):(i*100+100), 0, 0]))
        #if torch.mean(probs[(i*100):(i*100+100), 0, 0]) > zero_prob_threshold:
        #    prediction_means = 0.0

        test_predict.append(prediction_means)
        test_predict_non_zero.append(prediction_non_zero)
        test_predict_10.append(prediction_10)
        test_predict_50.append(prediction_50)
        test_predict_90.append(prediction_90)
        index += 1
    test_predict_alltime.append(test_predict)
    test_predict_non_zero_alltime.append(test_predict_non_zero)
    test_predict_10_alltime.append(test_predict_10)
    test_predict_50_alltime.append(test_predict_50)
    test_predict_90_alltime.append(test_predict_90)

test_predict = np.array(test_predict_alltime) #(单步预测的时间数，时间序列SKU数)
test_predict = np.transpose(test_predict).reshape(-1) #(时间序列SKU数, 单步预测的时间数)

test_predict_non_zero = np.array(test_predict_non_zero_alltime) #(单步预测的时间数，时间序列SKU数)
test_predict_non_zero = np.transpose(test_predict_non_zero).reshape(-1) #(时间序列SKU数, 单步预测的时间数)

test_predict_10 = np.array(test_predict_10_alltime) #(单步预测的时间数，时间序列SKU数)
test_predict_10 = np.transpose(test_predict_10).reshape(-1) #(时间序列SKU数, 单步预测的时间数)

test_predict_50 = np.array(test_predict_50_alltime) #(单步预测的时间数，时间序列SKU数)
test_predict_50 = np.transpose(test_predict_50).reshape(-1) #(时间序列SKU数, 单步预测的时间数)

test_predict_90 = np.array(test_predict_90_alltime) #(单步预测的时间数，时间序列SKU数)
test_predict_90 = np.transpose(test_predict_90).reshape(-1) #(时间序列SKU数, 单步预测的时间数)

###save DeepAR prediction预测值
df_test_predict = pd.DataFrame({ 'test_y':test_y,'test_predict':test_predict})
#df_test_predict.to_csv('预测结果m5delta3124.csv',index=False)
test_predict = test_predict.reshape(-1)
print(test_predict.shape)

###save DeepAR distr_args分布参数
print('###save DeepAR distr_args分布参数')
mix_logits, loc, dist_scale = distr_args
#每个元素，都是(batchsize * num_samples, predict_length, num_components)
#第一维，0~99表示第1个时间序列样本的100条预测路径, 100~199表示第2个时间序列样本的100条预测路径
#!! 由于预测时每次样本数也为batchsize大小，因此distr_args只包含预测的最后一个batch内的分布参数

distr_args_list = []
# print(len(test_y))
# print(mix_logits)
# print(mix_logits.shape)
for i in range(int(mix_logits.shape[0]/100)): #!! 由于预测时每次样本数也为batchsize大小，因此distr_args只包含预测的最后一个batch内的分布参数
    single_distr_args = Categorical(logits=mix_logits).probs[i * 100][0].numpy().tolist() \
                        + loc[i * 100][0].numpy().tolist() \
                        + dist_scale[i * 100][0].numpy().tolist()
    distr_args_list.append(single_distr_args)
distr_args_df = pd.DataFrame(distr_args_list,columns=['w1','w2','w3','mean1','mean2','mean3','std1','std2','std3'])
distr_args_df.to_csv('概率分布参数(只有最后1个预测批次的参数).csv',index=False)
print()


'''
四、模型评估
'''
print('####模型评估####')
testmonth_num = 1 #预测区间时间点的数量（大于1不代表一定是多步预测，也可以是单步预测）
test_y_non_zero = []
test_predict_non_zero = []
for y_true, y_pred in zip(test_y, test_predict):
    if y_true != 0:  # 只保留实际值非零的预测对
        test_y_non_zero.append(y_true)
        test_predict_non_zero.append(y_pred)
##R2指标
Rsquare = r2_score(test_y, test_predict)
print('R2值：%f'%(Rsquare))

##RMSE指标
RMSE = mean_squared_error(test_y, test_predict, squared=False)
print('RMSE值:%f'%(RMSE))

RMSE_non_zero = mean_squared_error(test_y_non_zero, test_predict_non_zero, squared=False)
print('RMSE_non_zero值:%f'%(RMSE_non_zero))

##MAE指标
MAE = mean_absolute_error(test_y, test_predict)
MAE_non_zero = mean_absolute_error(test_y_non_zero, test_predict_non_zero)
print('MAE值:%f'%(MAE))
print('MAE_non_zero值:%f'%(MAE_non_zero))

##MASE指标
MASE1 = 0.0
num = int(len(test_y)/testmonth_num)
for i in range(num):
    MASE1 += mean_absolute_scaled_error(test_y[(testmonth_num*i):(testmonth_num*i+testmonth_num)], test_predict[(testmonth_num*i):(testmonth_num*i+testmonth_num)], df.iloc[i, 0:-4].values)
MASE1 = MASE1 / num
print('MASE1:%f' % (MASE1))

MASE_non_zero = 0.0
num = int(len(test_y_non_zero)/testmonth_num)
for i in range(num):
    y_true_segment = np.array(test_y_non_zero[(testmonth_num * i):(testmonth_num * i + testmonth_num)])
    y_pred_segment = np.array(test_predict_non_zero[(testmonth_num * i):(testmonth_num * i + testmonth_num)])

    MASE_non_zero += mean_absolute_scaled_error(
        y_true_segment,
        y_pred_segment,
        df.iloc[i, 0:-4].values
    )

MASE_non_zero /= num
print('MASE_non_zero: %f' % MASE_non_zero)

# MASE2 = mean_absolute_scaled_error(test_y, test_predict, df.iloc[:, 0:-4].values)
# print('MASE2:%f' % (MASE2))


##RMSSE指标
'''
RMSSE指标
'''
RMSSE_value = 0.0
num = int(len(test_y)/testmonth_num)
for i in range(num):
    RMSSE_value += root_mean_square_scaled_error(test_y[(testmonth_num*i):(testmonth_num*i+testmonth_num)], test_predict[(testmonth_num*i):(testmonth_num*i+testmonth_num)], df.iloc[i, 0:-4].values)
RMSSE_value = RMSSE_value / num
print('RMSSE1:%f' % (RMSSE_value))

RMSSE_value2 = 0.0
num = int(len(test_y)/testmonth_num)
for i in range(num):
    RMSSE_value2 += RMSSE(test_y[(testmonth_num*i):(testmonth_num*i+testmonth_num)], test_predict[(testmonth_num*i):(testmonth_num*i+testmonth_num)], df.iloc[i, 0:-4].values, h=testmonth_num)
RMSSE_value2 = RMSSE_value2 / num
print('RMSSE2:%f' % (RMSSE_value2))

RMSSE_non_zero = 0.0
num = int(len(test_y_non_zero)/testmonth_num)
RMSSE_non_zero = 0.0

for i in range(num):
    y_true_segment = np.array(test_y_non_zero[(testmonth_num * i):(testmonth_num * i + testmonth_num)])
    y_pred_segment = np.array(test_predict_non_zero[(testmonth_num * i):(testmonth_num * i + testmonth_num)])

    RMSSE_non_zero += RMSSE(
        y_true_segment,
        y_pred_segment,
        df.iloc[i, 0:-4].values,
        h=testmonth_num
    )

RMSSE_non_zero /= num
print('RMSSE_non_zero: %f' % RMSSE_non_zero)

##QuantileLoss指标
from gluonts.evaluation.metrics import quantile_loss
LS_list = [(0,1)]  #L表示相对于第1个预测时间t0的QuantileLoss区间起始点，S表示QuantileLoss区间长度
for LS_pair in LS_list:
    L = LS_pair[0]
    S = LS_pair[1]
    QuantileLoss2 = 0.0
    test_y2 = 0.0
    num = int(len(test_y) / testmonth_num)  ##时间序列sku数量
    for i in range(num):
        QuantileLoss2 += quantile_loss(test_y[(testmonth_num * i + L):(testmonth_num * i + L + S)],
                                      test_predict_10[(testmonth_num * i + L):(testmonth_num * i + L + S)],
                                      q=0.1)
        test_y2 += np.sum(test_y[(testmonth_num * i + L):(testmonth_num * i + L + S)])
    #print(QuantileLoss2,test_y2)
    QuantileLoss2 = QuantileLoss2 / test_y2
    print('L=%d,S=%d,rou=10%%时的QuantileLoss值：%f' % (L,S,QuantileLoss2))
print()
for LS_pair in LS_list:
    L = LS_pair[0]
    S = LS_pair[1]
    QuantileLoss2 = 0.0
    test_y2 = 0.0
    num = int(len(test_y) / testmonth_num)  ##时间序列sku数量
    for i in range(num):
        QuantileLoss2 += quantile_loss(test_y[(testmonth_num * i + L):(testmonth_num * i + L + S)],
                                      test_predict_50[(testmonth_num * i + L):(testmonth_num * i + L + S)],
                                      q=0.5)
        test_y2 += np.sum(test_y[(testmonth_num * i + L):(testmonth_num * i + L + S)])
    #print(QuantileLoss2,test_y2)
    QuantileLoss2 = QuantileLoss2 / test_y2
    print('L=%d,S=%d,rou=50%%时的QuantileLoss值：%f' % (L,S,QuantileLoss2))
print()
for LS_pair in LS_list:
    L = LS_pair[0]
    S = LS_pair[1]
    QuantileLoss2 = 0.0
    test_y2 = 0.0
    num = int(len(test_y) / testmonth_num)  ##时间序列sku数量
    for i in range(num):
        QuantileLoss2 += quantile_loss(test_y[(testmonth_num * i + L):(testmonth_num * i + L + S)],
                                      test_predict[(testmonth_num * i + L):(testmonth_num * i + L + S)],
                                      q=0.5)
        test_y2 += np.sum(test_y[(testmonth_num * i + L):(testmonth_num * i + L + S)])
    #print(QuantileLoss2,test_y2)
    QuantileLoss2 = QuantileLoss2 / test_y2
    print('L=%d,S=%d,rou=50%%时的QuantileLossprdm值：%f' % (L,S,QuantileLoss2))
print()
for LS_pair in LS_list:
    L = LS_pair[0]
    S = LS_pair[1]
    QuantileLoss2 = 0.0
    test_y2 = 0.0
    num = int(len(test_y) / testmonth_num)  ##时间序列sku数量
    for i in range(num):
        QuantileLoss2 += quantile_loss(test_y[(testmonth_num * i + L):(testmonth_num * i + L + S)],
                                      test_predict_90[(testmonth_num * i + L):(testmonth_num * i + L + S)],
                                      q=0.9)
        test_y2 += np.sum(test_y[(testmonth_num * i + L):(testmonth_num * i + L + S)])
    #print(QuantileLoss2,test_y2)
    QuantileLoss2 = QuantileLoss2 / test_y2
    print('L=%d,S=%d,rou=90%%时的QuantileLoss值：%f' % (L,S,QuantileLoss2))
print()

##Zero精确率、召回率、F1-Score
test_y_01 = [1 if x == 0 else 0 for x in test_y] #0为正类，1为负类
test_predict_01 = [1 if x == 0 else 0 for x in test_predict]
precision = precision_score(test_y_01, test_predict_01)
recall = recall_score(test_y_01, test_predict_01)
f1score = f1_score(test_y_01, test_predict_01)
print('Precision值:%f'%(precision))
print('Recall值:%f'%(recall))
print('F1-Score值:%f'%(f1score))

##分段RMSE指标
Segment_MAE(test_y, test_predict)
Segment_RMSE(test_y, test_predict)


train_series_non_zero = [train_series_list[i] for i in range(len(train_series_list)) if non_zero_mask[i]]

# Calculate non-zero MASE
non_zero_mase_values = [mean_absolute_scaled_error(a, p, t)
                       for a, p, t in zip(test_y_non_zero, test_predict_non_zero, train_series_non_zero)]
print(f'Non-zero MASE: {np.nanmean(non_zero_mase_values):.4f}')

# Calculate non-zero RMSSE
non_zero_rmsse_values = [root_mean_square_scaled_error(a, p, t)
                        for a, p, t in zip(test_y_non_zero, test_predict_non_zero, train_series_non_zero)]
print(f'Non-zero RMSSE: {np.nanmean(non_zero_rmsse_values):.4f}')
# ##SPEC指标
# SPEC = 0.0
# num = int(len(test_y)/testmonth_num)
# for i in range(num):
#     SPEC += spec(test_y[(testmonth_num*i):(testmonth_num*i+testmonth_num)], test_predict[(testmonth_num*i):(testmonth_num*i+testmonth_num)], a1=0.5, a2=0.5)
# SPEC = SPEC / num
# print('SPEC(0.5,0.5)值:%f' % (SPEC))
#
# SPEC2 = 0.0
# num = int(len(test_y)/testmonth_num)
# for i in range(num):
#     SPEC2 += spec(test_y[(testmonth_num*i):(testmonth_num*i+testmonth_num)], test_predict[(testmonth_num*i):(testmonth_num*i+testmonth_num)], a1=0.75, a2=0.25)
# SPEC2 = SPEC2 / num
# print('SPEC值(0.75,0.25):%f' % (SPEC2))
#
# SPEC3 = 0.0
# num = int(len(test_y)/testmonth_num)
# for i in range(num):
#     SPEC3 += spec(test_y[(testmonth_num*i):(testmonth_num*i+testmonth_num)], test_predict[(testmonth_num*i):(testmonth_num*i+testmonth_num)], a1=0.25, a2=0.75)
# SPEC3 = SPEC3 / num
# print('SPEC值(0.25,0.75):%f' % (SPEC3))

##Evaluator输出的指标(QuantileLoss结果很奇怪)
# forecasts_values = [x[0] for x in forecasts] #取出forecasts元组列表中的预测结果，而不包含分布参数
# from gluonts.evaluation import Evaluator
# evaluator = Evaluator()
# agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts_values), num_series=len(test_data))
# print(agg_metrics)

