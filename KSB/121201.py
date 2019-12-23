#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看个人上传数据目录
# ls ../kernel/

# 查看个人工作区文件
# ls ../work/

# 查看当前挂载的比赛数据集目录
# ls ../data/

# 参考

# https://www.biendata.com/models/category/2897/L_notebook

# In[2]:


# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import logging

log_fmt = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
logging.basicConfig(format=log_fmt, level=logging.INFO)

import warnings
warnings.filterwarnings('default')


def extract_day(s):
    return s.apply(lambda x: int(x.split('-')[0][1:]))


def extract_hour(s):
    return s.apply(lambda x: int(x.split('-')[1][1:]))


base_path = '../data'

# 加载邀请回答数据

train = pd.read_csv(f'{base_path}/invite_info_0926.txt', sep='\t', header=None)
train.columns = ['qid', 'uid', 'dt', 'label']
logging.info("invite %s", train.shape)

test = pd.read_csv(f'{base_path}/invite_info_evaluate_1_0926.txt', sep='\t', header=None)
test.columns = ['qid', 'uid', 'dt']
logging.info("test %s", test.shape)

sub = test.copy()

sub_size = len(sub)

train['day'] = extract_day(train['dt'])
train['hour'] = extract_hour(train['dt'])

test['day'] = extract_day(test['dt'])
test['hour'] = extract_hour(test['dt'])

del train['dt'], test['dt']

# 加载问题
ques = pd.read_csv(f'{base_path}/question_info_0926.txt', header=None, sep='\t')
ques.columns = ['qid', 'q_dt', 'title_t1', 'title_t2', 'desc_t1', 'desc_t2', 'topic']
del ques['title_t1'], ques['title_t2'], ques['desc_t1'], ques['desc_t2']
logging.info("ques %s", ques.shape)

ques['q_day'] = extract_day(ques['q_dt'])
ques['q_hour'] = extract_hour(ques['q_dt'])
del ques['q_dt']

# 加载回答
ans = pd.read_csv(f'{base_path}/answer_info_0926.txt', header=None, sep='\t')
ans.columns = ['aid', 'qid', 'uid', 'ans_dt', 'ans_t1', 'ans_t2', 'is_good', 'is_rec', 'is_dest', 'has_img',
               'has_video', 'word_count', 'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
               'reci_xxx', 'reci_no_help', 'reci_dis']
del ans['ans_t1'], ans['ans_t2']
logging.info("ans %s", ans.shape)

ans['a_day'] = extract_day(ans['ans_dt'])
ans['a_hour'] = extract_hour(ans['ans_dt'])
del ans['ans_dt']

ans = pd.merge(ans, ques, on='qid')
del ques

# 回答距提问的天数
ans['diff_qa_days'] = ans['a_day'] - ans['q_day']

# 时间窗口划分
# train
# val
train_start = 3838
train_end = 3867

val_start = 3868
val_end = 3874

label_end = 3867
label_start = label_end - 6

train_label_feature_end = label_end - 7
train_label_feature_start = train_label_feature_end - 22

train_ans_feature_end = label_end - 7
train_ans_feature_start = train_ans_feature_end - 50

val_label_feature_end = val_start - 1
val_label_feature_start = val_label_feature_end - 22

val_ans_feature_end = val_start - 1
val_ans_feature_start = val_ans_feature_end - 50

train_label_feature = train[(train['day'] >= train_label_feature_start) & (train['day'] <= train_label_feature_end)]
logging.info("train_label_feature %s", train_label_feature.shape)

val_label_feature = train[(train['day'] >= val_label_feature_start) & (train['day'] <= val_label_feature_end)]
logging.info("val_label_feature %s", val_label_feature.shape)

train_label = train[(train['day'] > train_label_feature_end)]

logging.info("train feature start %s end %s, label start %s end %s", train_label_feature['day'].min(),
             train_label_feature['day'].max(), train_label['day'].min(), train_label['day'].max())

logging.info("test feature start %s end %s, label start %s end %s", val_label_feature['day'].min(),
             val_label_feature['day'].max(), test['day'].min(), test['day'].max())

# 确定ans的时间范围
# 3807~3874
train_ans_feature = ans[(ans['a_day'] >= train_ans_feature_start) & (ans['a_day'] <= train_ans_feature_end)]

val_ans_feature = ans[(ans['a_day'] >= val_ans_feature_start) & (ans['a_day'] <= val_ans_feature_end)]

logging.info("train ans feature %s, start %s end %s", train_ans_feature.shape, train_ans_feature['a_day'].min(),
             train_ans_feature['a_day'].max())

logging.info("val ans feature %s, start %s end %s", val_ans_feature.shape, val_ans_feature['a_day'].min(),
             val_ans_feature['a_day'].max())

fea_cols = ['is_good', 'is_rec', 'is_dest', 'has_img', 'has_video', 'word_count',
            'reci_cheer', 'reci_uncheer', 'reci_comment', 'reci_mark', 'reci_tks',
            'reci_xxx', 'reci_no_help', 'reci_dis', 'diff_qa_days']


def extract_feature1(target, label_feature, ans_feature):
    # 问题特征
    t1 = label_feature.groupby('qid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['qid', 'q_inv_mean', 'q_inv_sum', 'q_inv_std', 'q_inv_count']
    target = pd.merge(target, t1, on='qid', how='left')

    # 用户特征
    t1 = label_feature.groupby('uid')['label'].agg(['mean', 'sum', 'std', 'count']).reset_index()
    t1.columns = ['uid', 'u_inv_mean', 'u_inv_sum', 'u_inv_std', 'u_inv_count']
    target = pd.merge(target, t1, on='uid', how='left')
    #
    # train_size = len(train)
    # data = pd.concat((train, test), sort=True)

    # 回答部分特征

    t1 = ans_feature.groupby('qid')['aid'].count().reset_index()
    t1.columns = ['qid', 'q_ans_count']
    target = pd.merge(target, t1, on='qid', how='left')

    t1 = ans_feature.groupby('uid')['aid'].count().reset_index()
    t1.columns = ['uid', 'u_ans_count']
    target = pd.merge(target, t1, on='uid', how='left')

    for col in fea_cols:
        t1 = ans_feature.groupby('uid')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['uid', f'u_{col}_sum', f'u_{col}_max', f'u_{col}_mean']
        target = pd.merge(target, t1, on='uid', how='left')

        t1 = ans_feature.groupby('qid')[col].agg(['sum', 'max', 'mean']).reset_index()
        t1.columns = ['qid', f'q_{col}_sum', f'q_{col}_max', f'q_{col}_mean']
        target = pd.merge(target, t1, on='qid', how='left')
        logging.info("extract %s", col)
    return target


train_label = extract_feature1(train_label, train_label_feature, train_ans_feature)
test = extract_feature1(test, val_label_feature, val_ans_feature)

# 特征提取结束
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)
assert len(test) == sub_size


# In[2]:



# 加载用户
user = pd.read_csv(f'{base_path}/member_info_0926.txt', header=None, sep='\t')
user.columns = ['uid', 'gender', 'creat_keyword', 'level', 'hot', 'reg_type', 'reg_plat', 'freq', 'uf_b1', 'uf_b2',
                'uf_b3', 'uf_b4', 'uf_b5', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5', 'score', 'follow_topic',
                'inter_topic']


# In[3]:


def parse_map(d):
    if d == '-1':
        return {}
    return dict([int(z.split(':')[0][1:]), float(z.split(':')[1])] for z in d.split(','))
user['inter_topic'] = user['inter_topic'].apply(parse_map)



def most_interest_topic(d):
    if len(d) == 0:
        return -1
    return list(d.keys())[np.argmax(list(d.values()))]

# In[ ]:


# 用户最感兴趣的topic
user['most_interest_topic'] = user['inter_topic'].apply(most_interest_topic)


# In[4]:


def parse_list_1(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))


user['follow_topic'] = user['follow_topic'].apply(parse_list_1)







# In[5]:


def get_interest_values(d):
    if len(d) == 0:
        return [0]
    return list(d.values())

# In[ ]:


# 用户topic兴趣值的统计特征
user['interest_values'] = user['inter_topic'].apply(get_interest_values)
user['min_interest_values'] = user['interest_values'].apply(np.min)
user['max_interest_values'] = user['interest_values'].apply(np.max)
user['mean_interest_values'] = user['interest_values'].apply(np.mean)
user['std_interest_values'] = user['interest_values'].apply(np.std)
user['num_atten_topic'] = user['follow_topic'].apply(len)
user['num_interest_topic'] = user['inter_topic'].apply(len)


# In[6]:


user[1:10]


# In[ ]:





# In[ ]:





# In[7]:


del user['follow_topic'],user['inter_topic'],user['interest_values']


# In[ ]:





# In[27]:


t = user.dtypes
cats = [x for x in t[t == 'object'].index if x not in [ 'uid']]
logging.info("user cat %s", cats)


# In[29]:


user[1:10]


# In[8]:



logging.info("user %s", user.shape)

unq = user.nunique()
logging.info("user unq %s", unq)



# merge user


# In[5]:


for x in unq[unq == 1].index:
    del user[x]
    logging.info('del unq==1 %s', x)

t = user.dtypes
cats = [x for x in t[t == 'object'].index if x not in [ 'uid']]
logging.info("user cat %s", cats)

for d in cats:
    lb = LabelEncoder()
    user[d] = lb.fit_transform(user[d])
    logging.info('encode %s', d)

q_lb = LabelEncoder()
q_lb.fit(list(train_label['qid'].astype(str).values) + list(test['qid'].astype(str).values))
train_label['qid_enc'] = q_lb.transform(train_label['qid'])
test['qid_enc'] = q_lb.transform(test['qid'])

u_lb = LabelEncoder()
u_lb.fit(user['uid'])
train_label['uid_enc'] = u_lb.transform(train_label['uid'])
test['uid_enc'] = u_lb.transform(test['uid'])


# In[6]:


user[1:10]


# In[7]:


train_label = pd.merge(train_label, user, on='uid', how='left')
test = pd.merge(test, user, on='uid', how='left')
logging.info("train shape %s, test shape %s", train_label.shape, test.shape)

data = pd.concat((train_label, test), axis=0, sort=True)
#del train_label, test

# count编码
count_fea = ['uid_enc', 'qid_enc', 'gender', 'freq', 'uf_c1', 'uf_c2', 'uf_c3', 'uf_c4', 'uf_c5']
for feat in count_fea:
    col_name = '{}_count'.format(feat)
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data.loc[data[col_name] < 2, feat] = -1
    data[feat] += 1
    data[col_name] = data[feat].map(data[feat].value_counts().astype(int))
    data[col_name] = (data[col_name] - data[col_name].min()) / (data[col_name].max() - data[col_name].min())
    # 

# 问题被回答的次数


# 压缩数据
t = data.dtypes
for x in t[t == 'int64'].index:
    data[x] = data[x].astype('int32')

for x in t[t == 'float64'].index:
    data[x] = data[x].astype('float32')

data['wk'] = data['day'] % 7

feature_cols = [x for x in data.columns if x not in ('label', 'uid', 'qid', 'dt', 'day')]
# target编码
logging.info("feature size %s", len(feature_cols))

X_train_all = data.iloc[:len(train_label)][feature_cols]
y_train_all = data.iloc[:len(train_label)]['label']
test = data.iloc[len(train_label):]
del data
assert len(test) == sub_size

logging.info("train shape %s, test shape %s", train_label.shape, test.shape)

fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for index, (train_idx, val_idx) in enumerate(fold.split(X=X_train_all, y=y_train_all)):
    break

X_train, X_val, y_train, y_val = X_train_all.iloc[train_idx][feature_cols], X_train_all.iloc[val_idx][feature_cols],                                  y_train_all.iloc[train_idx],                                  y_train_all.iloc[val_idx]
del X_train_all


# In[11]:


X_train


# In[9]:


import xgboost as xgb
from sklearn.metrics import accuracy_score
import scipy

dtrain = xgb.DMatrix( X_train, label=y_train)
dval=xgb.DMatrix( X_val, label=y_val)



# In[11]:


import warnings
warnings.filterwarnings('default')
num_round = 4000
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 多分类的问题             # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 8,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.8,              # 随机采样训练样本
    'colsample_bytree': 0.8,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.002,                  # 如同学习率
    'seed': 1000,
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'eval_metric': ['logloss', 'error', 'auc'],
    'verbose_eval':True,
    
    
}

# In[2]:


bst = xgb.train( params, dtrain, num_round, evals=[(dval,'zxh')])
dtest = xgb.DMatrix(test[feature_cols])
ans = bst.predict(dtest)

sub['label'] = ans


sub.to_csv('result1211.txt', index=None, header=None, sep='\t')


# In[ ]:




