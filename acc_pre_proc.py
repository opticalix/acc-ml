#!/usr/bin/python3
# TODO 尝试其他特征 找水平分量和垂直分量
import numpy as np
import pandas as pd
import os
import math
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from enum import Enum
from sklearn import datasets
from sklearn.manifold import TSNE
import time
from svmutil import *

from tsfeature import feature_core
from tsfeature.feature_time import Feature_time
from tsfeature.feature_fft import Feature_fft


class Hand(Enum):
    LEFT = 0
    RIGHT = 1


class State(Enum):
    STILLNESS = 1
    STAND = 2
    WALK = 3
    RUN = 4
    RIDE = 5
    SLEEP = 6


acc_data_dir = '../data/acc_tag_data/'
acc_data_file = '../data/acc_tag_data/acc-walk-1052757335-1560925892.txt'
svm_model_file_name = 'acc-to-state-model'
acc_hz = 25
seg_sec = 2
head_skip_sec = 1
tail_skip_sec = 2


def load_dataset(filename: str, label: int):
    data_mat = []
    label_mat = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_arr = line.strip().split(',')
            data_mat.append([float(line_arr[0]), float(line_arr[1]), float(line_arr[2])])
            label_mat.append(label)
        head_skip_cnt = head_skip_sec * acc_hz * seg_sec
        tail_skip_cnt = tail_skip_sec * acc_hz * seg_sec
        return data_mat[head_skip_cnt:len(data_mat) - tail_skip_cnt], label_mat[
                                                                      head_skip_cnt:len(data_mat) - tail_skip_cnt]


def get_according_state_val(filename: str):
    State.__members__.items()
    for name, member in State.__members__.items():
        hand = Hand.LEFT.value
        if filename.find(Hand.RIGHT.name.lower()) > 0:
            hand = Hand.RIGHT.value
        if filename.find(name.lower()) > 0:
            return name.lower(), member.value, hand


def combine_acc(row):
    sum = 0
    for i in range(0, len(row)):
        sum += np.square(row[i])
    return math.pow(sum, 1 / 2)


# 返回：每行对应2s的数据 25hz故50列 每个数值代表一个合加速度
def pre_proc(data, combine_func):
    acc_kind_cnt = 1
    feat_cnt = 4
    data_len = len(data)
    data_cnt_per_seg = acc_hz * seg_sec
    # discard data < 2s
    row_cnt = int(data_len / data_cnt_per_seg) - 1
    if row_cnt > 0:
        # ret = np.empty([row_cnt, feat_cnt], dtype=float)
        # 先存cb_acc
        cb_acc_tmp = np.empty([row_cnt, data_cnt_per_seg])
        for i in range(0, data_len):
            seg_id = int(i / data_cnt_per_seg)
            if seg_id < cb_acc_tmp.shape[0]:
                cb_acc_tmp[seg_id, i % data_cnt_per_seg] = combine_func(data[i])
        return cb_acc_tmp
    else:
        return None


def create_feature_with_label(data, label):
    tmp = np.empty([data.shape[0], 4], dtype=float)
    tmp[:, 0] = data.mean(axis=1)
    tmp[:, 1] = data.std(axis=1)
    tmp[:, 2] = stats.skew(data, axis=1)
    tmp[:, 3] = stats.kurtosis(data, axis=1)
    feature_with_label = pd.DataFrame(tmp, columns=['mean', 'std', 'skew', 'kurtosis'])
    feature_with_label['label'] = label
    return feature_with_label

# feature_with_label: ft1 ft2 label
# 多个state 绘制在一张图上.
# 先转换成
# f1 v1
# f1 v2
# f2 v3
# ...
def plot(feature_with_label, axes, plot_idx, label_name):
    to_draw = pd.DataFrame(columns=['feature', 'value'])
    rows = feature_with_label.shape[0]
    cols = feature_with_label.shape[1]
    for i in range(0, rows * (cols - 1)):
        group_idx = int(i / rows)
        f = feature_with_label.columns.values[group_idx]
        v = feature_with_label.iloc[i % rows, group_idx]
        to_draw.loc[i] = [f, v]

    ax = sns.stripplot(x="feature", y="value", data=to_draw, ax=axes[plot_idx])
    ax.set_xlabel(label_name)
    print('label_name=%s plot_idx=%s' % (label_name, plot_idx))


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


# 对合加速度的几个简单特征的可视化
def acc_plot_1():
    plt.figure()
    plot_idx = 0
    f, axes = plt.subplots(1, 3)
    for dir_name in os.listdir(acc_data_dir):
        child = os.path.join('%s/%s' % (acc_data_dir, dir_name))
        if os.path.isfile(child):
            label_name, label_val = get_according_state_val(child)
            data, labels = load_dataset(child, label_val)
            # if len(label) > 0 and label[0] == State.STILLNESS.value:
            combined_acc = pre_proc(data, combine_func=combine_acc)
            feature_with_label = create_feature_with_label(combined_acc, label=labels[0])
            plot(feature_with_label, axes, plot_idx, label_name)
            plot_idx += 1
    plt.show()


# 使用t_sne后的可视化
def t_sne_plot():
    all_ft = pd.DataFrame(columns=['mean', 'var', 'std', 'mode', 'max', 'min', 'over_zero', 'range',
                                   'dc', 'shape_mean', 'shape_var', 'shape_std', 'shape_skew', 'shape_kurt', 'amp_mean', 'amp_var', 'amp_std', 'amp_skew', 'amp_kurt'])
    all_label = pd.Series()
    all_idx = 0
    for dir_name in os.listdir(acc_data_dir):
        child = os.path.join('%s/%s' % (acc_data_dir, dir_name))
        if os.path.isfile(child):
            label_name, label_val = get_according_state_val(child)
            data, labels = load_dataset(child, label_val)
            combined_acc = pre_proc(data, combine_func=combine_acc)
            for i in range(0, combined_acc.shape[0]):
                ft19 = feature_core.get_feature(combined_acc[i, :])
                all_ft.loc[all_idx] = ft19
                all_label.loc[all_idx] = label_val
                all_idx += 1
    # 删掉空项
    all_ft = all_ft.drop('shape_kurt', axis=1)
    # print(all_ft)
    # print(all_label)

    # 用tsne降维后绘制
    tsne = TSNE(n_components=2, perplexity=40, early_exaggeration=12, init='pca', random_state=0)
    t0 = time.time()
    result = tsne.fit_transform(all_ft.values)
    fig = plot_embedding(result, all_label.values,
                         't-SNE embedding of the data (time %.2fs)'
                         % (time.time() - t0))
    plt.show(fig)


def save_as_libsvm_fmt():
    return


#refine features
def acc_fit_test():
    all_ft = pd.DataFrame(columns=['hand',
                                   'mean', 'var', 'std', 'mode', 'max', 'min',
                                   'fft_dc', 'fft_top_idx', 'fft_top'
                                   ])
    all_label = pd.Series()
    all_idx = 0
    # pred_ft = pd.DataFrame(columns=['hand',
    #                                 'mean', 'var', 'std', 'mode', 'max', 'min', 'over_zero', 'range',
    #                                'fft_dc', 'fft_max', 'fft_top_freqs'])
    # pred_label = pd.Series()
    # pred_idx = 0

    for dir_name in os.listdir(acc_data_dir):
        child = os.path.join('%s/%s' % (acc_data_dir, dir_name))
        if os.path.isfile(child):
            label_name, label_val, hand = get_according_state_val(child)
            data, labels = load_dataset(child, label_val)
            if dir_name.find('predict') < 0:
                combined_acc = pre_proc(data, combine_func=combine_acc)
                for i in range(0, combined_acc.shape[0]):
                    fts = get_feature(combined_acc[i, :])
                    # add hand
                    fts.insert(0, hand)
                    all_ft.loc[all_idx] = fts
                    all_label.loc[all_idx] = label_val
                    all_idx += 1

    print(all_ft)

    # -v 10
    m = svm_train(all_label.values.tolist(), all_ft.values.tolist(), '-s 0 -t 2 -d 3 -r 0 -g 0.0001 -c 100')
    # model_file = 'D:\\Workspace\\Python\\data\\acc_tag_data\\' + svm_model_file_name + '_' + time.strftime("%Y-%m-%d_%H:%M:%S") + '.model'
    # model_file = acc_data_dir + svm_model_file_name + "_" + time.strftime("%Y-%m-%d_%H:%M:%S")
    model_file = svm_model_file_name
    svm_save_model(model_file, m)
    # 使用-v交叉验证时 m不能predict了
    # p_label, p_acc, p_val = svm_predict(pred_label.values.tolist(), pred_ft.values.tolist(), m)
    # print(p_label)


# 使用lib计算features
# 加归一化 加fft特征 能将cva提高到97+%
def get_feature(arr):
    # 归一化
    arr = arr / len(arr)

    feature_list = list()
    # get time domain features
    feature_time = list()
    time = Feature_time(arr)
    feature_time.append(time.time_mean())
    feature_time.append(time.time_var())
    feature_time.append(time.time_std())
    feature_time.append(time.time_mode())
    feature_time.append(time.time_max())
    feature_time.append(time.time_min())
    feature_list.extend(feature_time)
    # get frequency domain features
    feature_fft = list()
    fft = Feature_fft(arr)
    feature_fft.append(fft.fft_dc())
    # fft_max, fft_top_freqs是tuple 不能直接作为feature
    # topK和max还不一样，为啥？argmax和argsort的区别
    top_idxs, top_freqs = fft.fft_topk_freqs(1)
    feature_fft.append(top_idxs[0])
    feature_fft.append(top_freqs[0])
    feature_list.extend(feature_fft)
    return feature_list



# acc_main()
# t_sne_plot()
acc_fit_test()