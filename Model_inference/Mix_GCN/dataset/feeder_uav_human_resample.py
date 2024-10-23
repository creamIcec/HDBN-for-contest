from .feeder_xyz import Feeder;
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from . import tools;
from sklearn.utils import shuffle;
import os;

coco_pairs = [(10, 8), (8, 6), (9, 7), (7, 5), 
                    (15, 13), (13, 11),(16, 14), (14, 12), 
                    (11, 5), (12, 6), (11, 12), (5, 6), 
                    (5, 0), (6, 0), (1, 0), (2, 0), 
                    (3, 1), (4, 2)]
class FeederUAVHumanResample(Feeder):
    #构造函数初始化
    def __init__(self, data_path: str, label_path: str,   #这里改成label_path, 因为我们比赛的数据集和标签分开的
                 data_split: str, #还是加回去data_split 
                 p_interval: list=[0.95], window_size: int=64,
                 bone: bool=False, vel: bool=False,
                 debug=False,  # 添加一个debug好了, 便于调试我们的feeder正不正确 
                 ):  
        #初始化父类
        super().__init__(data_path, data_split, p_interval, window_size, bone, vel);
        #初始化自己的属性
        self.label_path = label_path;
        self.debug = debug;
        self.load_data();

    def load_data_from_txt(self, file_path):
        # 创建一个列表来存放数据
        data = []

        # 打开文件并读取每一行
        with open(file_path, 'r') as file:
            for line in file:
                # 去掉首尾的空格符并按冒号分割，取得冒号后的部分
                value = int(line.strip().split(':')[1])
                data.append(value)

        # 将列表转换为 NumPy 数组，并返回
        return np.array(data)

    #重采样
    #1. 设置一个样本数阈值
    #2. 如果没有下一个类，则结束; 否则取得预先打好的表中的下一个类的样本数n
    #3. 如果n小于阈值，则进入4; 否则进入2
    #4. 从data中提取出所有这个类的样本, 存入列表a中
    #5. 计算阈值 - 样本数，得到缺少的样本数s
    #6. 以高斯分布为算法，以[0,a的长度)为范围，生成s个随机数作为新采样的样本的下标,存入列表b中
    #7. 将a中下标在b中的样本复制一次, 放入列表c中
    #8. 合并列表a和c得到结果, 进入2
    def resample(self, data, labels):
        threshold = 105;
        samples_table = self.load_data_from_txt(os.path.join(os.getcwd(), "dataset/classes_samples.txt"));
        result = [];
        result_labels = [];
        for index, samples in enumerate(samples_table):
            diff = samples - threshold;
            indices = np.where(labels == index)[0];
            class_data = data[indices];
            if(diff < 0):
                 # 生成 diff 个均匀分布的随机数，范围 [0, samples)
                resample_indices = np.random.randint(0, samples, -diff);
                extra_samples = [class_data[i].copy() for i in resample_indices];
                result_temp = np.concatenate((class_data, extra_samples), axis=0);
                result.append(result_temp);
            else:
                result.append(class_data);
            label_length = samples if diff > 0 else threshold;
            result_labels.append(np.full(label_length, index));
        X, y = shuffle(result, result_labels, random_state=42);
        result_X = np.concatenate(X, axis=0);
        result_y = np.concatenate(y, axis=0);
        return result_X, result_y;
                

    # 加载数据的方法
    def load_data(self):

        # 两个都加载, 按照不同阶段返回东西就好
        # data和label可以代表训练集的，也可以代表测试集的
        data = np.load(self.data_path);
        label = np.load(self.label_path);

        if(self.data_split == 'train'):
            data, label = self.resample(data, label);
            print(f"data: {data.shape}");
            print(f"label: {label.shape}");

        # 这里sample_name需要一点判断，还是写上好了
        data_type_name = 'test_' if self.data_split == 'test' else 'train_';

        # 刚刚大概看了一下，还是按照一起加载的逻辑来写比较好(先暂时这样), 不然其他地方也要改
        if not self.debug:
            self.data = data;
            self.label = label;
            self.sample_name = [data_type_name + str(i) for i in range(len(self.data))];  #还是给一个sample_name吧
        else:
            self.data = data[0:100];
            self.label = label[0:100];
            self.sample_name = [data_type_name + str(i) for i in range(100)];