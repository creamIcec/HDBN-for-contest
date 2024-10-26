import numpy as np;
from .feeder_uav import Feeder;
from feeders import tools;
import os;
from sklearn.utils import shuffle;

class FeederUAVHuman3DResampleRotate(Feeder):
    #构造函数初始化
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False 
                 ):  
        #初始化父类
        super().__init__(data_path, label_path, 
                        p_interval, split, 
                        random_choose, random_shift, 
                        random_move, random_rot, 
                        window_size, normalization, 
                        debug, use_mmap, 
                        bone, vel);
        self.load_data();
        if normalization:
            self.get_mean_map()

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
        samples_table = self.load_data_from_txt(os.path.join(os.getcwd(), "feeders/classes_samples.txt"));
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
        # 这里sample_name需要一点判断，还是写上好了
        data_type_name = 'test_' if self.split == 'test' else 'train_';

        if(self.split == 'train'):
            data, label = self.resample(data, label);
            print(f"data: {data.shape}");
            print(f"label: {label.shape}");

        # 大概看了一下，还是按照一起加载的逻辑来写比较好(先暂时这样), 不然其他地方也要改
        if not self.debug:
            self.data = data;
            self.label = label;
            self.sample_name = [data_type_name + str(i) for i in range(len(self.data))];  #还是给一个sample_name吧
        else:
            self.data = data[0:100];
            self.label = label[0:100];
            self.sample_name = [data_type_name + str(i) for i in range(100)];

        # N, T, _ = self.data.shape
        # self.data = self.data.reshape((N, T, 2, 17, 3)).transpose(0, 4, 1, 3, 2)
        #N,C,T,V,M <-我们的
        #N,M,C,V,T <-原来的
        # self.data = self.data.transpose(0, 4, 1, 3, 2)

    def __getitem__(self, index):
        C, T, V, M = self.data[1].shape  #获取到每个样本数据的shape
        data_numpy = self.data[index]    #获取到一个样本
        #print(f"xshape: data_numpy shape:{data_numpy.shape}")
        label = self.label[index]        #获取到对应的label
        data_numpy = np.array(data_numpy)        #将数据转换为numpy兼容格式
        if not(np.any(data_numpy)):
            data_numpy = np.array(self.data[0])
            
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0) #和下一个函数冲突，全0会被报错
        # reshape Tx(MVC) to CTVM

        if(valid_frame_num == 0):
            data_numpy = np.zeros((2,64,17,300));

        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy, channel=3)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        #print(f"xshape: getitem: {data_numpy.shape}");
        return data_numpy, label, index
