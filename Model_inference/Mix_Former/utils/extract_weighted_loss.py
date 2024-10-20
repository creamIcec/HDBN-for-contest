import numpy as np;

def extract_weighted_loss(labels):
    '''
    用于统计每个类的样本数量并根据它生成每个类的权重。
    :params np.ndarray labels: 加载好的标签数组。
    '''

    sample_count = labels.shape[0];   #样本总数
    distro = np.zeros(classes);       #保存每个类有多少样本的数组，下标是类编号，对应位置的值是那个类的样本数
    
    for i in range(sample_count):     #对于每个样本
        distro[labels[i]] += 1;       #对应类编号的样本数+1

    print(f"distro:{distro}");
    result = np.zeros(classes);       #保存每个类的权重的数组
    for index, count in enumerate(distro):      #对于distro中的每个元素, 取得它的类编号和样本数    
        result[index] = sample_count / count;   #计算权重
    
    
    print(f"result:{result}");
    return result;                    #返回结果