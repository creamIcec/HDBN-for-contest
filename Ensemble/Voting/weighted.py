import pickle;
import numpy as np;

former_names = {
    "former_b_m_r_w": "../scores/Mix_Former/mixformer_BM_r_w.pkl",
    "former_b_m": "../scores/Mix_Former/mixformer_BM_r_w.pkl",
    "former_j": "../scores/Mix_Former/mixformer_J.pkl",
}

gcn_names = {
    "gcn_b_m": "../scores/Mix_GCN/ctrgcn_V1_J_3d_bone_vel.pkl",
    "gcn_j": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
    "gcn_b": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
}

# 加载预处理的数据
def load_data(gcn: bool = False, former: bool = False):
    # 假设每个模型都有自己的特征集，形状为 (2000, 155)
    data_list = []
    if former:
        for name in former_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    if gcn:
        for name in gcn_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)

    data_np = np.array(data_list);
    print(data_np.shape);

   
    X = data_np.transpose(1, 0, 2)
    print(X.shape);
    y = np.load("test_label_A.npy")  # 使用numpy加载实际的标签

    return X,y

def softmax(X):
    return np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 0, X)  # 对每个 155 维向量进行 softmax 处理

def voting_hard(X):
    final_pred = np.array([])
    for index in range(X.shape[0]):
        votes = [np.argmax(item) for item in X[index]];  #对于每个模型产生的对于每个样本的155维向量，通过softmax和argmax取得分数最高的类别
        print(votes);
        final_pred = np.append(final_pred, max(set(votes), key=votes.count))
    return final_pred

def weighted(X):
    # 设置每个模型的权重
    weights = [0.2,0.2,0.2,0.2,0.2,0.2]
    
    # 每个样本的最终预测
    final_pred = np.array([])
    
    for index in range(X.shape[0]):
        weighted_sum = np.zeros(X.shape[2])  # 初始化一个等形的零向量
        for model_index in range(X.shape[1]):
            weighted_sum += weights[model_index] * softmax(X[index][model_index])  # 按权重给各个模型的预测加权
        final_pred = np.append(final_pred, np.argmax(weighted_sum))  # 取权重和的最大值
    
    return final_pred

if __name__ == "__main__":
    X, y = load_data(gcn=True, former=False);
    result = weighted(X);
    total = y.shape[0];
    right_count = 0;
    for result_i, y_i in zip(result, y):
        if(result_i == y_i):
            right_count += 1;
    acc = right_count / total;
    print(f"Accuracy: {acc * 100}%");
