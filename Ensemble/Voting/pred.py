import pickle
import numpy as np

gcn_names = {
    "ctrgcn_jm_3d": "../scores/Mix_GCN/test/ctrgcn_V1_JM_3d.pkl",
    "ctrgcn_b_3d": "../scores/Mix_GCN/test/ctrgcn_V1_B_3d.pkl",
    "ctrgcn_j_3d": "../scores/Mix_GCN/test/ctrgcn_V1_J_3d.pkl",
    "ctrgcn_j_3d_resample": "../scores/Mix_GCN/test/ctrgcn_V1_J_3d_resample.pkl",
    "ctrgcn_b_2d": "../scores/Mix_GCN/test/ctrgcn_V1_B_2d.pkl",
    "ctrgcn_j_2d": "../scores/Mix_GCN/test/ctrgcn_V1_J_2d.pkl",
    "ctrgcn_bm_2d": "../scores/Mix_GCN/test/ctrgcn_V1_BM_2d.pkl",
    "ctrgcn_jm_2d": "../scores/Mix_GCN/test/ctrgcn_V1_JM_2d.pkl",
    "tdgcn_j_2d": "../scores/Mix_GCN/test/tdgcn_V1_J_2d.pkl",
}

former_names = {
    "former_bm_r_w_2d": "../scores/Mix_Former/test/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/test/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/test/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/test/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/test/mixformer_B_3d.pkl",
    "former_jm_2d": "../scores/Mix_Former/test/mixformer_JM_2d.pkl",
}


weights = [
           0.888425345498543, 
           0.7548072299032186, 
           1.057142687681162, 
           0.6361776727168669, 
           -0.20625044990175384,
           0.07099191761987358,
           0.2718989922188763, 
           0.060246068027737174,
           0.1492628870982368,
           0.24095834339453534, 
           0.251462666418275, 
           0.13389165554582272,
           0.4633915366414254,
           0.4117076143975217, 
           -0.06925064114488473
           ]

# 加载预处理的数据
def load_data(gcn: bool = False, former: bool = False):
    data_list = []
    if gcn:
        for name in gcn_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(4599)])
            data_list.append(data)
    if former:
        for name in former_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(4599)])
            data_list.append(data)
    
    data_np = np.array(data_list)
    
    X = data_np.transpose(1, 0, 2)  # X shape: (samples, models, features)

    return X

def softmax(X):
    return np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 0, X)  # 对每个 155 维向量进行 softmax 处理

def generate_confidence(X, weights):
    # 每个样本的置信度
    confidences = []
    
    for index in range(X.shape[0]):
        weighted_sum = np.zeros(X.shape[2])  # 初始化一个等形的零向量
        for model_index in range(X.shape[1]):
            weighted_sum += weights[model_index] * softmax(X[index][model_index])  # 按权重给各个模型的预测加权
        confidences.append(weighted_sum)  # 保存每个样本的置信度向量
    
    return np.array(confidences)

if __name__ == "__main__":
    X = load_data(gcn=True, former=True)
    confidences = generate_confidence(X, weights)
    
    # 保存置信度到文件
    np.save("pred.npy", confidences)
    
    print("Confidence scores have been saved to 'pred.npy'")
