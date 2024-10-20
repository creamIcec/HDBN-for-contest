import pickle
import numpy as np

gcn_names = {
    "gcn_b_m": "../scores/Mix_GCN/test/ctrgcn_V1_J_3d_bone_vel.pkl",
    "gcn_j": "../scores/Mix_GCN/test/ctrgcn_V1_J_3d.pkl",
    "gcn_b": "../scores/Mix_GCN/test/ctrgcn_V1_B_3d.pkl",
    "gcn_j_2d": "../scores/Mix_GCN/test/ctrgcn_V1_BM_2d.pkl",
    "gcn_bm_2d": "../scores/Mix_GCN/test/ctrgcn_V1_BM_2d.pkl"
}

former_names = {
    #"former_b_m_r_w": "../scores/Mix_Former/mixformer_BM_r_w.pkl",
    #"former_b_m": "../scores/Mix_Former/mixformer_BM_r_w.pkl",
    "former_j": "../scores/Mix_Former/test/mixformer_J.pkl",
}

weights = [4.37866616e-01, 1.00000000e+00, 4.33429203e-01, 7.17090016e-04, 6.09304965e-02, 9.47969778e-01]

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
