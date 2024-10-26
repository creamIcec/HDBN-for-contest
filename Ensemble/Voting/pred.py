import pickle
import numpy as np

gcn_names = {
    "ctrgcn_jm_3d": "../scores/Mix_GCN/test/ctrgcn_V1_JM_3d.pkl",
    "ctrgcn_b_3d": "../scores/Mix_GCN/test/ctrgcn_V1_B_3d.pkl",
    "ctrgcn_j_3d": "../scores/Mix_GCN/test/ctrgcn_V1_J_3d.pkl",
    "ctrgcn_j_3d_resample": "../scores/Mix_GCN/test/ctrgcn_V1_J_3d_resample.pkl",
    "ctrgcn_j_3d_resample_rotate": "../scores/Mix_GCN/test/ctrgcn_V1_J_3d_resample_rotate.pkl",
    "ctrgcn_b_2d": "../scores/Mix_GCN/test/ctrgcn_V1_B_2d.pkl",
    "ctrgcn_j_2d": "../scores/Mix_GCN/test/ctrgcn_V1_J_2d.pkl",
    "ctrgcn_bm_2d": "../scores/Mix_GCN/test/ctrgcn_V1_BM_2d.pkl",
    "ctrgcn_jm_2d": "../scores/Mix_GCN/test/ctrgcn_V1_JM_2d.pkl",
    "tdgcn_j_2d": "../scores/Mix_GCN/test/tdgcn_V1_J_2d.pkl",
    "blockgcn_j_3d": "../scores/Mix_GCN/test/blockgcn_J_3d.pkl",
    "blockgcn_jm_3d": "../scores/Mix_GCN/test/blockgcn_JM_3d.pkl",
    "blockgcn_b_3d": "../scores/Mix_GCN/test/blockgcn_B_3d.pkl",
    "blockgcn_bm_3d": "../scores/Mix_GCN/test/blockgcn_BM_3d.pkl"
}

former_names = {
    "former_bm_r_w_2d": "../scores/Mix_Former/test/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/test/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/test/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/test/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/test/mixformer_B_3d.pkl",
    "former_j_3d_resample_rotate": "../scores/Mix_Former/test/mixformer_J_3d_resample_rotate.pkl",
    "former_jm_2d": "../scores/Mix_Former/test/mixformer_JM_2d.pkl",
}


weights = [0.132377534465366, 0.06877511084986354, 0.5340968388065989, 0.5792173020149407, 0.9214838373773134, 0.2403483555752039, 0.2918068623014298, 0.2556687311107895, 0.8409363952453985, 0.6727574872744368, 0.7010583117945367, 0.7475527136571108, 1.1932769273268906, 0.6216619217905547, -0.44110597372528204, 0.547895816287816, 0.16693778195985073, -0.281731522747238, 0.7249454063922118, 0.5105840725689023, 0.44092780561153583]

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
