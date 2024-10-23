import pickle
import numpy as np
from pyswarm import pso
import time

gcn_names = {
    "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
    "ctrgcn_b_3d": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
    "ctrgcn_j_3d": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
    "ctrgcn_j_3d_resample": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample.pkl",
    #"ctrgcn_b_2d": "../scores/Mix_GCN/ctrgcn_V1_B_2d.pkl",
    #"ctrgcn_j_2d": "../scores/Mix_GCN/ctrgcn_V1_J_2d.pkl",
    #"ctrgcn_bm_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
    #"ctrgcn_jm_2d": "../scores/Mix_GCN/ctrgcn_V1_JM_2d.pkl",
    #"tdgcn_j_2d": "../scores/Mix_GCN/tdgcn_V1_J_2d.pkl",
}

former_names = {
    #"former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    #"former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    #"former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl"
    #"former_jm": "../scores/Mix_Former/mixformer_JM.pkl",
}

#weights_full = [1.8317366,0.32466949,2.,0.42696798,1.14389486,0.08981595,0.7829189,0.058318,0.56409362,1.40943682,0.01963489,0.54166845,1.38688713]

# 加载预处理的数据
def load_data(gcn: bool = False, former: bool = False):
    # 假设每个模型都有自己的特征集，形状为 (2000, 155)
    data_list = []
    if gcn:
        for name in gcn_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    if former:
        for name in former_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    
    data_np = np.array(data_list)
    
    X = data_np.transpose(1, 0, 2)  # X shape: (samples, models, features)
    y = np.load("test_label_A.npy")  # 使用numpy加载实际的标签

    return X, y

def softmax(X):
    return np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 0, X)  # 对每个 155 维向量进行 softmax 处理

def weighted(X, weights):
    # 每个样本的最终预测
    final_pred = np.array([])
    
    for index in range(X.shape[0]):
        weighted_sum = np.zeros(X.shape[2])  # 初始化一个等形的零向量
        for model_index in range(X.shape[1]):
            weighted_sum += weights[model_index] * softmax(X[index][model_index])  # 按权重给各个模型的预测加权
        final_pred = np.append(final_pred, np.argmax(weighted_sum))  # 取权重和的最大值
    
    return final_pred

def loss_function(weights, X, y):
    predictions = weighted(X, weights)
    accuracy = np.mean(predictions == y)
    return -accuracy  # 我们希望最大化准确率，所以返回负值

def optimize_weights_pso(X, y):
    lb = [0] * X.shape[1]  # 下边界为0
    ub = [2] * X.shape[1]  # 上边界为2
    weights, _ = pso(loss_function, lb, ub, args=(X, y), swarmsize=50, maxiter=20, debug=True)
    return weights

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

if __name__ == "__main__":
    X, y = load_data(gcn=True, former=True)
    start_time = time.time()
    optimized_weights = optimize_weights_pso(X, y)
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Optimized Weights (PSO): {optimized_weights}")

    # 使用优化后的权重进行加权预测
    result = weighted(X, optimized_weights)
    acc = accuracy_score(y, result)
    print(f"Accuracy with Optimized Weights (PSO): {acc}%")
