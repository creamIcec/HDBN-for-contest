import pickle
import numpy as np
import pyswarms as ps
import time


gcn_names = {
    "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
    "ctrgcn_b_3d": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
    "ctrgcn_j_3d": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
    "ctrgcn_j_3d_resample": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample.pkl",
    "ctrgcn_b_2d": "../scores/Mix_GCN/ctrgcn_V1_B_2d.pkl",
    "ctrgcn_j_2d": "../scores/Mix_GCN/ctrgcn_V1_J_2d.pkl",
    "ctrgcn_bm_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
    "ctrgcn_jm_2d": "../scores/Mix_GCN/ctrgcn_V1_JM_2d.pkl",
    "tdgcn_j_2d": "../scores/Mix_GCN/tdgcn_V1_J_2d.pkl",
}

former_names = {
    "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
    "former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
}

#weights_full = [1.8317366,0.32466949,2.,0.42696798,1.14389486,0.08981595,0.7829189,0.058318,0.56409362,1.40943682,0.01963489,0.54166845,1.38688713]
# [0,1,2,8,9,11,12] [1.19698386 1.74628345 1.7523663 1.07512374 0.26303087 0.92568324 1.02412561]
# [0.84578394 1.66633269 1.39956553 0.96742822 0.67005108 0.33380525 0.74805379]

# 加载预处理的数据
def load_data(gcn: bool = False, former: bool = False):
    data_list = []
    if gcn:
        for name in gcn_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    if former:
        for name in former_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    
    data_np = np.array(data_list)
    X = data_np.transpose(1, 0, 2)  # X shape: (samples, models, features)
    y = np.load("test_label_A.npy")  # 加载实际标签
    return X, y

def softmax(X):
    return np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 0, X)

def weighted(X, weights):
    final_pred = np.array([])
    for index in range(X.shape[0]):
        weighted_sum = np.zeros(X.shape[2])
        for model_index in range(X.shape[1]):
            weighted_sum += weights[model_index] * softmax(X[index][model_index])
        final_pred = np.append(final_pred, np.argmax(weighted_sum))
    return final_pred

def weighted_fast(X, weights):
    softmaxed = np.apply_along_axis(softmax, 2, X)  # 对所有数据进行一次性 softmax
    weighted_sum = np.tensordot(weights, softmaxed, axes=([0], [1]))  # 使用张量乘法替代循环
    return np.argmax(weighted_sum, axis=-1)

def loss_function(weights, X, y):
    """
    weights: 是一个 (n_particles, n_models) 的数组，其中每一行是每个粒子的权重向量
    我们需要对每个粒子单独计算损失，并返回损失列表
    """
    n_particles = weights.shape[0]
    losses = []
    
    for i in range(n_particles):
        # 取第 i 个粒子的权重向量
        current_weights = weights[i]
        # 用当前粒子的权重计算预测结果
        predictions = weighted_fast(X, current_weights)
        # 计算准确率
        accuracy = np.mean(predictions == y)
        # 计算损失（负准确率）
        losses.append(-accuracy)
    
    return np.array(losses)

def optimize_weights_pso(X, y, init_pos=None):
    # 定义PSO的参数和边界
    lb = [-1] * X.shape[1]
    ub = [2] * X.shape[1]
    bounds = (lb, ub)
    
    # 定义PSO的选项，比如粒子数、速度和惯性
    options = {'c1': 0.5, 'c2': 0.5, 'w': 1.1}  # 权重参数可以调整

    # 创建PSO优化器
    optimizer = ps.single.GlobalBestPSO(
        n_particles=50,  # 粒子数量
        dimensions=X.shape[1],  # 维度与模型数量相同
        options=options,
        bounds=bounds,
        init_pos=init_pos  # 初始权重（如果提供）
    )
    
    # 优化过程
    cost, pos = optimizer.optimize(loss_function, iters=40, X=X, y=y, n_processes=8)
    return pos

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

if __name__ == "__main__":
    X, y = load_data(gcn=True, former=True)
    start_time = time.time()

    # 假设初始权重是均匀分布的
    init_pos = np.random.uniform(low=0, high=2, size=(50, X.shape[1]))  # 50个粒子，每个粒子有X.shape[1]个权重
    # 初始点
    initial_weights = np.array([
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
           ]);  #目前最好的参数
    init_pos = np.tile(initial_weights, (50,1));
    
    # 优化权重
    optimized_weights = optimize_weights_pso(X, y, init_pos=init_pos)
    end_time = time.time()
    
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Optimized Weights (PSO): {optimized_weights}")

    # 使用优化后的权重进行加权预测
    result = weighted_fast(X, optimized_weights)
    acc = accuracy_score(y, result)
    print(f"Accuracy with Optimized Weights (PSO): {acc}%")