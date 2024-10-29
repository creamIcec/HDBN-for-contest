import pickle
import numpy as np
import pyswarms as ps
import time

gcn_names = {
    "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
    "ctrgcn_b_3d": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
    "ctrgcn_j_3d": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
    "ctrgcn_j_3d_resample": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample.pkl",
    "ctrgcn_j_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample_rotate.pkl",
    "ctrgcn_b_2d": "../scores/Mix_GCN/ctrgcn_V1_B_2d.pkl",
    "ctrgcn_j_2d": "../scores/Mix_GCN/ctrgcn_V1_J_2d.pkl",
    "ctrgcn_bm_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
    "ctrgcn_jm_2d": "../scores/Mix_GCN/ctrgcn_V1_JM_2d.pkl",
    "tdgcn_j_2d": "../scores/Mix_GCN/tdgcn_V1_J_2d.pkl",
    "blockgcn_j_3d": "../scores/Mix_GCN/blockgcn_J_3d.pkl",
    "blockgcn_jm_3d": "../scores/Mix_GCN/blockgcn_JM_3d.pkl",
    "blockgcn_b_3d": "../scores/Mix_GCN/blockgcn_B_3d.pkl",
    "blockgcn_bm_3d": "../scores/Mix_GCN/blockgcn_BM_3d.pkl",
    "ctrgcn_b_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_B_3d_resample_rotate.pkl",
    "degcn_J_3d": "../scores/Mix_GCN/degcn_J_3d.pkl",
    "degcn_B_3d": "../scores/Mix_GCN/degcn_B_3d.pkl",
    "tegcn_V1_J_3d": "../scores/Mix_GCN/tegcn_V1_J_3d.pkl"
}

former_names = {
    "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
    "former_j_3d_resample_rotate": "../scores/Mix_Former/mixformer_J_3d_resample_rotate.pkl",
    "former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
    "former_b_3d_resample_rotate": "../scores/Mix_Former/mixformer_B_3d_resample_rotate.pkl",
    "skateformer_j_3d": "../scores/Mix_Former/skateformer_B_3d.pkl",
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
    ub = [4] * X.shape[1]
    bounds = (lb, ub)
    
    # 定义PSO的选项，比如粒子数、速度和惯性
    options = {'c1': 0.5, 'c2': 0.5, 'w': 5.0}  # 权重参数可以调整

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
    init_pos = np.random.uniform(low=-1, high=4, size=(50, X.shape[1]))  # 50个粒子，每个粒子有X.shape[1]个权重
    # 初始点
    
    # 77.3
    #initial_weights = np.array([0.132377534465366, 0.06877511084986354, 0.5340968388065989, 0.5792173020149407, 0.9214838373773134, 0.2403483555752039, 0.2918068623014298, 0.2556687311107895, 0.8409363952453985, 0.6727574872744368, 0.7010583117945367, 0.7475527136571108, 1.1932769273268906, 0.6216619217905547, -0.44110597372528204, 0.547895816287816, 0.16693778195985073, -0.281731522747238, 0.7249454063922118, 0.5105840725689023, 0.44092780561153583]);  #目前最好的参数
     
    #
    ''' 
    initial_weights = np.array([-0.19540389,0.56044536,1.85167447,1.41851924,
                                1.7006764,1.28111759,1.15653304,0.80695606,
                                1.13790083,1.61385519,1.41323749,1.02417482,
                                1.90221028,1.0301845,1.09568457,1.82490868,
                                0.30463735,1.05411108,-0.16420213,0.65185975,
                                1.05092157,-0.63822924])
    '''
    initial_weights = [0.2931585521138443, 0.01628226700493285, 0.6175191316322914, 0.6951296975167518, 0.7206297422357253, 0.7856904981956878, 1.057594514111687, -0.1427378045893291, 1.3867152100500046, -0.7765156290692687, 2.403129591017633, -0.07183271295560331, 2.5025598936507545, 1.6555241819685287, 1.622452787594002, 1.570008806183127, 1.3705078537195217, 3.2309046500985987, 0.9905355181425732, 1.5224143262820893, 0.7184125994846191, 0.23131151423944288, 1.0274011983356686, 0.3097873124138009, 0.01670177475196635, 0.641031868807555, 1.931818771653408]

    
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