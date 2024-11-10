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
    "blockgcn_bm_3d": "../scores/Mix_GCN/test/blockgcn_BM_3d.pkl",
    "ctrgcn_b_3d_resample_rotate": "../scores/Mix_GCN/test/ctrgcn_V1_B_3d_resample_rotate.pkl",
    "degcn_J_3d": "../scores/Mix_GCN/test/degcn_J_3d.pkl",
    "degcn_B_3d": "../scores/Mix_GCN/test/degcn_B_3d.pkl",
    "degcn_BM_3d": "../scores/Mix_GCN/test/degcn_BM_3d.pkl",
    "tegcn_V1_J_3d": "../scores/Mix_GCN/test/tegcn_V1_J_3d.pkl",
    "tegcn_V1_B_3d": "../scores/Mix_GCN/test/tegcn_V1_B_3d.pkl"
}

former_names = {
    "former_bm_r_w_2d": "../scores/Mix_Former/test/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/test/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/test/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/test/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/test/mixformer_B_3d.pkl",
    "former_j_3d_resample_rotate": "../scores/Mix_Former/test/mixformer_J_3d_resample_rotate.pkl",
    "former_jm_2d": "../scores/Mix_Former/test/mixformer_JM_2d.pkl",
    "former_b_3d_resample_rotate": "../scores/Mix_Former/test/mixformer_B_3d_resample_rotate.pkl",
    "skateformer_b_3d": "../scores/Mix_Former/test/skateformer_B_3d.pkl",
    "skateformer_j_3d": "../scores/Mix_Former/test/skateformer_J_3d.pkl"
}


#weights = [-0.3932759431416317, 1.578663620941732, 2.8718534966520393, -0.7040674026188839, -0.04803901594753768, 0.804353739137729, 0.5684942364123925, 0.2648625015420424, -0.21362505864425924, 0.7272211313760325, 1.8592083256721168, -0.049671497597091246, 1.6419109670031031, 0.5686215349024772, -0.1306379486913656, 1.779273282927985, 1.8652049994424034, 2.686836105600424, 0.8309710689689027, 2.143595543434594, 1.0224760045027292, 0.4155238249376796, 0.5773712827797061, 0.9756010165086924, 0.21146006108742688, 0.5486826163984877, 3.0666412073054667]
weights = [
    1.9110264857820212,
    -0.45230115036601665,
    2.0329118340678805,
    -0.2137147074586212,
    1.0719177302576752,
    -2.5685688688791624,
    0.13268533367898,
    -2.6796443544744273,
    2.933392492233706,
    4.710933343862143,
    4.112379567022963,
    -1.49835525647782,
    4.4473179114161265,
    1.32150628508955,
    3.2450059353876664,
    2.590427685519466,
    3.4127252510186894,
    3.391000548513706,
    2.692721456990763,
    0.49789783147079725,
    1.9989714727193102,
    2.042898417976021,
    -0.593346150642901,
    -0.18263226744378636,
    -1.6052751449090876,
    2.7910241019248008,
    -0.4897537207992642,
    1.5375553380749984,
    3.7486831205707993,
    7.708803728359109,
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
