import pickle;
import numpy as np;

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
    "degcn_BM_3d": "../scores/Mix_GCN/degcn_BM_3d.pkl",
    "tegcn_V1_J_3d": "../scores/Mix_GCN/tegcn_V1_J_3d.pkl",
    "tegcn_V1_B_3d": "../scores/Mix_GCN/tegcn_V1_B_3d.pkl"
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
    "skateformer_b_3d": "../scores/Mix_Former/skateformer_B_3d.pkl",
    "skateformer_j_3d": "../scores/Mix_Former/skateformer_J_3d.pkl"
}

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
    #weights = [4,9,8,5,4,0.5,1,1.2]   # 先gcn后former
    #weights = [0.34798005,1.,0.67848884, 0.10209003, 0.00545483, 0.15840923,0.19249647,0.6435569]
    weights = [1.4668139067381416, 1.1816778798085186, 1.1591087279604277, 
               -0.9248643623406638, 1.1737676062929492, 2.0906060168609195, 1.2345609820218932, -3.032847242194509, 4.370294270393593, 1.0496717844872845, 1.8317057128810603, 0.11062871156959181, 5.149027684302668, 2.885539401460868, 1.0182407713698152, 2.6652137459295644, 3.7565384820528083, 2.723063979264471, 4.8818672170370565, 2.5786177554654275, 0.5524049118916912, 0.2967090288104616, -0.6875776468142071, 0.41173909877491544, 1.8485930929652135, 1.1199114172729159, 0.759049337941745, 0.8234006575695129, 
               3.595903457316497, 1.880281708476684]
    
    # 每个样本的最终预测
    final_pred = np.array([])
    
    for index in range(X.shape[0]):
        weighted_sum = np.zeros(X.shape[2])  # 初始化一个等形的零向量
        for model_index in range(X.shape[1]):
            weighted_sum += weights[model_index] * softmax(X[index][model_index])  # 按权重给各个模型的预测加权
        final_pred = np.append(final_pred, np.argmax(weighted_sum))  # 取权重和的最大值
    
    return final_pred

if __name__ == "__main__":
    X, y = load_data(gcn=True, former=True);
    result = weighted(X);
    total = y.shape[0];
    right_count = 0;
    for result_i, y_i in zip(result, y):
        if(result_i == y_i):
            right_count += 1;
    acc = right_count / total;
    print(f"Accuracy: {acc * 100}%");
