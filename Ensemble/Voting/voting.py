import pickle;
import numpy as np;

gcn_names = {
    "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
    "ctrgcn_b_3d": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
    "ctrgcn_j_3d": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
    "ctrgcn_j_3d_resample": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample.pkl",
    "ctrgcn_b_2d": "../scores/Mix_GCN/ctrgcn_V1_B_2d.pkl",
    "ctrgcn_j_2d": "../scores/Mix_GCN/ctrgcn_V1_J_2d.pkl",
    #"ctrgcn_bm_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
    #"ctrgcn_jm_2d": "../scores/Mix_GCN/ctrgcn_V1_JM_2d.pkl",
    #"tdgcn_j_2d": "../scores/Mix_GCN/tdgcn_V1_J_2d.pkl",
}

former_names = {
    #"former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    #"former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    #"former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
    #"former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
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


if __name__ == "__main__":
    X, y = load_data(gcn=True, former=True);
    result = voting_hard(X);
    total = y.shape[0];
    right_count = 0;
    for result_i, y_i in zip(result, y):
        #print(int(result_i), y_i);
        if(result_i == y_i):
            right_count += 1;
    acc = right_count / total;
    print(f"Accuracy: {acc * 100}%");