import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

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
    
    X = data_np.transpose(1, 0, 2).reshape(data_np.shape[1], -1)  # 将 X 形状调整为 (samples, models * features)
    y = np.load("test_label_A.npy")  # 使用numpy加载实际的标签

    return X, y

if __name__ == "__main__":
    X, y = load_data(gcn=True, former=True)
    
    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义基学习器
    base_learners = [
        ('lr', LogisticRegression(max_iter=1000)),
    ]
    
    # 定义元学习器（堆叠分类器）
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
    
    # 训练堆叠模型
    stacking_model.fit(X_train, y_train)
    
    # 在测试集上进行预测
    y_pred = stacking_model.predict(X_test)
    
    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy with Stacking Meta-Learning: {acc * 100:.2f}%")
