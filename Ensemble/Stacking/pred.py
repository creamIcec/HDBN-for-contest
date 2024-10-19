import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pickle

former_names = {
    "former_b_m_r_w": "",
    "former_b_m": "",
    "former_j": "",
}

gcn_names = {
    "gcn_b_m": "Mix_GCN/test/ctrgcn_V1_J_3d_bone_vel.pkl",
    "gcn_j": "Mix_GCN/test/ctrgcn_V1_J_3d.pkl"
}

data_size = 4599;

def load_data(gcn: bool = False, former: bool = False):
    data_list = []
    if former:
        for name in former_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(data_size)])
            data_list.append(data)
    if gcn:
        for name in gcn_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(data_size)])
            data_list.append(data)

    # 将所有模型的数据进行拼接
    X = np.concatenate(data_list, axis=1)
    return X

# 定义元学习器模型
class MetaLearner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 减少隐藏层的神经元数量
        self.fc2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 层
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 在第一层之后添加 Dropout
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # 加载数据
    X = load_data(gcn=True, former=False)

    # 转换为 PyTorch 张量并创建 DataLoader
    dataset = TensorDataset(torch.tensor(X))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = MetaLearner(input_dim=X.shape[1], output_dim=155)

    # 加载训练好的模型权重
    model.load_state_dict(torch.load("meta_learner_weights.pth"))
    model.eval()

    # 生成置信度
    confidences = []
    with torch.no_grad():
        for X_batch in dataloader:
            X_batch = X_batch[0].float()
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)  # 计算每个类别的置信度
            confidences.append(probabilities.numpy())

    # 保存置信度到文件
    confidences = np.vstack(confidences)
    np.save("pred.npy", confidences)
    print("置信度文件已保存为 pred.npy")
