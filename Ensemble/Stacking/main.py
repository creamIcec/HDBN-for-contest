import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle

former_names = {
    "former_b_m_r_w": "Mix_Former/mixformer_BM_r_w.pkl",
    "former_b_m": "Mix_Former/mixformer_BM_r_w.pkl",
    "former_j": "Mix_Former/mixformer_J.pkl",
}

gcn_names = {
    "gcn_b_m": "Mix_GCN/ctrgcn_V1_J_3d_bone_vel.pkl",
    "gcn_j": "Mix_GCN/ctrgcn_V1_J_3d.pkl"
}

def extract_weighted_loss(labels):
    '''
    用于统计每个类的样本数量并根据它生成每个类的权重。
    :params np.ndarray labels: 加载好的标签数组。
    '''
    classes = 155;

    sample_count = labels.shape[0];   #样本总数
    distro = np.zeros(classes);       #保存每个类有多少样本的数组，下标是类编号，对应位置的值是那个类的样本数
    
    for i in range(sample_count):     #对于每个样本
        distro[labels[i]] += 1;       #对应类编号的样本数+1

    print(f"distro:{distro}");
    result = np.zeros(classes, dtype=np.float32);       #保存每个类的权重的数组
    for index, count in enumerate(distro):      #对于distro中的每个元素, 取得它的类编号和样本数    
        result[index] = 1 - count / sample_count;   #计算权重
    
    
    print(f"result:{result}");
    return result;                    #返回结果

# 加载预处理的数据
def load_data(gcn: bool = False, former: bool = False):
    # 假设每个模型都有自己的特征集，形状为 (2000, 155)
    data_list = []
    if former:
        for name in former_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    if gcn:
        for name in gcn_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)

    # 将所有模型的数据进行拼接
    X = np.concatenate(data_list, axis=1)
    y = np.load("test_label_A.npy")  # 使用numpy加载实际的标签
    return X, y

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

# 训练元学习器
def train(model, dataloader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.float(), y_batch.long()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}")

# 评估元学习器
def eval(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.float(), y_batch.long()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # 加载数据
    X, y = load_data(gcn=False, former=True)

    # 定义交叉验证
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # 初始化元学习器模型
    input_dim = X.shape[1]
    output_dim = 155  # 类别数

    # 交叉验证
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X)):
        print(f'Fold {fold+1}/{k_folds}')
        
        # 分割数据
        X_train, X_test = X[train_ids], X[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]

        # 转换为 PyTorch 张量并创建 DataLoader
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 初始化模型
        model = MetaLearner(input_dim, output_dim)

        # 定义损失函数和优化器
        # 带权重交叉熵损失
        weights = torch.from_numpy(extract_weighted_loss(y));
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # 训练并评估模型
        train(model, train_loader, criterion, optimizer, epochs=50)
        eval(model, test_loader)
