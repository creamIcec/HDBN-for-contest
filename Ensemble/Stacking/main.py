import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import logging

# 设置日志记录
logging.basicConfig(filename='meta_learner_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    logging.info(f"Class distribution: {distro}")
    result = np.zeros(classes, dtype=np.float32);       #保存每个类的权重的数组
    for index, count in enumerate(distro):      #对于distro中的每个元素, 取得它的类编号和样本数    
        result[index] = 1 - count / sample_count;   #计算权重
    
    logging.info(f"Class weights: {result}")
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

    # 使用简单的重复采样方法对数据进行均衡处理
    classes, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    X_resampled, y_resampled = [], []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        num_samples_to_add = max_count - len(cls_indices)
        resampled_indices = np.random.choice(cls_indices, num_samples_to_add, replace=True)
        X_resampled.append(X[cls_indices])
        X_resampled.append(X[resampled_indices])
        y_resampled.extend([cls] * (len(cls_indices) + num_samples_to_add))

    X_resampled = np.vstack(X_resampled)
    y_resampled = np.array(y_resampled)
    return X_resampled, y_resampled

# 定义数据集分割函数
def split_data(X, y, train_ratio=0.8):
    '''
    随机将数据集分割为训练集和测试集。
    :param np.ndarray X: 特征数据。
    :param np.ndarray y: 标签数据。
    :param float train_ratio: 训练集的比例，默认为 0.8。
    :return: 分割后的训练集和测试集 (X_train, X_test, y_train, y_test)。
    '''
    return train_test_split(X, y, train_size=train_ratio, random_state=42)

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
        log_message = f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}"
        print(log_message)
        logging.info(log_message)

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
    log_message = f"Evaluation Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}"
    print(log_message)
    logging.info(log_message)

if __name__ == "__main__":
    # 加载数据
    X, y = load_data(gcn=True, former=False)

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)

    # 转换为 PyTorch 张量并创建 DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = MetaLearner(input_dim=X.shape[1], output_dim=155)

    # 定义损失函数和优化器
    weights = torch.from_numpy(extract_weighted_loss(y_train))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 训练并评估模型
    train(model, train_loader, criterion, optimizer, epochs=50)
    eval(model, test_loader)

    # 保存训练好的模型权重
    torch.save(model.state_dict(), "meta_learner_weights.pth")
    logging.info("元学习器权重已保存为 meta_learner_weights.pth")
    print("元学习器权重已保存为 meta_learner_weights.pth")
