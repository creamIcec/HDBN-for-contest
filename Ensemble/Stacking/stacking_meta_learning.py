import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import logging

# 设置日志记录
logging.basicConfig(filename='meta_learner_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    "degcn_B_3d": "../scores/Mix_GCN/degcn_B_3d.pkl"
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
    "skateformer_j_3d": "../scores/Mix_Former/skateformer_B_3d.pkl"
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
    X = np.array(data_list).transpose(1, 0, 2)
    #X = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, X)  # 对每个 155 维向量进行 softmax 处理
    X = np.sum(X, axis=1)  # 对每个样本的 n 个 155 维向量加和
    #X = X.reshape(2000, -1)
    print(X.shape);
    y = np.load("test_label_A.npy")  # 使用numpy加载实际的标签
    
    # 注意: 保持数据分布，此处只是为了Stacking, 如果做任何处理将会破坏原始数据
    return X,y

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
        self.fc1 = nn.Linear(input_dim, 256)  # 减少隐藏层的神经元数量以降低模型复杂度 
        self.bn1 = nn.BatchNorm1d(256)  # 添加 Batch Normalization 层
        self.fc2 = nn.Linear(256,512)
        self.bn2 = nn.BatchNorm1d(512)  # 添加 Batch Normalization 层
        self.dropout = nn.Dropout(0.6)  # 增加 Dropout 概率以防止过强
        self.fc3 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # 在第一层之后添加 Dropout
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# 训练元学习器
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=50):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
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
        log_message = f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}"
        print(log_message)
        logging.info(log_message)
        scheduler.step(total_loss / len(train_loader))  # 调整学习率
        
        # 每一轮训练后进行评估
        eval(model, test_loader)

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

if __name__ == "__main__":
    # 加载数据
    X, y = load_data(gcn=True, former=True)

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)

    # 转换为 PyTorch 张量并创建 DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    input_dim = X.shape[1]
    model = MetaLearner(input_dim=input_dim, output_dim=155)

    # 定义损失函数和优化器
    weights = torch.from_numpy(extract_weighted_loss(y_train))
    criterion = FocalLoss(alpha=1, gamma=2, weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练并评估模型
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=80)

    # 保存训练好的模型权重
    torch.save(model.state_dict(), "meta_learner_weights.pth")
    logging.info("元学习器权重已保存为 meta_learner_weights.pth")
    print("元学习器权重已保存为 meta_learner_weights.pth")