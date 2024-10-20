import numpy as np
from sklearn.model_selection import train_test_split

# 假设输入数据和标签数据以npy文件存储，形状分别为(N, C, T, V, M)和(N, 155)
input_file_path = './data/train_joint.npy'
label_file_path = './data/train_label.npy'

# 加载数据
input_data = np.load(input_file_path)  # 形状为 (N, C, T, V, M)
output_labels = np.load(label_file_path)  # 形状为 (N, 155)

# 训练集和验证集划分
train_joint_A, val_joint_A, train_label_A, val_label_A = train_test_split(
    input_data, output_labels, test_size=0.2, random_state=42
)

# 保存到文件
np.save('train_joint_A.npy', train_joint_A)
np.save('val_joint_A.npy', val_joint_A)
np.save('train_label_A.npy', train_label_A)
np.save('val_label_A.npy', val_label_A)

print("训练集和测试集已划分并保存为 .npy 文件。")
