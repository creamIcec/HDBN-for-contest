import os
import argparse
import numpy as np 

CS_train_V1 = [0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                21, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 
                61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 73, 76, 77, 78, 79, 80, 
                81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100, 102, 103, 105, 
                106, 110, 111, 112, 114, 115, 116, 117, 118]

CS_train_V2 = [0, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 
                26, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 
                49, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 
                72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 86, 87, 88, 89, 91, 
                92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 
                108, 109, 110, 111, 112, 113, 114, 115, 117, 118]

def extract_pose(ske_txt_path: str) -> np.ndarray:
    with open(ske_txt_path, 'r') as f: 
        num_frame = int(f.readline()) # the frame num
        joint_data = [] # T M V C
        for t in range(num_frame): # for each frame
            num_body = int(f.readline()) # the body num
            one_frame_data = np.zeros((num_body, 17, 2)) # M 17 2 
            for m in range(num_body): # for each body
                f.readline() # skip this line, e.g. 000 0 0 0 0 0 0 0 0 0
                num_joints = int(f.readline()) # the num joins, equal to 17
                assert num_joints == 17
                for v in range(num_joints): # for each joint
                    xy = np.array(f.readline().split()[:2], dtype = np.float64)
                    one_frame_data[m, v] = xy
            joint_data.append(one_frame_data)
        joint_data = np.array(joint_data)  
    return joint_data # T M 17 2 

def get_max_frame(root_Skeleton_path:str, samples_txt: list) -> int:
    max_frame = 0
    for idx, sample in enumerate(samples_txt):
        ske_path = root_Skeleton_path + '/' + sample
        with open(ske_path, 'r') as f:
            cur_frame = int(f.readline()) # the frame num
            if cur_frame > max_frame: max_frame = cur_frame
    return max_frame 

def project_to_2d(joint_data):
    """
    将3D骨架数据投影为2D，删除 z 坐标。

    参数:
    - joint_data: 5D的Numpy数组, 形状为 (N, C, V, T, M)，代表 N 个样本，每个样本有 C 个通道 (x, y, z)，V 个关节点，T 帧，M 个人。

    返回:
    - 经过投影的 2D 骨架数据，形状为 (N, 2, V, T, M)，即删除掉 z 通道，保留2D信息。
    """
    
    # joint_data 的形状是 (N, C, V, T, M)，我们只保留前两个通道 x, y
    joint_data_2d = joint_data[:, :2, :, :, :]  # 直接切片操作，保留 C 维度中的 x 和 y

    return joint_data_2d

def main(train_data_path: str, data_name: str,  train_label_path: str, label_name: str) -> None:
    # 加载骨架数据和标签
    joint_data = np.load(train_data_path)  # 5维数组 N C V T M
    labels = np.load(train_label_path)     # 1维数组 N，标签

    # 将 3D 数据投影为 2D 数据
    joint_data_2d = project_to_2d(joint_data)

    # 保存处理后的 2D 骨架数据和标签，分别保存


    with open(f"./save_2d_pose/{data_name}.npy", "wb") as f:
        np.save(f, joint_data_2d)  # 保存2D骨架数据
    with open(f"./save_2d_pose/{label_name}.npy", "wb") as f:
        np.save(f, labels)          # 保存标签
    
    print(f"All done! 数据已成功转换为2D并保存!")

def get_parser():
    parser = argparse.ArgumentParser(description = 'extract_2dpose_from_test_dataset') 
    parser.add_argument(
        '--test_dataset_path', 
        type = str,
        default = '../Test_dataset') # It's better to use absolute paths.
    return parser

# python extract_2dpose.py --test_dataset_path ../Test_dataset             
if __name__ == "__main__":
    main('./3dto2d/test_joint_C.npy', "test_joint_C_2d", './3dto2d/polyfill_test_label_C.npy', "polyfill_test_label_C_2d");