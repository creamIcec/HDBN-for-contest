import numpy as np

if __name__ == "__main__":
    
    # 生成形状为 (4599,) 的随机数据
    new_data = np.random.rand(4599)

    # 保存为 npy 文件
    np.save("new_data.npy", new_data)
    print("新数据文件已保存为 new_data.npy")