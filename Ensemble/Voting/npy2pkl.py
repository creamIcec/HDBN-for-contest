import pickle;
import numpy as np;

if __name__ == "__main__":
    npy_list = [("tegcn_V1_B_3d", "../scores/Mix_GCN/tegcn_V1_B_3d.npy")];

    for index, (name, path) in enumerate(npy_list):
        data = {}    
        npy_data = np.load(path);
        for sample in range(npy_data.shape[0]):
            data[f"test_{sample}"] = npy_data[sample];
        with open(f"../scores/Mix_GCN/{name}.pkl", "wb") as f:
            pickle.dump(data, f);