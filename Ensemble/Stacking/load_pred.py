import numpy as np;

if __name__ == "__main__":
    data = np.load("./pred.npy");
    print(data.shape);
    print(data[0]);