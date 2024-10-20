import pickle;
import numpy as np;

if __name__ == "__main__":
    with open('./Mix_Former/mixformer_J.pkl', 'rb') as f:
        score = pickle.load(f);
        print(score["test_0"]);
    with open('./test_label_A.npy', 'rb') as f:
        score = np.load(f);
        print(score.shape);