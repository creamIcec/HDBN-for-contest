import pickle
import numpy as np
from scipy.optimize import differential_evolution
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Model paths
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
    "degcn_J_3d": "../scores/Mix_GCN/degcn_J_3d.pkl"
}

former_names = {
    "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
    "former_j_3d_resample_rotate": "../scores/Mix_Former/mixformer_J_3d_resample_rotate.pkl",
    "former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
    "former_b_3d_resample_rotate": "../scores/Mix_Former/mixformer_B_3d_resample_rotate.pkl"
}

# Load data
def load_data(gcn: bool = False, former: bool = False):
    data_list = []
    if gcn:
        for name in gcn_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    if former:
        for name in former_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    
    data_np = np.array(data_list)
    X = data_np.transpose(1, 0, 2)  # X shape: (samples, models, features)
    y = np.load("test_label_A.npy")  # Load true labels
    return X, y

# Softmax function
def softmax(X):
    return np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 0, X)

# Weighted prediction
def weighted_fast(X, weights):
    softmaxed = np.apply_along_axis(softmax, 2, X)  # Apply softmax to all data
    weighted_sum = np.tensordot(weights, softmaxed, axes=([0], [1]))  # Tensor multiplication for weighted sum
    return np.argmax(weighted_sum, axis=-1)

# Loss function
def loss_function(weights, X, y):
    predictions = weighted_fast(X, weights)
    accuracy = np.mean(predictions == y)
    loss = -accuracy
    logger.info(f"Weights: {weights}, Accuracy: {accuracy}, Loss: {loss}")
    return loss

# Differential Evolution optimization
def optimize_weights_de(X, y):
    # Bounds for weights
    bounds = [(0, 2) for _ in range(X.shape[1])]
    
    # Optimization using Differential Evolution
    result = differential_evolution(
        func=lambda weights: loss_function(weights, X, y),
        bounds=bounds,
        strategy='best1bin',
        maxiter=40,
    )
    return result.x

# Accuracy calculation
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

if __name__ == "__main__":
    X, y = load_data(gcn=True, former=True)
    start_time = time.time()

    # Optimize weights using Differential Evolution
    logger.info("Starting optimization using Differential Evolution")
    optimized_weights = optimize_weights_de(X, y)
    end_time = time.time()
    
    logger.info(f"Optimization completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Optimized Weights (Differential Evolution): {optimized_weights}")

    # Use optimized weights for weighted prediction
    result = weighted_fast(X, optimized_weights)
    acc = accuracy_score(y, result)
    logger.info(f"Accuracy with Optimized Weights (Differential Evolution): {acc}%")
