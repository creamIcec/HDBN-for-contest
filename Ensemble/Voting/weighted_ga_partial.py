import pickle
import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from deap import base, creator, tools, algorithms
import time
import random
import multiprocessing
from functools import partial

gcn_3d_names = {
    "ctrgcn_jm_3d": "../scores/Mix_GCN/ctrgcn_V1_JM_3d.pkl",
    "ctrgcn_b_3d": "../scores/Mix_GCN/ctrgcn_V1_B_3d.pkl",
    "ctrgcn_j_3d": "../scores/Mix_GCN/ctrgcn_V1_J_3d.pkl",
    "ctrgcn_j_3d_resample": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample.pkl",
    "ctrgcn_j_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_J_3d_resample_rotate.pkl",
    "blockgcn_j_3d": "../scores/Mix_GCN/blockgcn_J_3d.pkl",
    "blockgcn_jm_3d": "../scores/Mix_GCN/blockgcn_JM_3d.pkl",
    "blockgcn_b_3d": "../scores/Mix_GCN/blockgcn_B_3d.pkl",
    "blockgcn_bm_3d": "../scores/Mix_GCN/blockgcn_BM_3d.pkl",
    "ctrgcn_b_3d_resample_rotate": "../scores/Mix_GCN/ctrgcn_V1_B_3d_resample_rotate.pkl",
    "degcn_J_3d": "../scores/Mix_GCN/degcn_J_3d.pkl"
}

gcn_2d_names = {
    "ctrgcn_b_2d": "../scores/Mix_GCN/ctrgcn_V1_B_2d.pkl",
    "ctrgcn_j_2d": "../scores/Mix_GCN/ctrgcn_V1_J_2d.pkl",
    "ctrgcn_bm_2d": "../scores/Mix_GCN/ctrgcn_V1_BM_2d.pkl",
    "ctrgcn_jm_2d": "../scores/Mix_GCN/ctrgcn_V1_JM_2d.pkl",
    "tdgcn_j_2d": "../scores/Mix_GCN/tdgcn_V1_J_2d.pkl",
}

former_3d_names = {
    "former_j_3d": "../scores/Mix_Former/mixformer_J_3d.pkl",
    "former_b_3d": "../scores/Mix_Former/mixformer_B_3d.pkl",
    "former_j_3d_resample_rotate": "../scores/Mix_Former/mixformer_J_3d_resample_rotate.pkl",
}

former_2d_names = {
    "former_bm_r_w_2d": "../scores/Mix_Former/mixformer_BM_r_w_2d.pkl",
    "former_bm_2d": "../scores/Mix_Former/mixformer_BM_2d.pkl",
    "former_j_2d": "../scores/Mix_Former/mixformer_J_2d.pkl",
    "former_jm_2d": "../scores/Mix_Former/mixformer_JM_2d.pkl",
}

# 加载预处理的数据
def load_data(gcn_2d: bool = False, former_2d: bool = False, gcn_3d: bool = False, former_3d: bool = False):
    # 假设每个模型都有自己的特征集，形状为 (2000, 155)
    data_list = []
    if gcn_2d:
        for name in gcn_2d_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    if former_2d:
        for name in former_2d_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    if former_3d:
        for name in former_3d_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    if gcn_3d:
        for name in gcn_3d_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    
    data_np = np.array(data_list)
    
    X = data_np.transpose(1, 0, 2)  # X shape: (samples, models, features)
    y = np.load("test_label_A.npy")  # 使用numpy加载实际的标签

    return data_list, X, y

def softmax(X):
    return np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 0, X)  # 对每个 155 维向量进行 softmax 处理

def weighted(X, weights):
    # 每个样本的最终预测
    final_pred = np.array([])
    
    for index in range(X.shape[0]):
        weighted_sum = np.zeros(X.shape[2])  # 初始化一个等形的零向量
        softmax_values = np.array([softmax(X[index][model_index]) for model_index in range(X.shape[1])])  # 对每个模型计算softmax
        
        # 对每个模型的预测结果按权重加权
        for model_index in range(X.shape[1]):
            weighted_sum += weights[model_index] * softmax_values[model_index]
        
        final_pred = np.append(final_pred, np.argmax(weighted_sum))  # 取权重和的最大值
    
    return final_pred

def loss_function(weights, X, y):
    predictions = weighted(X, weights)
    accuracy = np.mean(predictions == y)
    return -accuracy  # 我们希望最大化准确率，所以返回负值

def evaluate(individual, X, y):
    return -loss_function(individual, X, y),


def optimize_weights_ga(X, y, n_generations=25, population_size=50):
    # 创建遗传算法工具箱
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 使用 partial 将 X 和 y 绑定到 evaluate 函数
    evaluate_with_data = partial(evaluate, X=X, y=y)
    toolbox.register("evaluate", evaluate_with_data)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 注册并行 map
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # 初始化种群
    population = toolbox.population(n=population_size)

    # 运行遗传算法
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, 
                        stats=None, halloffame=None, verbose=True)

    # 获取最优个体
    best_individual = tools.selBest(population, k=1)[0]
    pool.close()
    pool.join()  # 确保所有进程关闭
    return best_individual

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def save_partial_pickle(data_list, weights, gcn_2d: bool, former_2d: bool, gcn_3d: bool, former_3d: bool):

    data = np.array(data_list);  # shape: (models, samples, features)
    #data = data.transpose(1,0,2);  

    if gcn_2d == False and former_2d == False and gcn_3d == False and former_3d == False:
        raise ValueError("请至少指定一种模型集。");
    
    if gcn_2d == True:
        name = "gcn_2d";
    elif former_2d == True:
        name = "former_2d";
    elif gcn_3d == True:
        name = "gcn_3d";
    elif former_3d == True:
        name = "former_3d";

    data_dict = {};
    with open(f"./partial/partial_{name}.pkl", "wb") as f:
        sample_weighted = [];
        for index in range(data.shape[0]):
            sample_weighted.append(data[index] * weights[index]);
        print(sample_weighted[0].shape)
        sample_weighted = np.array(sample_weighted);
        print(sample_weighted.shape);
        #sample_weighted = sample_weighted.transpose(1,0,2);
        sample_weighted = np.sum(sample_weighted, axis=0);
        print(sample_weighted.shape);
        for index in range(sample_weighted.shape[0]):
            data_dict[f"test_{index}"] = sample_weighted[index];
        pickle.dump(data_dict, f);
        print(len(data_dict));
        print(len(data_dict['test_0']));

if __name__ == "__main__":

    gcn_2d = False;
    gcn_3d = False;
    former_2d = False;
    former_3d = False;

    data_list, X, y = load_data(gcn_2d=gcn_2d, gcn_3d=gcn_3d, former_2d=former_2d, former_3d=former_3d)
    start_time = time.time()
    # 使用遗传算法进行优化
    optimized_weights = optimize_weights_ga(X, y)
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Optimized Weights (GA): {optimized_weights}")

    # 使用优化后的权重进行加权预测
    result = weighted(X, optimized_weights)
    acc = accuracy_score(y, result)
    print(f"Accuracy with Optimized Weights (GA): {acc}%")

    # 保存局部集成结果
    save_partial_pickle(data_list, optimized_weights, gcn_2d=gcn_2d, gcn_3d=gcn_3d, former_2d=former_2d, former_3d=former_3d);
