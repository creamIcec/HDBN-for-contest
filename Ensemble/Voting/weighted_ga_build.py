import pickle
import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from deap import base, creator, tools, algorithms
import time
import random
import multiprocessing
from functools import partial

partial_names = {
    "former_2d": "./partial/partial_former_2d.pkl",
    "former_3d": "./partial/partial_former_3d.pkl",
    "gcn_2d": "./partial/partial_gcn_2d.pkl",
    "gcn_3d": "./partial/partial_gcn_3d.pkl"
}

# 加载预处理的数据
def load_data(former_2d: bool, former_3d: bool, gcn_2d: bool, gcn_3d: bool):
    # 假设每个模型都有自己的特征集，形状为 (2000, 155)
    data_list = []
    if gcn_2d:
        name = partial_names["gcn_2d"];
        with open(name, 'rb') as f:
            data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
        data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
        data_list.append(data)
    if gcn_3d:
        name = partial_names["gcn_3d"];
        with open(name, 'rb') as f:
            data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
        data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
        data_list.append(data)
    if former_2d:
        name = partial_names["former_2d"];
        with open(name, 'rb') as f:
            data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
        data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
        data_list.append(data)
    if former_3d:
        name = partial_names["former_3d"];
        with open(name, 'rb') as f:
            data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
        data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
        data_list.append(data)
    
    data_np = np.array(data_list)

    print(data_np.shape)
    
    X = data_np.transpose(1, 0, 2)  # X shape: (samples, models, features)
    y = np.load("test_label_A.npy")  # 使用numpy加载实际的标签

    return X, y

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


def optimize_weights_ga(X, y, n_generations=30, population_size=50):
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

if __name__ == "__main__":

    gcn_2d = True;
    gcn_3d = True;
    former_2d = True;
    former_3d = True;

    X, y = load_data(gcn_2d=gcn_2d, gcn_3d=gcn_3d, former_2d=former_2d, former_3d=former_3d);
    start_time = time.time()
    # 使用遗传算法进行优化
    optimized_weights = optimize_weights_ga(X, y, n_generations=30)
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Optimized Weights (GA): {optimized_weights}")

    # 使用优化后的权重进行加权预测
    result = weighted(X, optimized_weights)
    acc = accuracy_score(y, result)
    print(f"Accuracy with Optimized Weights (GA): {acc}%")
