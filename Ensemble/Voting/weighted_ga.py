import pickle
import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from deap import base, creator, tools, algorithms
import time
import random
import multiprocessing
from functools import partial

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
    "degcn_B_3d": "../scores/Mix_GCN/degcn_B_3d.pkl",
    "degcn_BM_3d": "../scores/Mix_GCN/degcn_BM_3d.pkl",
    "tegcn_V1_J_3d": "../scores/Mix_GCN/tegcn_V1_J_3d.pkl",
    "tegcn_V1_B_3d": "../scores/Mix_GCN/tegcn_V1_B_3d.pkl"
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
    "skateformer_b_3d": "../scores/Mix_Former/skateformer_B_3d.pkl",
    "skateformer_j_3d": "../scores/Mix_Former/skateformer_J_3d.pkl"
}

initial_weights =  [
    1.9110264857820212,
    -0.45230115036601665,
    2.0329118340678805,
    -0.2137147074586212,
    1.0719177302576752,
    -2.5685688688791624,
    0.13268533367898,
    -2.6796443544744273,
    2.933392492233706,
    4.710933343862143,
    4.112379567022963,
    -1.49835525647782,
    4.4473179114161265,
    1.32150628508955,
    3.2450059353876664,
    2.590427685519466,
    3.4127252510186894,
    3.391000548513706,
    2.692721456990763,
    0.49789783147079725,
    1.9989714727193102,
    2.042898417976021,
    -0.593346150642901,
    -0.18263226744378636,
    -1.6052751449090876,
    2.7910241019248008,
    -0.4897537207992642,
    1.5375553380749984,
    3.7486831205707993,
    7.708803728359109,
]

# 64, 66, 76

#weights = [0.2931585521138443, 0.01628226700493285, 0.6175191316322914, 0.6951296975167518, 0.7206297422357253, 0.7856904981956878, 1.057594514111687, -0.1427378045893291, 1.3867152100500046, -0.7765156290692687, 2.403129591017633, -0.07183271295560331, 2.5025598936507545, 1.6555241819685287, 1.622452787594002, 1.570008806183127, 1.3705078537195217, 3.2309046500985987, 0.9905355181425732, 1.5224143262820893, 0.7184125994846191, 0.23131151423944288, 1.0274011983356686, 0.3097873124138009, 0.01670177475196635, 0.641031868807555, 1.931818771653408]

# 加载预处理的数据
def load_data(gcn: bool = False, former: bool = False):
    # 假设每个模型都有自己的特征集，形状为 (2000, 155)
    data_list = []
    if gcn:
        for name in gcn_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    if former:
        for name in former_names.values():
            with open(name, 'rb') as f:
                data_dict = pickle.load(f)  # 使用pickle加载字典格式的数据
            data = np.array([data_dict[f"test_{i}"] for i in range(2000)])
            data_list.append(data)
    
    data_np = np.array(data_list)
    
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
    return -loss_function(individual,X,y),
    #print(performance);
    #return performance;
    #performance = -loss_function(individual, X, y);
    #penalty = sum(abs(weight) for weight in individual if weight < 0);
    #return performance - penalty,


def optimize_weights_ga(X, y, n_generations=30, population_size=60):
    # 创建遗传算法工具箱
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -2, 10)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 使用 partial 将 X 和 y 绑定到 evaluate 函数
    evaluate_with_data = partial(evaluate, X=X, y=y)
    toolbox.register("evaluate", evaluate_with_data)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.8, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 设置统计信息
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # 保留全局最优个体
    hof = tools.HallOfFame(1)

    # 注册并行 map
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # 初始化种群
    population = toolbox.population(n=population_size)
    for i, individual in enumerate(population):
        individual[:] = initial_weights

    # 在每代开始时加入精英保留操作
    def ea_with_elitism(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
        # 确保全局最优个体保存在 halloffame 中
        if halloffame is not None:
            halloffame.update(population)
        
        # 记录每一代的日志
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # 评估初始种群
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # 开始迭代
        for gen in range(1, ngen + 1):
            # 选择下一代个体
            offspring = toolbox.select(population, len(population) - 1)
            offspring = list(map(toolbox.clone, offspring))

            # 进行交叉操作
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 进行变异操作
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 保证精英个体不被破坏，直接加入下一代
            elite = tools.selBest(population, k=1)[0]
            offspring.append(elite)

            # 评估新一代的个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 更新种群
            population[:] = offspring

            # 更新 Hall of Fame 并记录统计信息
            if halloffame is not None:
                halloffame.update(population)
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

    # 使用我们定义的带精英保留的遗传算法
    population, logbook = ea_with_elitism(
        population, toolbox, cxpb=0.5, mutpb=0.6, ngen=n_generations, 
        stats=stats, halloffame=hof, verbose=True
    )

    # 关闭进程池
    pool.close()
    pool.join()

    # 返回全局最优个体
    return hof[0]

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

if __name__ == "__main__":
    X, y = load_data(gcn=True, former=True)
    start_time = time.time()
    # 使用遗传算法进行优化
    optimized_weights = optimize_weights_ga(X, y, n_generations=20, population_size=60)
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Optimized Weights (GA): {optimized_weights}")

    # 使用优化后的权重进行加权预测
    result = weighted(X, optimized_weights)
    acc = accuracy_score(y, result)
    print(f"Accuracy with Optimized Weights (GA): {acc}%")
