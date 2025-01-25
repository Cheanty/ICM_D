from matplotlib import pyplot as plt
from config import *
import pandas as pd
import networkx as nx
import heapq
import random
import re
# 加载数据
def load_data():
    nodes = pd.read_csv(f'../{DATA_PATH}/nodes_clear.csv', usecols=NODES_LIST)
    edges = pd.read_csv(f'../{DATA_PATH}/edges_clear.csv', usecols=EDGES_LIST)
    return nodes, edges

# 构建网络
def build_network():
    nodes, edges = load_data()
    G = nx.Graph()
    for _, row in nodes.iterrows():
        if row['highway'] in NODE_HIGHWAY_DICT:
            highway = NODE_HIGHWAY_DICT[row['highway']]
        else: highway = 0
        G.add_node(row['osmid'], x=row['x'], y=row['y'],highway=highway)
    for _, row in edges.iterrows():
        if row['highway'] in EDGE_HIGHWAY_DICT:
            highway = EDGE_HIGHWAY_DICT[row['highway']]
        else: highway = 0
        if pd.isna(row['lanes']) or row['lanes'] == '':
            lanes = 2  # 如果是NaN或者空字符串，设置为默认值
        else:
            lanes = int(row['lanes'])  # 转换为整数
        if not row['maxspeed']: maxspeed = 60
        else:
            match = re.match(r'(\d+)', row['maxspeed'])
            maxspeed = float(match.group(1))
        if not row['oneway']: oneway = 2
        else: oneway = 1
        if not row['length']: length = 100
        else : length = float(row['length'])
        G.add_edge(row['u'], row['v'], osmid=row['osmid'], highway=highway, lanes=lanes, maxspeed=maxspeed, oneway=oneway, length=length)
    return G

def add_degree_to_nodes(G):
    '''度'''
    degree = dict(G.degree())
    for node in G.nodes:
        G.nodes[node]['degree'] = degree[node]

def add_time_to_edges(G):
    '''通行时间'''
    for u,v,data in G.edges(data=True):
        if data['highway'] ==0: data['highway'] = 1
        tm = data['length']/data['maxspeed']/data['highway']
        G[u][v]['time'] = tm - (G.nodes[u]['highway'] + G.nodes[v]['highway'])*tm*NODE_HIGHWAY_K

def add_capacity_to_edges(G):
    '''道路容量'''
    for u,v,data in G.edges(data=True):
        G[u][v]['capacity'] = data['lanes']*data['highway']*data['oneway']

def add_traffic_to_node(G):
    '''节点流量'''
    for node in G.nodes:
        G.nodes[node]['traffic'] = 0
        for neighbor, edge_data in G.adj[node].items():
            G.nodes[node]['traffic'] += edge_data['capacity']
        G.nodes[node]['traffic'] = G.nodes[node]['traffic'] / 2 / G.nodes[node]['degree']

def set_traffic_zero_to_edges(G):
    '''道路流量'''
    for u,v,data in G.edges(data=True):
        G[u][v]['traffic'] = 0

def init_graph():
    # G = build_network()
    gml_file_path = f'../{DATA_PATH}/a1_graph.gml'
    G = nx.read_gml(gml_file_path, destringizer=int)
    # 添加大桥
    # G.add_edge(11763173296, 49415813,osmid = "",highway = 60 ,lanes=6,
    #             maxspeed=55,oneway=2,length=2600,
    #             capacity=10000,traffic=0,time=100)
    add_degree_to_nodes(G)
    add_time_to_edges(G)
    add_capacity_to_edges(G)
    add_traffic_to_node(G)
    set_traffic_zero_to_edges(G)
    return G

def change_time(traffic,capacity,time):
    if traffic <= capacity:
        return time
    else: return traffic  * time / capacity

def dijkstra(G, start):
    # 初始化
    distances = {node: float('inf') for node in G.nodes}  # 设置初始距离为无穷大
    distances[start] = 0  # 起始节点到自己的距离是0
    previous_nodes = {node: None for node in G.nodes}  # 记录每个节点的前驱节点
    priority_queue = [(0, start)]  # 优先队列，存储(距离, 节点)
    
    while priority_queue:
        # 获取当前最短路径的节点
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # 如果当前节点的距离已经大于已知的最短距离，跳过
        if current_distance > distances[current_node]:
            continue
        
        # 遍历当前节点的邻居节点
        for neighbor, edge_data in G.adj[current_node].items():
            time = change_time(edge_data['traffic']+G.nodes[current_node]['traffic'], edge_data['capacity'], edge_data['time'])
            distance = current_distance + time  # 计算新路径的距离
            
            # 如果新路径比当前已知的距离短，更新
            if distance < distances[neighbor]:
                edge_data['traffic'] += G.nodes[current_node]['traffic']
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # 返回所有节点的最短路径距离和前驱节点（可以用来回溯路径）
    return distances, previous_nodes

def time_sum(G,order):
    total = 0
    node_time = []
    for i in range(len(order)-1):
        distances, _ = dijkstra(G, order[i])
        node_time.append(distances)
        total += sum(distances.values())
    set_traffic_zero_to_edges(G)
    return total, node_time

# 计算个体的适应度
def fitness(G, order):
    total_time, _ = time_sum(G, order)
    print(total_time)
    return total_time

# 随机生成一个初始路径（个体）
def generate_individual(nodes):
    return random.sample(nodes, len(nodes))

# 选择操作：轮盘赌选择法
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probs = [fitness_score / total_fitness for fitness_score in fitness_scores]
    parents = random.choices(population, probs, k=2)
    return parents

def crossover(parent1, parent2):
    # 随机选择交叉点
    crossover_point1 = random.randint(0, len(parent1) - 2)
    crossover_point2 = random.randint(crossover_point1 + 1, len(parent1) - 1)
    
    # 复制父代1的子序列部分到子代1
    child1 = [None] * len(parent1)
    child1[crossover_point1:crossover_point2 + 1] = parent1[crossover_point1:crossover_point2 + 1]
    
    # 复制父代2的剩余元素到子代1
    pointer = 0
    for i in range(len(parent2)):
        if parent2[i] not in child1:
            while child1[pointer] is not None:
                pointer += 1
            child1[pointer] = parent2[i]
    
    # 同样的方法生成 child2
    child2 = [None] * len(parent1)
    child2[crossover_point1:crossover_point2 + 1] = parent2[crossover_point1:crossover_point2 + 1]
    
    pointer = 0
    for i in range(len(parent1)):
        if parent1[i] not in child2:
            while child2[pointer] is not None:
                pointer += 1
            child2[pointer] = parent1[i]
    
    return child1, child2

# 变异操作：随机交换两个节点
def mutate(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# 主遗传算法函数
def genetic_algorithm(G, generations=100, population_size=50, mutation_rate=0.1):
    # 获取所有节点的osmid
    nodes = list(G.nodes)
    # 初始化种群
    population = [generate_individual(nodes) for _ in range(population_size)]
    
    for generation in range(generations):
        # 计算每个个体的适应度
        fitness_scores = [fitness(G, individual) for individual in population]
        print(f"现在第 {generation} 代,最小适应度为min{fitness_scores}")
        # 如果当前最好的个体已经满足优化条件，提前终止
        if min(fitness_scores) == 0:
            print(f"找到最优解，在第 {generation} 代结束")
            break
        
        # 选择父代
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            
            # 变异操作
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        # 更新种群
        population = new_population[:population_size]

        # 输出当前最优解
        best_individual = population[fitness_scores.index(min(fitness_scores))]
        print(f"Generation {generation + 1}: Best Time = {min(fitness_scores)}, Best Order = {best_individual}")
    
    # 返回最优解
    best_individual = population[fitness_scores.index(min(fitness_scores))]
    return best_individual
def plot_network(G):
    # 提取节点位置
    pos = nx.get_node_attributes(G, 'pos')

    # 绘制图
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_size=5, with_labels=False, edge_color='blue', alpha=0.7)
    plt.title("Baltimore Road Network")
    plt.show()
if __name__ == '__main__':
    # 初始化图


    G = init_graph()
    # first_5_nodes = list(G.nodes(data=True))[:1]
    # for node ,data in first_5_nodes:
    #     print(node)
    #     print(data)

    # 执行遗传算法
    best_order = genetic_algorithm(G,10,10,0.1)

    # 打印最优路径顺序
    print("最优路径顺序:", best_order)
