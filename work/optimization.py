from config import *
import pandas as pd
import networkx as nx
import heapq

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
        if not row['lanes']: lanes = 2
        else : lanes = row['lanes']
        if not row['maxspeed']: maxspeed = 60
        else: maxspeed = row['maxspeed']
        if not row['oneway']: oneway = 2
        else: oneway = 1
        if not row['length']: length = 100
        else : length = row['length']
        G.add_edge(row['u'], row['v'], osmid=row['osmid'], highway=highway, lanes=lanes, maxspeed=maxspeed, oneway=oneway, length=length)
    return G

def add_degree_to_nodes(G):
    '''度'''
    degree = dict(G.degree())
    for node in G.nodes:
        G.nodes[node]['degree'] = degree[node]

def add_weight_to_edges(G):
    '''通行时间'''
    for u,v,data in G.edges(data=True):
        tm = data['length']/data['maxspeed']/data['highway']
        G[u][v]['weight'] = tm - (G.nodes[u]['highway'] + G.nodes[v]['highway'])*tm*NODE_HIGHWAY_K

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
        G.nodes[node]['traffic'] = G.nodes[node]['traffic'] / 2

def add_traffic_to_edges(G):
    '''道路流量'''
    for u,v,data in G.edges(data=True):
        G[u][v]['traffic'] = 0

def change_wight(G, u, v, traffic):
    pass

def dijkstra(G, start, end):
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
            weight = edge_data.get('weight', 1)  # 获取边的权重，默认为1
            distance = current_distance + weight  # 计算新路径的距离
            
            # 如果新路径比当前已知的距离短，更新
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # 通过 previous_nodes 回溯路径
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    
    path = path[::-1]  # 反转路径，使其从起始节点到目标节点
    return distances[end], path


G = build_network()
add_degree_to_nodes(G)
add_weight_to_edges(G)
add_capacity_to_edges(G)