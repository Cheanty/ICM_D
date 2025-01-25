import pandas as pd
import numpy as np
from config import *
# 假设已经加载了 nodes 数据
nodes = pd.read_csv(f'../{DATA_PATH}/nodes_all.csv', usecols=NODES_LIST)


# 计算距离的函数
def calculate_distance(p1, p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2

# 找到P1和P2对应最近的节点
def find_nearest_node(p1, nodes):
    min_distance = float('inf')
    nearest_node = None
    for idx, row in nodes.iterrows():
        node_location = (row['y'], row['x'])  # 假设y是纬度，x是经度
        if row['y'] > P1[0] or row['x'] > P1[1]:  # 假设y是纬度，x是经度
            continue
        distance = calculate_distance(p1, node_location)
        if distance < min_distance:
            min_distance = distance
            nearest_node = row
    return nearest_node, min_distance

# 找到P1和P2的最近节点
nearest_node_p1, distance_p1 = find_nearest_node(P1, nodes)


# 输出结果
print("Nearest node to P1:")
print("osmid:", nearest_node_p1['osmid'])
print("Distance to P1:", distance_p1)
print("Node location:", (nearest_node_p1['y'], nearest_node_p1['x']))