# 导入库
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import differential_evolution
from config import *


# 数据清理
def clear_data():
    # 加载 noeds_all.csv
    # 该文件包含了所有的节点的数据
    nodes = pd.read_csv(f'../{DATA_PATH}/nodes_all.csv', usecols=NODES_LIST)

    # 加载 edges_all.csv
    # 该文件包含了所有的边的数据
    edges = pd.read_csv(f'../{DATA_PATH}/edges_all.csv', usecols=EDGES_LIST)

    # 筛选出在多边形内的节点
    polygon = Polygon(PCOORDS)
    nodes['inside_polygon'] = nodes.apply(
        lambda row: polygon.contains(Point(row['x'], row['y'])),
        axis=1
    )
    nodes_inside_polygon = nodes[nodes['inside_polygon']].drop(columns=['inside_polygon'])
    # 提取多边形内的节点 osmid
    node_ids_inside_polygon = set(nodes_inside_polygon['osmid'])
    
    # 筛选出多边形内的边
    edges_inside_polygon = edges[
        (edges['u'].isin(node_ids_inside_polygon)) &
        (edges['v'].isin(node_ids_inside_polygon))
    ]
    # 输出结果
    nodes_inside_polygon = nodes_inside_polygon[NODES_LIST]
    edges_inside_polygon = edges_inside_polygon[EDGES_LIST]
    
    # 加载 MDOT_SHA_Annual_Averyage_Daily_Traffic_Baltimore.csv
    # 该文件包含了巴尔的摩地区的交通流量数据
    traffic_data = pd.read_csv(f'../{DATA_PATH}/MDOT_SHA_Annual_Average_Daily_Traffic_Baltimore.csv', usecols= TRAFFIC_DATA_LIST)

    # 筛选 'node start' 和 'node(s) end' 都不为空的行
    traffic_data_filtered = traffic_data[
        traffic_data['node start'].notna() & traffic_data['node(s) end'].notna()
    ]
    # 计算 AADT 和 AAWDT 的加权平均值
    traffic_data_filtered['AADTA'] = traffic_data_filtered[AADTS_COLUMNS].mean(axis=1)
    traffic_data_filtered['AAWDTA'] = traffic_data_filtered[AAWDTS_COLUMNS].mean(axis=1)
    
    # 删除原始的 AADT 和 AAWDT 字段
    traffic_data_filtered.drop(columns=AADTS_COLUMNS + AAWDTS_COLUMNS, inplace=True)
    traffic_clear = traffic_data_filtered
    # 利用traffic_clear的数据,继续清理数据和计算每条边的流量
    node_used = set()
    # 提取 'node start' 字段中的节点集合并加入 node_used
    for node_set in traffic_data_filtered['node start'].dropna():
        node_used.update(eval(node_set))  # 将字符串表示的集合转换为集合并更新
    # 提取 'node(s) end' 字段中的节点集合并加入 node_used
    for node_set in traffic_data_filtered['node(s) end'].dropna():
        node_used.update(eval(node_set)) 
    # 2. 删除 nodes_inside_polygon 中不在 node_used 中的节点
    nodes_clear = nodes_inside_polygon[nodes_inside_polygon['osmid'].isin(node_used)]
    # 3. 删除 edges_inside_polygon 中不在 node_used 中的边
    edges_clear = edges_inside_polygon[
        edges_inside_polygon['u'].isin(node_used) & edges_inside_polygon['v'].isin(node_used)
    ]

    # 保存清理后的数据
    
    traffic_clear.to_csv(f'../{DATA_PATH}/traffic_clear.csv', index=False)
    nodes_clear.to_csv(f'../{DATA_PATH}/nodes_clear.csv', index=False)
    edges_clear.to_csv(f'../{DATA_PATH}/edges_clear.csv', index=False)
    
    return nodes_clear, edges_clear, traffic_clear

def load_data():
    nodes = pd.read_csv(f'../{DATA_PATH}/nodes_clear.csv')
    edges = pd.read_csv(f'../{DATA_PATH}/edges_clear.csv')
    traffic_data = pd.read_csv(f'../{DATA_PATH}/traffic_clear.csv')

    return nodes, edges, traffic_data

# 构建网络
def build_network():
    nodes, edges, traffic_data = load_data()
    print(len(nodes), len(edges), len(traffic_data))
    # 初始化有向图
    G = nx.Graph()
    
    # 添加节点
    for _, row in nodes.iterrows():
        G.add_node(row['osmid'], pos=(row['x'], row['y']))
        # 添加边
    for _, row in edges.iterrows():
        traffic_row = traffic_data[traffic_data['node start'] == row['u']]  # 匹配交通流量
        G.add_edge(row['u'], row['v'], length=row['length'], maxspeed=row['maxspeed'])
    return G

# 可视化网络

def plot_network(G):
    # 提取节点位置
    pos = nx.get_node_attributes(G, 'pos')

    # 绘制图
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_size=5, with_labels=False, edge_color='blue', alpha=0.7)
    plt.title("Baltimore Road Network")
    plt.show()
    
# G = build_network()
# plot_network(G)