import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from config import DATA_PATH
import networkx as nx
import matplotlib.pyplot as plt
# 加载数据
def load_data():
    # 加载 noeds_all.csv
    # 该文件包含了所有的节点的数据
    nodes = pd.read_csv(f'../{DATA_PATH}/nodes_all.csv')

    # 加载 edges_all.csv
    # 该文件包含了所有的边的数据
    edges = pd.read_csv(f'../{DATA_PATH}/edges_all.csv')

    # 加载 MDOT_SHA_Annual_Averyage_Daily_Traffic_Baltimore.csv
    # 该文件包含了巴尔的摩地区的交通流量数据
    traffic_data = pd.read_csv(f'../{DATA_PATH}/MDOT_SHA_Annual_Average_Daily_Traffic_Baltimore.csv')

    # 加载 Edge_Names_With_Nodes.csv
    edge_names = pd.read_csv(f'../{DATA_PATH}/Edge_Names_With_Nodes.csv')

    # 数据清理
    
    # 将 nodes 中的 geomerty 字符串转换为 Point 对象
    nodes['geometry'] = nodes['geometry'].apply(loads)

    # 将 edges 中的 geomerty 字符串转换为 LineString 对象
    edges['geometry'] = edges['geometry'].apply(loads)

    return nodes, edges, traffic_data, edge_names

# 构建网络
def build_network():
    nodes, edges, traffic_data, edge_names = load_data()
    
    # 初始化有向图
    G = nx.Graph()
    
    # 添加节点
    for _, row in nodes.iterrows():
        G.add_node(row['osmid'], pos=(row['x'], row['y']), highway=row['highway'], junction=row['junction'])
        # 添加边
    for _, row in edges.iterrows():
        traffic_row = traffic_data[traffic_data['node start'] == row['u']]  # 匹配交通流量
        aadt = traffic_row['AADT (Current)'].values[0] if not traffic_row.empty else 0  # 默认流量为 0
        G.add_edge(row['u'], row['v'], length=row['length'], maxspeed=row['maxspeed'], aadt=aadt, geometry=row['geometry'])
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
    
G = build_network()
plot_network(G)