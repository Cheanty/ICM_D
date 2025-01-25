# 导入库
from math import sqrt
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from config import *
import folium
from folium.plugins import HeatMap
import ast
from folium import LinearColormap

import math

def haversine_distance(P1, P2):
    """
    计算两个经纬度点之间的球面距离（单位：公里）
    :param P1: 第一个点的经纬度 [纬度, 经度]
    :param P2: 第二个点的经纬度 [纬度, 经度]
    :return: 两点之间的距离（单位：公里）
    """
    # 地球半径（单位：公里）
    R = 6371.0
    
    # 将经纬度从度转换为弧度
    lat1, lon1 = math.radians(P1[0]), math.radians(P1[1])
    lat2, lon2 = math.radians(P2[0]), math.radians(P2[1])
    
    # 纬度和经度的差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine 公式
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # 计算距离
    distance = R * c
    return distance

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
    nodes = pd.read_csv(f'../{DATA_PATH}/nodes_clear.csv')
    edges = pd.read_csv(f'../{DATA_PATH}/edges_clear.csv')
    # 初始化有向图
    G = nx.Graph()
    
    # 添加节点
    for _, row in nodes.iterrows():
        G.add_node(row['osmid'], pos=(row['x'], row['y']))
        # 添加边
    for _, row in edges.iterrows():
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

def change(p1,p2,flow,fm):
    node_mid = [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]
    P_mid = [(P1[0]+P2[0])/2,(P1[1]+P2[1])/2]
    mx = haversine_distance(P4,P_mid)
    ml = haversine_distance(P3,P_mid)*0.8
    l = haversine_distance(node_mid,P_mid)
    if l < ml:
        return flow + fm*0.4 - fm*0.2*l/ml
    elif l > ml and l < mx:
        return flow - fm*0.4*l/mx
    else:
        return flow
    
def changes(p1,p2,reds):
    node_mid = [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]
    P_mid = [(P1[0]+P2[0])/2,(P1[1]+P2[1])/2]
    mx = haversine_distance(P4,P_mid)
    ml = haversine_distance(P3,P_mid)*0.8
    l = haversine_distance(node_mid,P_mid)
    if l < ml:
        return sqrt(reds)
    elif l > ml and l < mx:
        return reds
    else:
        return reds
    
def hot_map():
    # 加载数据
    edges = pd.read_csv(f'../{DATA_PATH}/traffic_clear.csv')
    nodes = pd.read_csv(f'../{DATA_PATH}/nodes_all.csv', usecols=NODES_LIST)
    
    # 计算流量的最大值和最小值，用于归一化流量
    flow_max = edges['AADTA'].max()  # 流量最大值
    flow_min = edges['AADTA'].min()  # 流量最小值
    print(flow_max, flow_min)
    # 创建地图对象，以巴尔的摩为中心
    map_center = CENTER  # (纬度, 经度)
    mymap = folium.Map(location=map_center, zoom_start=13)

    # 创建颜色渐变（从蓝色到红色）
    colormap = LinearColormap(['blue', 'green', 'yellow', 'red'], vmin=flow_min, vmax=flow_max)

    # 存储热力图数据
    heat_data = []
    # 遍历所有的边，获取节点信息并计算流量
    for _, row in edges.iterrows():
        try:
            # 安全地解析 node start 和 node(s) end 字段为集合
            start_node_ids = ast.literal_eval(row['node start'])  # 起始节点
            end_node_ids = ast.literal_eval(row['node(s) end'])  # 终止节点
        except (ValueError, SyntaxError):
            print(f"Error parsing node start or node(s) end for edge {row['osmid']}")
            continue
        
        # 遍历所有的起点和终点节点
        for start_id in start_node_ids:
            start_node_df = nodes[nodes['osmid'] == start_id]
            if not start_node_df.empty:
                start_node = start_node_df.iloc[0]
                start_coords = [start_node['y'], start_node['x']]  # 获取纬度和经度
            else:
                print(f"Warning: Start node {start_id} not found in nodes")
                continue

            for end_id in end_node_ids:
                end_node_df = nodes[nodes['osmid'] == end_id]
                if not end_node_df.empty:
                    end_node = end_node_df.iloc[0]
                    end_coords = [end_node['y'], end_node['x']]  # 获取纬度和经度
                else:
                    print(f"Warning: End node {end_id} not found in nodes")
                    continue
                # 将起点和终点加入热力图数据
                heat_data.append(start_coords)  # 添加起点坐标
                heat_data.append(end_coords)  # 添加终点坐标

    # 添加热力图层
    HeatMap(heat_data).add_to(mymap)

    # 保存为 HTML 文件
    mymap.save("traffic_flow_map2.html")
    print("交通流量路线图已保存为 'traffic_flow_map.html'")

# 将 RGB 值转换为 HEX 格式
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

def route_map():
    # 加载数据
    edges = pd.read_csv(f'../{DATA_PATH}/traffic_clear.csv')
    nodes = pd.read_csv(f'../{DATA_PATH}/nodes_all.csv', usecols=NODES_LIST)
    
    # 计算流量的最大值和最小值，用于归一化流量
    flow_max = edges['AADTA'].max()  # 流量最大值
    flow_min = edges['AADTA'].min()  # 流量最小值
    # 创建地图对象，以巴尔的摩为中心
    map_center = CENTER # (纬度, 经度)
    mymap = folium.Map(location=map_center, zoom_start=13)

    # 遍历所有的边，获取节点信息并计算流量
    for _, row in edges.iterrows():
        try:
            # 安全地解析 node start 和 node(s) end 字段为集合
            start_node_ids = ast.literal_eval(row['node start'])  # 起始节点
            end_node_ids = ast.literal_eval(row['node(s) end'])  # 终止节点
        except (ValueError, SyntaxError):
            print(f"Error parsing node start or node(s) end for edge {row['osmid']}")
            continue
        
        # 遍历所有的起点和终点节点
        for start_id in start_node_ids:
            start_node_df = nodes[nodes['osmid'] == start_id]
            if not start_node_df.empty:
                start_node = start_node_df.iloc[0]
                start_coords = [start_node['y'], start_node['x']]  # 获取纬度和经度
            else:
                print(f"Warning: Start node {start_id} not found in nodes")
                continue

            for end_id in end_node_ids:
                end_node_df = nodes[nodes['osmid'] == end_id]
                if not end_node_df.empty:
                    end_node = end_node_df.iloc[0]
                    end_coords = [end_node['y'], end_node['x']]  # 获取纬度和经度
                else:
                    print(f"Warning: End node {end_id} not found in nodes")
                    continue

                flow = row['AADTA'] if 'AADTA' in row else 0  # 如果没有流量数据，默认为 0
                flow = change(start_coords,end_coords,flow,flow_max)
                flow_max = max(flow_max,flow)
                flow_min = min(flow_min,flow)
                # 归一化流量值，得到颜色和宽度
                normalized_flow = (flow - flow_min) / (flow_max - flow_min)  # 归一化流量到 [0, 1] 范围
                reds = changes(start_coords,end_coords,normalized_flow)
                
                line_width = 1 + 4 * normalized_flow  # 根据归一化流量来设置线宽，最大线宽为 5
                
                # 根据流量设置颜色，流量越大越红，流量越小越绿
                
                line_color = [
                      
                    reds,  # 红色部分
                    (1-reds),# 绿色部分
                    0  # 蓝色部分
                ]
                
                # 将 RGB 转换为 HEX
                hex_color = rgb_to_hex(line_color)

                # 绘制道路的线条
                folium.PolyLine([start_coords, end_coords], color=hex_color, weight=line_width, opacity=1).add_to(mymap)
    width_brage = 5
    color_brage = [1,0,0]
    hex_brage = rgb_to_hex(color_brage)
    folium.PolyLine([P1, P2], color=hex_brage, weight=width_brage, opacity=1).add_to(mymap)
    mymap.save("traffic_flow_map2.html")
    print("道路流量图已保存为 'traffic_flow_map2.html'")
    # 保存为 HTML 文件
    # mymap.save("traffic_flow_map.html")
    # print("道路流量图已保存为 'traffic_flow_map.html'")

if __name__ == "__main__":
    route_map()