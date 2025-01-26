import pandas as pd
from config import *
from folium.plugins import HeatMap
import folium
from math import radians, sin, cos, sqrt, atan2

# Haversine 距离计算函数
def haversine(lat1, lon1, lat2, lon2):
    # 转换为弧度
    R = 6371  # 地球半径，单位 km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # 计算 Haversine 距离
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c  # 返回距离，单位为 km
    return distance

bus_stop = pd.read_csv(f'../{DATA_PATH}/Bus_Stops.csv',usecols=BUS_STOPS_LIST)
# 获取每个 BUS_POS 距离最近的 bus_stop
def get_nearest_bus_stop():
    nearest_stops = []
    
    for pos in BUS_POS:
        lat_pos, lon_pos = pos
        min_distance = float('inf')  # 初始化最小距离为无限大
        nearest_stop = None
        
        for _, row in bus_stop.iterrows():
            lat_stop = row['Y']  # 假设 Y 列为站点纬度
            lon_stop = row['X']  # 假设 X 列为站点经度
            
            # 计算当前站点和 BUS_POS 点之间的距离
            distance = haversine(lat_pos, lon_pos, lat_stop, lon_stop)
            
            # 如果当前站点距离更近，则更新最近站点
            if distance < min_distance:
                min_distance = distance
                nearest_stop = row['stop_id']  # 站点 ID 或者其他标识符
        
        # 将最近的站点和距离加入结果列表
        nearest_stops.append(nearest_stop)
    filtered_bus_stop = bus_stop[bus_stop['stop_id'].isin(nearest_stops)]
    return filtered_bus_stop

def filtered_bus_stop_map():
    fus = get_nearest_bus_stop()
    m = folium.Map(CENTER, zoom_start=13)
    for index, row in fus.iterrows():
        folium.Marker([row['Y'], row['X']], popup=row['stop_id']).add_to(m)
    m.save(f'../{DATA_PATH}/mydata/filtered_bus_stop_map.html')

def bus_pos():
    m = folium.Map(CENTER, zoom_start=13)
    # 遍历 BUS_POS 中的所有坐标点
    for pos in BUS_POS:
        folium.Marker(pos,popup=pos).add_to(m)
    # 保存为 HTML 文件
    m.save(f'../{DATA_PATH}/mydata/bus_stop_posmap.html')
def bus_hotmap(bus_stop):
    m = folium.Map(CENTER, zoom_start=13)
    # 创建一个列表来存储热力图的数据点
    heat_data = []
    for index, row in bus_stop.iterrows():
        lat = row['Y']
        lon = row['X']
        # 添加到热力图数据
        heat_data.append([lat, lon])
        # folium.Marker([lat, lon], popup=row['stop_id']).add_to(m)
    # 创建热力图
    HeatMap(heat_data).add_to(m)
    m.save(f'../{DATA_PATH}/mydata/Bus_stop_hot.html')


    
bus_pos()