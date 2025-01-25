import folium
from config import *
from mdl import load_data

# 加载数据
nodes, _, _ = load_data()

# 定义地图中心位置（例如，巴尔的摩的坐标）
map_center = [39.2904, -76.6122]  # (纬度, 经度)

# 创建一个地图对象
mymap = folium.Map(location=map_center, zoom_start=13)

# 遍历所有的节点并标记
for idx, row in nodes.iterrows():
    node_location = [row['y'], row['x']]  # 假设y是纬度，x是经度
    folium.Marker(
        location=node_location, 
        popup=f"Node OSMID: {row['osmid']}",  # 可以在弹窗中显示更多信息
        icon=folium.Icon(color="blue")  # 设置图标的颜色
    ).add_to(mymap)

# 保存地图到 HTML 文件
mymap.save("map_with_nodes.html")
