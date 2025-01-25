import folium
from config import *
from mdl import load_data

# 加载数据
nodes, _, _ = load_data()

# 定义地图中心位置（例如，巴尔的摩的坐标）
map_center = [39.2904, -76.6122]  # (纬度, 经度)

# 创建一个地图对象
mymap = folium.Map(location=map_center, zoom_start=13)

# 在地图上标记 PCOORDS 中的所有点
for coord in PCOORDS:
    folium.Marker(location=coord, popup=f"Lat: {coord[0]}, Lon: {coord[1]}").add_to(mymap)

# 保存地图到 HTML 文件
mymap.save("map_with_points.html")