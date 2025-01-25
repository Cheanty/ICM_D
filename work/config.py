# 数据文件路径
DATA_PATH = '2025_Problem_D_Data'
# 巴尔的摩市包围多边形的顶点坐标
PCOORDS = [
    [39.38054154645972, -76.75751332410483],  # 注意顺序为 [x, y]
    [39.42776160126864, -76.64593342697009],
    [39.41078728523181, -76.55666950932739],
    [39.37762240467791, -76.50826100014423],
    [39.33089953041552, -76.47564533896755],
    [39.30008782715071, -76.45710591011272],
    [39.24932520325849, -76.43959644925673],
    [39.20762676871198, -76.46964315992778],
    [39.178834487323954, -76.61469105598914],
    [39.23696573959844, -76.70075624770088],
    [39.30061917760752, -76.75682667717022],
    [39.33647606027192, -76.7660963905619]# 闭合多边形
]
# 点坐标
P1 = [39.21286677595327, -76.53365499270008] # 大桥左端点坐标
P2 = [39.22289474240978, -76.52074552097815] # 大桥右端点坐标
P3 = [39.25206942419113, -76.57005067500856] # 倒塌大桥的上方大桥坐标
P4 = [39.33795138713097, -76.64177270247241] # 市中心湾区坐标
# 节点数据的列名
NODES_LIST = ['osmid', 'x', 'y', 'highway']
# 边数据的列名
EDGES_LIST = ['u', 'v', 'osmid','highway','lanes','maxspeed','oneway','length','geometry']
# 交通数据的列名
TRAFFIC_DATA_LIST = [
    'node start', 'node(s) end',
    'County Code','Rural / Urban',
    'Functional Class Code',
    'K-Factor','D-Factor',
    'AADT 2014','AAWDT 2014',
    'AADT 2015','AAWDT 2015',
    'AADT 2016','AAWDT 2016',
    'AADT 2017','AAWDT 2017',
    'AADT 2018','AAWDT 2018',
    'AADT 2019','AAWDT 2019',
    'AADT 2020','AAWDT 2020',
    'AADT 2021','AAWDT 2021',
    'AADT 2022','AAWDT 2022',
    'AADT (Current)','AAWDT (Current)',
    ]
# 交通流量历年列表
AADTS_COLUMNS = [
    'AADT 2014', 'AADT 2015', 'AADT 2016', 'AADT 2017', 'AADT 2018', 'AADT 2019', 'AADT 2020', 'AADT 2021', 'AADT 2022', 'AADT (Current)'
]
# 交通流量工作日历年列表
AAWDTS_COLUMNS = [
    'AAWDT 2014', 'AAWDT 2015', 'AAWDT 2016', 'AAWDT 2017', 'AAWDT 2018', 'AAWDT 2019', 'AAWDT 2020', 'AAWDT 2021', 'AAWDT 2022', 'AAWDT (Current)'
]

CENTER = [39.2904, -76.6122] # 巴尔的摩的坐标

# highway 类型

NODE_HIGHWAY_DICT = {
    'crossing': -6,  # 人行横道，减慢交通流量
    'elevator': -3,  # 电梯入口，主要影响步行交通
    'give_way': -5,  # 让行标志，车辆需减速
    'mini_roundabout': -4,  # 小型环岛，车辆减速通过
    'motorway_junction': 5,  # 高速公路交汇点，提升通行效率
    'speed_camera': -4,  # 速度监控摄像头，减速通行
    'stop': -7,  # 停止标志，显著降低通行效率
    'traffic_signals': -5,  # 红绿灯控制点，影响通行效率
    'turning_circle': -3,  # 转弯环岛，减慢通行速度
    'turning_loop': -4  # 调头环路，降低通行效率
}
# highway 因子

NODE_HIGHWAY_K = 0.01

EDGE_HIGHWAY_DICT = {
    'busway': 20,  # 公交车道，较低的通行效率
    'disused': 1,   # 废弃道路，无法通行
    'living_street': 30,  # 生活街区，通行效率较低
    'motorway': 100,  # 高速公路，通行效率极高
    'primary': 85,  # 主干道，通行效率较高
    'primary_link': 80,  # 主干道连接路，通行效率高
    'residential': 40,  # 住宅区道路，通行效率一般
    'secondary': 60,  # 次要道路，通行效率一般
    'secondary_link': 50,  # 次要道路连接路，通行效率较低
    'tertiary': 70,  # 第三级道路，适中通行效率
    'tertiary_link': 45,  # 第三级道路连接路，通行效率较低
    'trunk': 90,  # 干道，通行效率接近高速
    'unclassified': 35  # 未分类道路，通行效率较低
}

# highway 因子
MAX_HIGHWAY = 100
EDGE_HIGHWAY_K = 100


# 通行系数

TRAFFIC_K = 1