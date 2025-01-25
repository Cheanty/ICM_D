DATA_PATH = '2025_Problem_D_Data'
# 定义多边形的顶点坐标
PCOORDS = [
    [-76.71133808741571, 39.37150725022421],  # 注意顺序为 [x, y]
    [-76.53032389319677, 39.37173042805728],
    [-76.52949874495596, 39.20994236132783],
    [-76.55002295212472, 39.19698836504025],
    [-76.58348246357684, 39.20758178486897],
    [-76.61144960092174, 39.23413862079039],
    [-76.71122636603121, 39.27812585458132],
    [-76.71133808741571, 39.37150725022421]  # 闭合多边形
]
P1 = [39.21286677595327, -76.53365499270008]
P2 = [39.22289474240978, -76.52074552097815]
NODES_LIST = ['osmid', 'x', 'y', 'highway']
EDGES_LIST = ['u', 'v', 'osmid','highway','lanes','maxspeed','oneway','length','geometry']
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

AADTS_COLUMNS = [
    'AADT 2014', 'AADT 2015', 'AADT 2016', 'AADT 2017', 'AADT 2018', 'AADT 2019', 'AADT 2020', 'AADT 2021', 'AADT 2022', 'AADT (Current)'
]
AAWDTS_COLUMNS = [
    'AAWDT 2014', 'AAWDT 2015', 'AAWDT 2016', 'AAWDT 2017', 'AAWDT 2018', 'AAWDT 2019', 'AAWDT 2020', 'AAWDT 2021', 'AAWDT 2022', 'AAWDT (Current)'
]