
## 模型构建

### 数据分析

题目给出了所有节点的各种数据，以及所有边的各种数据，同时给出了每条道路的一年内的日平均流量

1. 构建AADT流量图
2. 构建AAWDT流量图
3. 构建路线图
4. 构建模型

# 节点和边的权重定义

## 节点类型（NODE_HIGHWAY_DICT）
- `crossing`: -6
- `elevator`: -3
- `give_way`: -5
- `mini_roundabout`: -4
- `motorway_junction`: 5
- `speed_camera`: -4
- `stop`: -7
- `traffic_signals`: -5
- `turning_circle`: -3
- `turning_loop`: -4

## 道路类型（EDGE_HIGHWAY_DICT）
- `busway`: -2
- `disused`: -10
- `living_street`: -1
- `motorway`: 10
- `primary`: 8
- `primary_link`: 6
- `residential`: -2
- `secondary`: 5
- `secondary_link`: 3
- `tertiary`: 2
- `tertiary_link`: 1
- `trunk`: 9
- `unclassified`: -1
