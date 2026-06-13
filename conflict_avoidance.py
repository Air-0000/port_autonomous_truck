"""
港区运输冲突检测与路径重规划模块
第九赛道 - 港口自动驾驶集卡调度问题

主要功能：
1. 顶牛（对向相遇）检测与预防
2. 人机冲突检测与预防
3. 路径重规划
4. 速度调整策略

评分影响：
- 顶牛：次数 × 10分/次（最高扣分因素）
- 人机冲突：次数 × 2分/次

坐标说明：
- y轴向下为正向，x轴向右为正向
- angle：y轴向上为正向，x轴向右为正向，角度顺时针增加
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# ============================================================================
# 可调整参数配置
# ============================================================================

# ------------------- 安全距离参数 -------------------
SAFE_DISTANCE_DEADLOCK = 15.0        # 顶牛检测安全距离（米）, 建议范围: 10-25
SAFE_DISTANCE_HUMAN = 20.0           # 人机冲突安全距离（米）, 建议范围: 15-30
SAFE_DISTANCE_CAR_FOLLOWING = 25.0   # 跟车安全距离（米）, 建议范围: 20-40
MIN_GAP_FOR_PASSING = 3.5            # 错车最小间隙（米）, 建议范围: 3-5

# ------------------- 速度参数 -------------------
MAX_TRUCK_SPEED = 15.0               # 最大行驶速度（m/s）
SPEED_REDUCTION_RATIO = 0.6          # 冲突时速度折减系数
HUMAN_SPEED_PREDICTION_ERR = 2.0     # 人工驾驶车辆速度预测误差（m/s）

# ------------------- 时间参数 -------------------
REACTION_TIME = 1.0                  # 驾驶员反应时间（秒）
LOOKAHEAD_TIME = 5.0                 # 预测前瞻时间（秒）
BRAKE_DECELERATION = 3.0              # 刹车减速度（m/s²）

# ------------------- 道路参数 -------------------
NARROW_ROAD_WIDTH = 6.0              # 狭窄路段宽度阈值（米）, 建议范围: 5-8
LANE_WIDTH = 3.5                     # 标准车道宽度（米）

# ============================================================================
# 数据结构定义
# ============================================================================

class VehicleState:
    """车辆状态数据结构"""
    def __init__(self, vehicle_id: str, vehicle_type: str, x: float, y: float,
                 road_id: str, lane_id: str, speed: float, angle: float,
                 distance_on_road: float):
        self.id = vehicle_id
        self.type = vehicle_type  # 'company', 'other', 'human'
        self.x = x
        self.y = y
        self.road_id = road_id
        self.lane_id = lane_id
        self.speed = speed
        self.angle = angle  # 弧度，y轴向上为正向，逆时针增加
        self.distance_on_road = distance_on_road

    @staticmethod
    def from_csv_row(row: dict) -> 'VehicleState':
        """从CSV行创建车辆状态"""
        return VehicleState(
            vehicle_id=str(row['vehicle_id']),
            vehicle_type=row['vehicle_type'],
            x=float(row['x']),
            y=float(row['y']),
            road_id=str(row['road_id']),
            lane_id=str(row['lane_id']),
            speed=float(row['speed']),
            angle=float(row['angle']),
            distance_on_road=float(row['distance_on_road'])
        )


class RoadNetwork:
    """路网数据结构"""
    def __init__(self):
        self.links = {}           # link_id -> Link对象
        self.nodes = {}           # node_id -> Node对象
        self.adjacency = defaultdict(list)  # 邻接表
        self.link_geometry = {}   # link_id -> 几何信息

    def add_link(self, link_id: str, start_node: str, end_node: str,
                 lanes: int, width: float, is_bidirectional: bool = True):
        """添加路段"""
        self.links[link_id] = {
            'id': link_id,
            'start_node': start_node,
            'end_node': end_node,
            'lanes': lanes,
            'width': width,
            'is_bidirectional': is_bidirectional,
            'length': 0.0  # 待计算
        }
        self.adjacency[start_node].append(link_id)
        if is_bidirectional:
            self.adjacency[end_node].append(link_id)

    def get_road_width(self, link_id: str) -> float:
        """获取道路宽度"""
        if link_id in self.links:
            return self.links[link_id]['width']
        return LANE_WIDTH * self.links[link_id]['lanes']

    def is_narrow_road(self, link_id: str) -> bool:
        """判断是否为狭窄路段（无法错车）"""
        width = self.get_road_width(link_id)
        return width < NARROW_ROAD_WIDTH

    def get_opposite_lane(self, link_id: str, lane_id: str) -> Optional[str]:
        """获取对向车道（用于顶牛检测）"""
        if link_id not in self.links:
            return None
        link = self.links[link_id]
        # 假设车道ID格式为 "link_id_lane_num"
        parts = lane_id.split('_')
        if len(parts) >= 3:
            lane_num = int(parts[-1])
            # 对向车道的车道号
            opposite_lane_num = link['lanes'] - lane_num + 1
            return f"{parts[0]}_{parts[1]}_{opposite_lane_num}"
        return None


# ============================================================================
# 辅助函数
# ============================================================================

def normalize_angle(angle: float) -> float:
    """将角度标准化到 [0, 2π) 范围"""
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle


def angle_to_vector(angle: float) -> Tuple[float, float]:
    """角度转换为方向向量（基于angle定义：y轴向上为正向）"""
    # angle: y轴向上为正向，逆时针增加
    # 标准数学坐标系：x轴向右，y轴向上，逆时针增加
    # 这里需要转换：y轴向下为正向
    dx = np.sin(angle)
    dy = -np.cos(angle)  # y轴向下为正，所以取负
    return dx, dy


def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """计算两点之间的欧氏距离"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def is_moving_toward(point1: Tuple[float, float], angle1: float,
                     point2: Tuple[float, float]) -> bool:
    """
    判断点1是否朝点2方向移动
    angle: y轴向上为正向，逆时针增加
    """
    dx, dy = angle_to_vector(angle1)
    # 从point1到point2的向量
    to_target = (point2[0] - point1[0], point2[1] - point1[1])
    distance = np.sqrt(to_target[0]**2 + to_target[1]**2)
    if distance < 0.1:
        return False
    # 归一化
    to_target = (to_target[0]/distance, to_target[1]/distance)
    # 点积判断是否同向
    dot = dx * to_target[0] + dy * to_target[1]
    return dot > 0.3  # 夹角小于约73度


def calculate_relative_speed(speed1: float, speed2: float,
                             angle1: float, angle2: float) -> float:
    """计算相对速度（沿连线方向）"""
    dx1, dy1 = angle_to_vector(angle1)
    dx2, dy2 = angle_to_vector(angle2)
    # 相对速度向量
    rel_vx = speed1 * dx1 - speed2 * dx2
    rel_vy = speed1 * dy1 - speed2 * dy2
    return np.sqrt(rel_vx**2 + rel_vy**2)


def predict_position(pos: Tuple[float, float], angle: float,
                    speed: float, time: float) -> Tuple[float, float]:
    """预测车辆在time秒后的位置"""
    dx, dy = angle_to_vector(angle)
    return (pos[0] + dx * speed * time, pos[1] + dy * speed * time)


def get_lateral_offset(lane_id: str, vehicle_x: float, road_start_x: float) -> float:
    """获取车辆在车道上的横向偏移"""
    # 简化：假设车道宽度为LANE_WIDTH
    parts = lane_id.split('_')
    if len(parts) >= 3:
        lane_num = int(parts[-1])
        lane_center_x = road_start_x + (lane_num - 0.5) * LANE_WIDTH
        return abs(vehicle_x - lane_center_x)
    return 0.0


# ============================================================================
# 核心检测与规划函数
# ============================================================================

def detect_deadlock(vehicle_states: List[VehicleState],
                    road_network: RoadNetwork) -> Dict[str, Dict[str, Any]]:
    """
    顶牛检测函数

    检测本公司集卡是否与对向车辆在狭窄路段相遇，形成顶牛

    输入：
        vehicle_states: 所有车辆状态列表
        road_network: 路网结构

    输出：
        {车辆id: {"type": "deadlock", "other_vehicle": id, "distance": m, "severity": high/medium/low}}
    """
    conflicts = {}

    # 分离本公司集卡和其他车辆
    company_trucks = [v for v in vehicle_states if v.type == 'company']
    other_trucks = [v for v in vehicle_states if v.type != 'company']

    for truck in company_trucks:
        truck_pos = (truck.x, truck.y)
        truck_link = truck.road_id

        # 检查是否为狭窄路段
        if not road_network.is_narrow_road(truck_link):
            continue

        # 获取对向车道ID
        opposite_lane = road_network.get_opposite_lane(truck_link, truck.lane_id)
        if not opposite_lane:
            continue

        # 在对向车道上寻找车辆
        for other in other_trucks:
            if other.road_id != truck_link:
                continue
            if other.lane_id != opposite_lane:
                continue

            other_pos = (other.x, other.y)
            distance = calculate_distance(truck_pos, other_pos)

            # 检查是否在安全距离内
            if distance > SAFE_DISTANCE_DEADLOCK:
                continue

            # 判断是否相对而行（顶牛核心判断）
            # 两车是否互相朝对方移动
            truck_to_other = is_moving_toward(truck_pos, truck.angle, other_pos)
            other_to_truck = is_moving_toward(other_pos, other.angle, truck_pos)

            # 判断严重程度
            severity = "low"
            if truck_to_other and other_to_truck:
                # 完全对向 - 最高风险
                severity = "high"
            elif truck_to_other or other_to_truck:
                severity = "medium"

            # 额外检查：是否都无法避让（速度较低）
            if truck.speed < 3.0 and other.speed < 3.0:
                severity = "high"

            if truck_to_other or other_to_truck:
                conflicts[truck.id] = {
                    "type": "deadlock",
                    "other_vehicle": other.id,
                    "distance": distance,
                    "severity": severity,
                    "truck_pos": truck_pos,
                    "other_pos": other_pos
                }
                break  # 每辆车只报一个最严重的冲突

    return conflicts


def detect_human_conflict(vehicle_states: List[VehicleState],
                          road_network: RoadNetwork) -> Dict[str, Dict[str, Any]]:
    """
    人机冲突检测函数

    检测本公司集卡与人工驾驶集卡在同一车道或近距离并行的情况

    输入：
        vehicle_states: 所有车辆状态列表
        road_network: 路网结构

    输出：
        {车辆id: {"type": "human_conflict", "other_vehicle": id, "distance": m}}
    """
    conflicts = {}

    # 分离本公司集卡和人工驾驶集卡
    company_trucks = [v for v in vehicle_states if v.type == 'company']
    human_trucks = [v for v in vehicle_states if v.type == 'human']

    for truck in company_trucks:
        truck_pos = (truck.x, truck.y)

        for human in human_trucks:
            # 必须在同一道路
            if human.road_id != truck.road_id:
                continue

            human_pos = (human.x, human.y)
            distance = calculate_distance(truck_pos, human_pos)

            # 人机冲突使用更大的安全距离（人工驾驶难以预测）
            if distance > SAFE_DISTANCE_HUMAN:
                continue

            # 检查是否在同一车道或相邻车道
            same_lane = (human.lane_id == truck.lane_id)
            adjacent_lane = abs(int(human.lane_id.split('_')[-1]) -
                               int(truck.lane_id.split('_')[-1])) == 1

            if same_lane or adjacent_lane:
                # 判断是否同向行驶
                same_direction = is_moving_toward(truck_pos, truck.angle, human_pos)

                # 计算 TTC (Time To Collision)
                relative_speed = calculate_relative_speed(
                    truck.speed, human.speed, truck.angle, human.angle
                )

                if relative_speed > 0:
                    ttc = distance / relative_speed
                else:
                    ttc = float('inf')

                conflicts[truck.id] = {
                    "type": "human_conflict",
                    "other_vehicle": human.id,
                    "distance": distance,
                    "same_lane": same_lane,
                    "ttc": ttc,
                    "same_direction": same_direction
                }
                break

    return conflicts


def check_potential_deadlock(vehicle_id: str,
                             planned_route: List[str],
                             other_vehicles: List[VehicleState],
                             road_network: RoadNetwork,
                             lookahead_time: float = LOOKAHEAD_TIME) -> bool:
    """
    死锁预防函数

    预测在计划路径上是否会发生顶牛，提前改变策略

    输入：
        vehicle_id: 车辆ID
        planned_route: 计划路径 [link_id1, link_id2, ...]
        other_vehicles: 其他车辆列表
        road_network: 路网结构
        lookahead_time: 前瞻时间（秒）

    输出：
        True 表示预测到顶牛风险
    """
    # 找到本公司车辆
    vehicle = None
    for v in other_vehicles:
        if v.id == vehicle_id:
            vehicle = v
            break

    if vehicle is None:
        return False

    vehicle_pos = (vehicle.x, vehicle.y)

    for link_id in planned_route:
        # 只检查狭窄路段
        if not road_network.is_narrow_road(link_id):
            continue

        # 预测时间内到达该路段的车辆
        for other in other_vehicles:
            if other.type == 'company':
                continue  # 只检查非本公司车辆

            other_pos = (other.x, other.y)

            # 检查对向车道
            opposite_lane = road_network.get_opposite_lane(link_id, vehicle.lane_id)
            if opposite_lane and other.lane_id == opposite_lane:
                # 检查其他车辆是否会进入该路段
                other_predicted_pos = predict_position(
                    other_pos, other.angle, other.speed, lookahead_time
                )

                # 简单判断：其他车辆在前方且朝该路段移动
                distance_to_link = calculate_distance(vehicle_pos, other_predicted_pos)
                toward_link = is_moving_toward(other_pos, other.angle, vehicle_pos)

                if toward_link and distance_to_link < SAFE_DISTANCE_DEADLOCK * 2:
                    return True

    return False


def replan_route(vehicle_id: str,
                 current_pos: Tuple[float, float],
                 destination: str,
                 road_network: RoadNetwork,
                 shortest_path_matrix: Dict[Tuple[str, str], List[str]],
                 avoid_links: List[str]) -> Optional[List[str]]:
    """
    路径重规划函数

    当原路径有冲突风险时，计算新的无冲突最短路径

    输入：
        vehicle_id: 车辆ID
        current_pos: 当前位置 (x, y)
        destination: 目的地节点ID
        road_network: 路网结构
        shortest_path_matrix: 最短路径矩阵 {(起点, 终点): [link_id1, link_id2, ...]}
        avoid_links: 需要避开的路段列表

    输出：
        [link_id1, link_id2, ...] 绕行路径，或 None（无法规划）
    """
    # 找到当前位置最近的节点
    current_node = find_nearest_node(current_pos, road_network)

    if (current_node, destination) not in shortest_path_matrix:
        return None

    original_path = shortest_path_matrix[(current_node, destination)]

    # 过滤掉需要避开的路段
    if not avoid_links:
        return original_path

    # 尝试逐步绕行
    for link_to_avoid in avoid_links:
        if link_to_avoid not in original_path:
            continue

        # 找到需要绕行的断点
        avoid_index = original_path.index(link_to_avoid)

        # 尝试从断点前一路段绕行到断点后一路段
        if avoid_index > 0 and avoid_index < len(original_path) - 1:
            prev_link = original_path[avoid_index - 1]
            next_link = original_path[avoid_index + 1]

            # 找相邻可替代道路
            detour = find_detour(prev_link, next_link, road_network, avoid_links)
            if detour:
                # 重组路径
                new_path = original_path[:avoid_index] + detour + original_path[avoid_index + 1:]
                return new_path

    # 无法绕行
    return None


def find_detour(prev_link: str, next_link: str,
                road_network: RoadNetwork,
                avoid_links: List[str]) -> Optional[List[str]]:
    """
    查找两个路段之间的绕行路径

    通过BFS在邻接图中搜索，避开avoid_links
    """
    if prev_link not in road_network.links or next_link not in road_network.links:
        return None

    start_node = road_network.links[prev_link]['end_node']
    end_node = road_network.links[next_link]['start_node']

    # 如果起点和终点直接相连，不需要绕行
    if start_node == end_node:
        return []

    # BFS 搜索最短路径
    from collections import deque

    queue = deque([(start_node, [])])
    visited = {start_node}

    while queue:
        node, path = queue.popleft()

        if node == end_node:
            return path

        for link_id in road_network.adjacency[node]:
            if link_id in avoid_links:
                continue

            next_node = road_network.links[link_id]['end_node']
            if next_node not in visited:
                visited.add(next_node)
                queue.append((next_node, path + [link_id]))

    return None


def find_nearest_node(pos: Tuple[float, float],
                      road_network: RoadNetwork) -> str:
    """找到距离位置最近的节点"""
    min_distance = float('inf')
    nearest_node = None

    for node_id in road_network.nodes:
        node_pos = road_network.nodes[node_id]
        distance = calculate_distance(pos, node_pos)
        if distance < min_distance:
            min_distance = distance
            nearest_node = node_id

    return nearest_node or list(road_network.nodes.keys())[0]


def adjust_speed_for_conflict(vehicle_id: str,
                               vehicle_states: List[VehicleState],
                               conflict_info: Dict[str, Any]) -> float:
    """
    速度调整函数

    根据冲突信息计算推荐速度

    输入：
        vehicle_id: 车辆ID
        vehicle_states: 所有车辆状态
        conflict_info: 冲突信息

    输出：
        建议的速度值（可为负值表示停车）
    """
    # 找到车辆
    vehicle = None
    for v in vehicle_states:
        if v.id == vehicle_id:
            vehicle = v
            break

    if vehicle is None:
        return 0.0

    conflict_type = conflict_info.get("type", "")
    distance = conflict_info.get("distance", SAFE_DISTANCE_DEADLOCK)

    if conflict_type == "deadlock":
        # 顶牛情况：优先减速停车等待
        severity = conflict_info.get("severity", "medium")

        if severity == "high":
            # 严重顶牛：立即停车
            return -1.0  # 停车信号
        else:
            # 中等顶牛：减速
            return vehicle.speed * SPEED_REDUCTION_RATIO

    elif conflict_type == "human_conflict":
        # 人机冲突：保持更大安全距离
        ttc = conflict_info.get("ttc", float('inf'))
        same_direction = conflict_info.get("same_direction", True)

        if same_direction:
            # 同向行驶：保持安全距离
            safe_distance = SAFE_DISTANCE_HUMAN + HUMAN_SPEED_PREDICTION_ERR * ttc
            if distance < safe_distance:
                # 减速以保持距离
                return min(vehicle.speed * 0.5, vehicle.speed - BRAKE_DECELERATION * REACTION_TIME)
        else:
            # 对向行驶（更危险）
            return vehicle.speed * SPEED_REDUCTION_RATIO

    # 默认：轻微减速
    return vehicle.speed * 0.8


def calculate_safe_following_distance(speed: float) -> float:
    """计算安全跟车距离（基于速度）"""
    # 使用2秒跟车距离 + 速度相关距离
    return max(SAFE_DISTANCE_CAR_FOLLOWING, speed * 2.0 + 10.0)


def check_front_vehicle_conflict(vehicle: VehicleState,
                                  vehicle_states: List[VehicleState]) -> Optional[Dict]:
    """
    检测前方车辆冲突（用于跟车行驶）

    返回冲突信息或None
    """
    vehicle_pos = (vehicle.x, vehicle.y)

    # 寻找同车道前方车辆
    candidates = []
    for other in vehicle_states:
        if other.id == vehicle.id:
            continue
        if other.road_id != vehicle.road_id:
            continue
        if other.lane_id != vehicle.lane_id:
            continue

        other_pos = (other.x, other.y)
        distance = calculate_distance(vehicle_pos, other_pos)

        # 只考虑前方车辆
        if is_moving_toward(vehicle_pos, vehicle.angle, other_pos):
            candidates.append((distance, other))

    if not candidates:
        return None

    # 找最近的前方车辆
    candidates.sort(key=lambda x: x[0])
    closest_distance, closest_vehicle = candidates[0]

    safe_distance = calculate_safe_following_distance(vehicle.speed)

    if closest_distance < safe_distance:
        return {
            "type": "front_vehicle",
            "other_vehicle": closest_vehicle.id,
            "distance": closest_distance,
            "safe_distance": safe_distance
        }

    return None


# ============================================================================
# 综合避险决策主函数
# ============================================================================

def resolve_conflicts(vehicle_states: List[VehicleState],
                      vehicle_plans: Dict[str, Dict],
                      road_network: RoadNetwork,
                      shortest_path_matrix: Dict[Tuple[str, str], List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    综合避险决策函数

    整合所有冲突检测和解决逻辑，返回所有控制指令

    输入：
        vehicle_states: 所有车辆状态列表
        vehicle_plans: 车辆计划 {vehicle_id: {"destination": node_id, "planned_route": [...]}}
        road_network: 路网结构
        shortest_path_matrix: 最短路径矩阵

    输出：
        {vehicle_id: {
            "speed": 调整后速度,
            "route": 调整后路径,
            "lane_change": 是否需要变道,
            "conflict_detected": 冲突类型或None,
            "action": 具体动作描述
        }}
    """
    if shortest_path_matrix is None:
        shortest_path_matrix = {}

    decisions = {}

    # 获取本公司集卡
    company_trucks = [v for v in vehicle_states if v.type == 'company']

    # 1. 检测所有顶牛冲突
    deadlock_conflicts = detect_deadlock(vehicle_states, road_network)

    # 2. 检测所有人机冲突
    human_conflicts = detect_human_conflict(vehicle_states, road_network)

    for truck in company_trucks:
        truck_id = truck.id
        decision = {
            "speed": truck.speed,
            "route": vehicle_plans.get(truck_id, {}).get("planned_route", []),
            "lane_change": False,
            "conflict_detected": None,
            "action": "正常行驶"
        }

        # 检查顶牛冲突
        if truck_id in deadlock_conflicts:
            conflict = deadlock_conflicts[truck_id]
            decision["conflict_detected"] = "deadlock"

            # 检查是否为严重顶牛
            if conflict.get("severity") == "high":
                # 严重顶牛：停车等待
                decision["speed"] = -1.0
                decision["action"] = f"顶牛告警：停车等待对向车辆 {conflict['other_vehicle']}"
            else:
                # 中等顶牛：减速
                decision["speed"] = adjust_speed_for_conflict(truck_id, vehicle_states, conflict)
                decision["action"] = f"顶牛风险：减速避让 {conflict['other_vehicle']}"

            decisions[truck_id] = decision
            continue

        # 检查人机冲突
        if truck_id in human_conflicts:
            conflict = human_conflicts[truck_id]
            decision["conflict_detected"] = "human_conflict"

            new_speed = adjust_speed_for_conflict(truck_id, vehicle_states, conflict)
            decision["speed"] = max(0, new_speed)
            decision["action"] = f"人机冲突：与 {conflict['other_vehicle']} 保持距离"

            decisions[truck_id] = decision
            continue

        # 检查前方车辆冲突（跟车）
        front_conflict = check_front_vehicle_conflict(truck, vehicle_states)
        if front_conflict:
            decision["conflict_detected"] = "front_vehicle"
            # 保持安全距离
            safe_distance = front_conflict["safe_distance"]
            current_distance = front_conflict["distance"]

            if current_distance < safe_distance * 0.8:
                # 距离太近，需要减速
                ratio = current_distance / safe_distance
                decision["speed"] = truck.speed * ratio * 0.8
                decision["action"] = f"跟车过近：减速保持距离"

        # 死锁预防检查（预测性）
        if truck_id in vehicle_plans:
            planned_route = vehicle_plans[truck_id].get("planned_route", [])
            if planned_route and check_potential_deadlock(truck_id, planned_route,
                                                          vehicle_states, road_network):
                # 预测到顶牛风险，尝试重新规划路径
                if shortest_path_matrix:
                    destination = vehicle_plans[truck_id].get("destination")
                    avoid_links = list(deadlock_conflicts.keys())
                    new_route = replan_route(
                        truck_id, (truck.x, truck.y), destination,
                        road_network, shortest_path_matrix, avoid_links
                    )
                    if new_route:
                        decision["route"] = new_route
                        decision["action"] = "路径重规划：绕行避让潜在冲突"

        decisions[truck_id] = decision

    return decisions


# ============================================================================
# 辅助工具函数
# ============================================================================

def load_vehicle_states_from_csv(csv_path: str) -> List[VehicleState]:
    """
    从CSV文件加载车辆状态

    CSV格式：
    vehicle_id,vehicle_type,x,y,road_id,lane_id,speed,angle,distance_on_road
    """
    import csv

    states = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            states.append(VehicleState.from_csv_row(row))
    return states


def visualize_conflicts(vehicle_states: List[VehicleState],
                       conflicts: Dict[str, Dict],
                       output_path: str = "conflicts_visualization.png"):
    """
    可视化冲突（需要matplotlib）

    这是一个辅助调试函数
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 绘制所有车辆
        for v in vehicle_states:
            color = {'company': 'green', 'other': 'red', 'human': 'darkgreen'}.get(v.type, 'gray')
            ax.plot(v.x, v.y, 'o', color=color, markersize=8)
            ax.annotate(v.id, (v.x, v.y), fontsize=8)

        # 高亮冲突车辆
        for vehicle_id, conflict in conflicts.items():
            # 找到车辆位置
            for v in vehicle_states:
                if v.id == vehicle_id:
                    ax.plot(v.x, v.y, 'r*', markersize=15)
                    break

        plt.savefig(output_path)
        plt.close()
    except ImportError:
        print("matplotlib未安装，跳过可视化")


def get_conflict_summary(conflicts: Dict[str, Dict]) -> str:
    """生成冲突摘要报告"""
    summary = []
    summary.append(f"检测到 {len(conflicts)} 个冲突：")

    deadlock_count = sum(1 for c in conflicts.values() if c.get("type") == "deadlock")
    human_count = sum(1 for c in conflicts.values() if c.get("type") == "human_conflict")

    summary.append(f"  - 顶牛风险: {deadlock_count} 次")
    summary.append(f"  - 人机冲突: {human_count} 次")

    # 预估扣分
    estimated_penalty = deadlock_count * 10 + human_count * 2
    summary.append(f"  - 预估扣分: {estimated_penalty} 分")

    return "\n".join(summary)


# ============================================================================
# 单元测试
# ============================================================================

if __name__ == "__main__":
    # 创建模拟数据
    print("=" * 60)
    print("港区运输冲突检测模块 - 单元测试")
    print("=" * 60)

    # 创建模拟路网
    rn = RoadNetwork()
    rn.add_link("link_1", "node_A", "node_B", lanes=2, width=5.5, is_bidirectional=True)
    rn.add_link("link_2", "node_B", "node_C", lanes=2, width=7.0, is_bidirectional=True)
    rn.add_link("link_3", "node_B", "node_D", lanes=1, width=4.0, is_bidirectional=False)

    # 创建模拟车辆状态
    # 公司集卡A（向北行驶，在link_1上）
    truck_a = VehicleState(
        vehicle_id="truck_A",
        vehicle_type="company",
        x=100.0, y=50.0,
        road_id="link_1", lane_id="link_1_1",
        speed=8.0, angle=np.pi/2,  # 向y正方向（向下）
        distance_on_road=50.0
    )

    # 其他公司集卡B（向南行驶，在link_1对向车道）
    truck_b = VehicleState(
        vehicle_id="truck_B",
        vehicle_type="other",
        x=100.0, y=80.0,
        road_id="link_1", lane_id="link_1_2",
        speed=6.0, angle=-np.pi/2,  # 向y负方向（向上）
        distance_on_road=70.0
    )

    # 人工驾驶集卡C（在truck_A后面同车道）
    truck_c = VehicleState(
        vehicle_id="truck_C",
        vehicle_type="human",
        x=100.0, y=30.0,
        road_id="link_1", lane_id="link_1_1",
        speed=5.0, angle=np.pi/2,
        distance_on_road=30.0
    )

    vehicle_states = [truck_a, truck_b, truck_c]

    # 测试1：顶牛检测
    print("\n【测试1】顶牛检测")
    print("-" * 40)
    deadlock_conflicts = detect_deadlock(vehicle_states, rn)
    if deadlock_conflicts:
        for vid, info in deadlock_conflicts.items():
            print(f"  车辆 {vid} 与 {info['other_vehicle']} 发生顶牛风险")
            print(f"    距离: {info['distance']:.1f}m, 严重程度: {info['severity']}")
    else:
        print("  未检测到顶牛")

    # 测试2：人机冲突检测
    print("\n【测试2】人机冲突检测")
    print("-" * 40)
    human_conflicts = detect_human_conflict(vehicle_states, rn)
    if human_conflicts:
        for vid, info in human_conflicts.items():
            print(f"  车辆 {vid} 与 {info['other_vehicle']} 发生人机冲突")
            print(f"    距离: {info['distance']:.1f}m, TTC: {info['ttc']:.1f}s")
    else:
        print("  未检测到人机冲突")

    # 测试3：速度调整
    print("\n【测试3】速度调整建议")
    print("-" * 40)
    if deadlock_conflicts:
        for vid, info in deadlock_conflicts.items():
            new_speed = adjust_speed_for_conflict(vid, vehicle_states, info)
            print(f"  车辆 {vid}: 当前速度 {truck_a.speed} m/s -> 建议速度 {new_speed} m/s")

    # 测试4：综合避险决策
    print("\n【测试4】综合避险决策")
    print("-" * 40)
    vehicle_plans = {
        "truck_A": {"destination": "node_C", "planned_route": ["link_1", "link_2"]}
    }
    decisions = resolve_conflicts(vehicle_states, vehicle_plans, rn)

    for vid, decision in decisions.items():
        print(f"  车辆 {vid}:")
        print(f"    冲突类型: {decision['conflict_detected']}")
        print(f"    建议速度: {decision['speed']} m/s")
        print(f"    动作: {decision['action']}")

    # 测试5：冲突摘要
    print("\n【测试5】冲突摘要")
    print("-" * 40)
    all_conflicts = {**deadlock_conflicts, **human_conflicts}
    print(get_conflict_summary(all_conflicts))

    print("\n" + "=" * 60)
    print("单元测试完成")
    print("=" * 60)
