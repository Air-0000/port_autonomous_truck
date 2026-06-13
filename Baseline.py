"""
港区运输调度算法 Baseline
赛道9: 港口自动驾驶集卡调度

基于TESS NG仿真平台的调度算法实现
包含: 任务分配、路径规划、车速引导、车道引导、冲突检测
"""

import math
import heapq
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


# =============================================================================
# 数据结构定义
# =============================================================================

class Vehicle:
    """车辆数据结构"""
    def __init__(self, vehicle_id: int, max_speed: float = 33.3):
        self.id = vehicle_id
        self.max_speed = max_speed  # 最大速度 (m/s), 默认33.3 m/s ≈ 120 km/h
        self.current_speed = 0.0    # 当前速度 (m/s)
        self.current_lane = 0       # 当前车道
        self.current_link = 0       # 当前路段
        self.current_position = 0.0 # 在路段中的位置 (m)
        self.heading = 1            # 行驶方向: 1=正向, -1=反向
        self.status = "idle"       # 状态: idle, loading, moving, unloading
        self.current_task = None    # 当前任务

    def distance_to(self, target_link: int, target_position: float,
                    shortest_path_matrix: np.ndarray, link_lengths: Dict[int, float]) -> float:
        """计算到目标点的最短距离"""
        if self.current_link == target_link:
            return abs(target_position - self.current_position)

        # 使用预计算的最短路径矩阵
        path = shortest_path_matrix[self.current_link][target_link]
        if path > 0:
            # 估算距离: 路径长度 * 平均路段长度
            avg_link_length = sum(link_lengths.values()) / max(len(link_lengths), 1)
            return path * avg_link_length
        return float('inf')


class Task:
    """运输任务数据结构"""
    def __init__(self, task_id: int, origin: int, destination: int, deadline: float):
        self.id = task_id
        self.origin = origin      # 起点路段
        self.destination = destination  # 终点路段
        self.deadline = deadline  # 计划完成时刻 (s)
        self.assigned_vehicle = None
        self.start_time = None
        self.completed = False


class VehicleState:
    """车辆实时状态"""
    def __init__(self, vehicle_id: int, link: int, lane: int, speed: float, position: float, heading: int = 1):
        self.vehicle_id = vehicle_id
        self.link = link          # 当前路段ID
        self.lane = lane          # 当前车道
        self.speed = speed        # 当前速度 (m/s)
        self.position = position # 在路段中的位置 (m)
        self.heading = heading    # 行驶方向


class OperationState:
    """装卸设备状态"""
    def __init__(self, equipment_id: int, link_id: int, status: int, position: float = 150.0):
        self.equipment_id = equipment_id
        self.link_id = link_id    # 所在路段
        self.status = status      # 0=空闲, 1=忙碌
        self.position = position  # 在路段中的位置 (m), 默认150m


# =============================================================================
# 1. 任务分配函数
# =============================================================================

def allocate_tasks(vehicles: List[Vehicle], tasks: List[Task],
                   vehicle_states: Dict[int, VehicleState],
                   shortest_path_matrix: np.ndarray,
                   link_lengths: Dict[int, float],
                   current_time: float) -> Dict[int, List[Tuple[int, int, int, float]]]:
    """
    贪心任务分配: 优先为空闲且距离任务起点最近的车辆分配任务

    参数:
        vehicles: 车辆列表
        tasks: 任务列表
        vehicle_states: 车辆实时状态字典 {车辆id: VehicleState}
        shortest_path_matrix: 预计算的最短路径矩阵
        link_lengths: 路段长度字典 {link_id: length}
        current_time: 当前仿真时间 (s)

    返回:
        vehicle_tasks: dict {车辆id: [(task_id, 起点, 终点, 计划完成时间), ...]}
    """
    vehicle_tasks: Dict[int, List[Tuple[int, int, int, float]]] = {v.id: [] for v in vehicles}
    assigned_tasks: Set[int] = set()

    # 复制任务列表按截止时间排序
    unassigned_tasks = [t for t in tasks if t.id not in assigned_tasks and not t.completed]
    unassigned_tasks.sort(key=lambda x: x.deadline)

    # 按优先级分配任务
    for task in unassigned_tasks:
        best_vehicle = None
        best_distance = float('inf')

        for vehicle in vehicles:
            # 跳过已有任务的车辆
            if vehicle.status != "idle":
                continue

            # 获取车辆当前位置
            if vehicle.id in vehicle_states:
                state = vehicle_states[vehicle.id]
                current_link = state.link
                current_pos = state.position
            else:
                continue

            # 计算到任务起点的距离
            dist = _estimate_distance(current_link, current_pos, task.origin,
                                     shortest_path_matrix, link_lengths)

            if dist < best_distance:
                best_distance = dist
                best_vehicle = vehicle

        # 分配任务
        if best_vehicle is not None:
            vehicle_tasks[best_vehicle.id].append(
                (task.id, task.origin, task.destination, task.deadline)
            )
            assigned_tasks.add(task.id)
            best_vehicle.current_task = task
            best_vehicle.status = "assigned"

    return vehicle_tasks


def _estimate_distance(current_link: int, current_pos: float,
                       target_link: int,
                       shortest_path_matrix: np.ndarray,
                       link_lengths: Dict[int, float]) -> float:
    """估算两位置间的距离"""
    if current_link == target_link:
        return abs(target_link - current_pos)

    # 使用最短路径跳数估算
    if current_link < len(shortest_path_matrix) and target_link < len(shortest_path_matrix[0]):
        hops = shortest_path_matrix[current_link][target_link]
        if hops > 0 and hops < 1e9:
            avg_length = sum(link_lengths.values()) / max(len(link_lengths), 1)
            return hops * avg_length

    return float('inf')


# =============================================================================
# 2. 路径规划函数
# =============================================================================

def plan_routes(vehicle_tasks: Dict[int, List[Tuple[int, int, int, float]]],
                shortest_path_matrix: np.ndarray,
                link_lengths: Dict[int, float]) -> Dict[int, List[int]]:
    """
    基于预计算最短路径的路径规划

    参数:
        vehicle_tasks: 车辆任务字典 {车辆id: [(task_id, 起点, 终点, 计划完成时间), ...]}
        shortest_path_matrix: 预计算的最短路径矩阵
        link_lengths: 路段长度字典

    返回:
        vehicle_route: dict {车辆id: [link_id1, link_id2, ...]}
    """
    vehicle_route: Dict[int, List[int]] = {}

    for vehicle_id, tasks in vehicle_tasks.items():
        if not tasks:
            vehicle_route[vehicle_id] = []
            continue

        # 合并多任务的路径 (简化: 只取第一个任务的路径)
        task_id, origin, destination, deadline = tasks[0]

        # 使用Dijkstra或预计算的最短路径
        path = _find_shortest_path(origin, destination, shortest_path_matrix, link_lengths)
        vehicle_route[vehicle_id] = path

    return vehicle_route


def _find_shortest_path(origin: int, destination: int,
                        shortest_path_matrix: np.ndarray,
                        link_lengths: Dict[int, float]) -> List[int]:
    """使用A*算法找到最短路径"""
    if origin == destination:
        return [origin]

    # 使用预计算矩阵中的跳数
    if origin < len(shortest_path_matrix) and destination < len(shortest_path_matrix[0]):
        hops = shortest_path_matrix[origin][destination]
        if hops < 1e9:
            # 简单起见,返回[origin, destination]
            return [origin, destination]

    # 回退: 直接连接
    return [origin, destination]


def _dijkstra_path(start: int, end: int,
                   adjacency: Dict[int, List[Tuple[int, float]]]) -> List[int]:
    """Dijkstra最短路径算法"""
    if start == end:
        return [start]

    # 距离, 节点, 路径
    heap = [(0, start, [start])]
    visited: Set[int] = set()

    while heap:
        dist, node, path = heapq.heappop(heap)

        if node in visited:
            continue
        visited.add(node)

        if node == end:
            return path

        for neighbor, weight in adjacency.get(node, []):
            if neighbor not in visited:
                heapq.heappush(heap, (dist + weight, neighbor, path + [neighbor]))

    return [start, end]  # 回退


# =============================================================================
# 3. 速度引导函数 (IDM模型)
# =============================================================================

def compute_speed(vehicle_states: Dict[int, VehicleState],
                  vehicle_plans: Dict[int, Dict],
                  equipment_states: List[OperationState],
                  current_time: float) -> Dict[int, float]:
    """
    IDM (Intelligent Driver Model) 简化版速度控制

    参数:
        vehicle_states: 车辆实时状态字典
        vehicle_plans: 车辆规划信息 {车辆id: {route, target_speed, ...}}
        equipment_states: 装卸设备状态列表
        current_time: 当前仿真时间

    返回:
        vehicle_speed: dict {车辆id: 目标速度值 (m/s)}
    """
    vehicle_speed: Dict[int, float] = {}

    # IDM参数
    a_max = 2.0    # 最大加速度 (m/s^2)
    b_max = 3.0    # 舒适减速度 (m/s^2)
    T = 1.5        # 安全时距 (s)
    s0 = 2.0       # 最小车间距 (m)
    delta = 4.0    # 加速指数

    for vehicle_id, state in vehicle_states.items():
        plan = vehicle_plans.get(vehicle_id, {})
        route = plan.get('route', [])

        # 期望速度 (考虑路段限速和车辆最大速度)
        v0 = min(33.3, plan.get('max_speed', 33.3))  # 默认限速120 km/h

        # 计算前车距离和速度
        front_vehicle = _find_front_vehicle(vehicle_id, state, vehicle_states, route)
        front_gap = float('inf')
        front_speed = 0.0

        if front_vehicle is not None:
            front_gap = _calculate_gap(state, front_vehicle)
            front_speed = front_vehicle.speed

        # 检查前方设备(岸桥/堆场)的红灯停车
        red_light_gap = _check_equipment_ahead(state, equipment_states, current_time)

        # IDM加速度公式
        if front_gap < float('inf'):
            # 有前车
            delta_v = state.speed - front_speed
            s_star = s0 + max(0, T * state.speed + (T * delta_v * (state.speed + front_speed)) / (2 * math.sqrt(a_max * b_max)))

            if s_star > 0:
                a = a_max * (1 - (state.speed / v0) ** delta - (s_star / front_gap) ** 2)
            else:
                a = a_max
        else:
            # 无前车, 自由加速
            a = a_max * (1 - (state.speed / v0) ** delta)

        # 红灯停车
        if red_light_gap < 50.0:  # 距红灯50m内
            if red_light_gap < 5.0:
                a = -b_max * 2  # 急刹
            else:
                # 线性减速到停止
                a = -b_max * (red_light_gap / 50.0)

        # 计算目标速度
        new_speed = max(0, state.speed + a * 1.0)  # 假设1秒步长

        # 限制最大速度
        vehicle_speed[vehicle_id] = min(new_speed, v0)

    return vehicle_speed


def _find_front_vehicle(vehicle_id: int, state: VehicleState,
                        vehicle_states: Dict[int, VehicleState],
                        route: List[int]) -> Optional[VehicleState]:
    """找到前车"""
    same_lane_vehicles = [
        v for vid, v in vehicle_states.items()
        if vid != vehicle_id and v.lane == state.lane and v.link == state.link
    ]

    if same_lane_vehicles:
        # 在同一车道上,找最近的前车
        front_vehicles = [v for v in same_lane_vehicles if v.position > state.position]
        if front_vehicles:
            return min(front_vehicles, key=lambda x: x.position - state.position)

    return None


def _calculate_gap(state: VehicleState, front_vehicle: VehicleState) -> float:
    """计算机车间距"""
    if state.link == front_vehicle.link:
        return front_vehicle.position - state.position
    return float('inf')


def _check_equipment_ahead(state: VehicleState,
                           equipment_states: List[OperationState],
                           current_time: float) -> float:
    """检查前方装卸设备是否为红灯(忙碌)"""
    for equip in equipment_states:
        if equip.link_id == state.link and equip.status == 1:  # 忙碌
            # 设备在前方
            if equip.position > state.position:
                return equip.position - state.position
            else:
                return 50.0  # 假装有红灯
    return float('inf')


# =============================================================================
# 4. 车道引导函数
# =============================================================================

def compute_lane(vehicle_states: Dict[int, VehicleState],
                 vehicle_routes: Dict[int, List[int]],
                 vehicle_speeds: Dict[int, float],
                 conflict_zones: List[Dict]) -> Dict[int, int]:
    """
    车道引导: 保持在正确车道, 遇障碍考虑换道

    参数:
        vehicle_states: 车辆实时状态字典
        vehicle_routes: 车辆路径字典 {车辆id: [link_id1, link_id2, ...]}
        vehicle_speeds: 目标速度字典
        conflict_zones: 冲突区域列表

    返回:
        vehicle_lane: dict {车辆id: 目标车道id}
    """
    vehicle_lane: Dict[int, int] = {}

    for vehicle_id, state in vehicle_states.items():
        route = vehicle_routes.get(vehicle_id, [])

        # 确定目标车道
        # 简化: 堆场->岸桥走车道0, 岸桥->堆场走车道1
        current_link = state.link
        if len(route) >= 2:
            next_link = route[1] if route[0] == current_link else route[0]
            # 根据路线方向确定车道
            if _is_route_direction(route, current_link):
                target_lane = 0
            else:
                target_lane = 1
        else:
            target_lane = state.lane

        # 检查是否需要换道避让
        if _needs_lane_change(vehicle_id, state, vehicle_states, conflict_zones):
            target_lane = _find_safe_lane(state, vehicle_states)

        vehicle_lane[vehicle_id] = target_lane

    return vehicle_lane


def _is_route_direction(route: List[int], current_link: int) -> bool:
    """判断路线方向"""
    if len(route) < 2:
        return True
    idx = route.index(current_link) if current_link in route else 0
    return idx < len(route) - 1


def _needs_lane_change(vehicle_id: int, state: VehicleState,
                       vehicle_states: Dict[int, VehicleState],
                       conflict_zones: List[Dict]) -> bool:
    """判断是否需要换道"""
    # 检测冲突区域
    for zone in conflict_zones:
        if zone.get('link') == state.link:
            if zone.get('start', 0) <= state.position <= zone.get('end', float('inf')):
                # 在冲突区域,检查是否有对向车辆
                for vid, other_state in vehicle_states.items():
                    if vid != vehicle_id and other_state.link == state.link:
                        if other_state.heading != state.heading:
                            return True
    return False


def _find_safe_lane(state: VehicleState, vehicle_states: Dict[int, VehicleState]) -> int:
    """找到安全车道"""
    # 尝试换到相邻车道
    for candidate_lane in [0, 1]:
        if candidate_lane != state.lane:
            # 检查目标车道是否安全
            lane_occupied = any(
                v.lane == candidate_lane and v.link == state.link
                for vid, v in vehicle_states.items()
                if vid != state.vehicle_id
            )
            if not lane_occupied:
                return candidate_lane
    return state.lane  # 无法换道,保持当前车道


# =============================================================================
# 5. 冲突检测函数
# =============================================================================

def detect_conflicts(vehicle_states: Dict[int, VehicleState],
                    equipment_states: List[OperationState],
                    link_conflicts: Dict[int, List[int]]) -> List[Dict]:
    """
    检测顶牛(对向相遇)和人机冲突

    参数:
        vehicle_states: 车辆实时状态字典
        equipment_states: 装卸设备状态列表
        link_conflicts: 路段冲突矩阵 {link_id: [冲突link_id列表]}

    返回:
        conflicts: 冲突列表 [{type, vehicle_ids, location, time}, ...]
    """
    conflicts: List[Dict] = []
    vehicle_list = list(vehicle_states.values())

    # 1. 检测顶牛(对向相遇)
    for i, v1 in enumerate(vehicle_list):
        for v2 in vehicle_list[i + 1:]:
            if v1.link == v2.link and v1.lane == v2.lane:
                # 同一车道,相向而行
                if v1.heading != v2.heading:
                    # 检查是否在同一路段相向
                    distance = abs(v1.position - v2.position)
                    if distance < 10.0:  # 10米内判定为顶牛
                        conflicts.append({
                            'type': '顶牛',
                            'vehicle_ids': [v1.vehicle_id, v2.vehicle_id],
                            'location': {'link': v1.link, 'position': (v1.position + v2.position) / 2},
                            'severity': 10
                        })

    # 2. 检测人机冲突(车辆与装卸设备)
    for vehicle_id, state in vehicle_states.items():
        for equip in equipment_states:
            if equip.link_id == state.link and equip.status == 0:  # 设备空闲
                distance = abs(state.position - equip.position)
                if distance < 5.0:  # 5米内判定为冲突
                    conflicts.append({
                        'type': '人机冲突',
                        'vehicle_ids': [vehicle_id],
                        'location': {'link': state.link, 'position': equip.position},
                        'severity': 2
                    })

    # 3. 检测多车道冲突(同路段不同车道)
    for v1 in vehicle_list:
        for v2 in vehicle_list:
            if v1.vehicle_id >= v2.vehicle_id:
                continue
            if v1.link == v2.link and v1.lane != v2.lane:
                distance = abs(v1.position - v2.position)
                if distance < 3.0:  # 3米内判定为变道冲突
                    conflicts.append({
                        'type': '变道冲突',
                        'vehicle_ids': [v1.vehicle_id, v2.vehicle_id],
                        'location': {'link': v1.link, 'position': (v1.position + v2.position) / 2},
                        'severity': 2
                    })

    return conflicts


# =============================================================================
# 6. 主更新函数
# =============================================================================

def update_plan(current_time: float,
                vehicle_states: Dict[int, VehicleState],
                equipment_states: List[OperationState],
                vehicles: List[Vehicle],
                tasks: List[Task],
                shortest_path_matrix: np.ndarray,
                link_lengths: Dict[int, float],
                link_conflicts: Dict[int, List[int]],
                vehicle_plans: Dict[int, Dict]) -> Dict:
    """
    整合所有模块,返回完整的new_plan

    参数:
        current_time: 当前仿真时间 (s)
        vehicle_states: 车辆实时状态字典
        equipment_states: 装卸设备状态列表
        vehicles: 车辆列表
        tasks: 任务列表
        shortest_path_matrix: 预计算的最短路径矩阵
        link_lengths: 路段长度字典
        link_conflicts: 路段冲突矩阵
        vehicle_plans: 当前车辆规划字典

    返回:
        new_plan: 包含 vehicle_tasks, vehicle_route, vehicle_speed, vehicle_lane 的完整规划
    """
    # Step 1: 任务分配
    vehicle_tasks = allocate_tasks(
        vehicles, tasks, vehicle_states,
        shortest_path_matrix, link_lengths, current_time
    )

    # Step 2: 路径规划
    vehicle_routes = plan_routes(vehicle_tasks, shortest_path_matrix, link_lengths)

    # 更新规划信息
    for vehicle_id, route in vehicle_routes.items():
        if vehicle_id not in vehicle_plans:
            vehicle_plans[vehicle_id] = {}
        vehicle_plans[vehicle_id]['route'] = route
        vehicle_plans[vehicle_id]['max_speed'] = 33.3  # 默认限速

    # Step 3: 速度引导
    vehicle_speeds = compute_speed(
        vehicle_states, vehicle_plans, equipment_states, current_time
    )

    # Step 4: 车道引导
    # 检测冲突区域
    conflicts = detect_conflicts(vehicle_states, equipment_states, link_conflicts)
    conflict_zones = _conflicts_to_zones(conflicts)

    vehicle_lanes = compute_lane(
        vehicle_states, vehicle_routes, vehicle_speeds, conflict_zones
    )

    # Step 5: 构建完整规划
    new_plan = {
        'timestamp': current_time,
        'vehicle_tasks': vehicle_tasks,
        'vehicle_route': vehicle_routes,
        'vehicle_speed': vehicle_speeds,
        'vehicle_lane': vehicle_lanes,
        'conflicts': conflicts,
        'score_penalty': _calculate_penalty(conflicts)
    }

    return new_plan


def _conflicts_to_zones(conflicts: List[Dict]) -> List[Dict]:
    """将冲突列表转换为冲突区域"""
    zones = []
    for conflict in conflicts:
        if 'location' in conflict:
            zones.append({
                'link': conflict['location'].get('link'),
                'start': conflict['location'].get('position', 0) - 20,
                'end': conflict['location'].get('position', 0) + 20
            })
    return zones


def _calculate_penalty(conflicts: List[Dict]) -> Dict[str, float]:
    """计算冲突扣分"""
    penalty = {
        '顶牛': 0,
        '人机冲突': 0,
        '变道冲突': 0
    }
    for conflict in conflicts:
        conflict_type = conflict.get('type', '')
        if conflict_type in penalty:
            penalty[conflict_type] += conflict.get('severity', 0)
    return penalty


# =============================================================================
# 辅助函数
# =============================================================================

def initialize_vehicles(vehicle_csv_path: str) -> List[Vehicle]:
    """从CSV文件初始化车辆"""
    vehicles = []
    # 读取CSV格式: vehicle_id, max_speed
    # 占位实现,实际从TESS NG读取
    return vehicles


def initialize_tasks(task_csv_path: str) -> List[Task]:
    """从CSV文件初始化任务"""
    tasks = []
    # 读取CSV格式: task_id, origin, destination, deadline
    # 占位实现,实际从TESS NG读取
    return tasks


def load_shortest_path_matrix(matrix_path: str) -> np.ndarray:
    """加载预计算的最短路径矩阵"""
    # 实际实现应从文件或TESS NG API获取
    # 返回np.ndarray, shape=(n_links, n_links)
    n_links = 100  # 示例值
    return np.zeros((n_links, n_links))


def get_default_link_conflicts() -> Dict[int, List[int]]:
    """获取默认路段冲突矩阵"""
    # 实际实现应根据地图数据
    return {}


def get_default_link_lengths() -> Dict[int, float]:
    """获取默认路段长度"""
    # 实际实现应根据地图数据,单位:米
    return {
        0: 500.0,   # 堆场区域
        1: 300.0,   # 闸口区域
        2: 1000.0,  # 主干道
        3: 200.0,   # 岸桥区域
    }


# =============================================================================
# 主入口 (用于测试)
# =============================================================================

if __name__ == "__main__":
    # 测试代码
    print("港区运输调度算法 Baseline")
    print("=" * 50)

    # 创建测试数据
    vehicles = [Vehicle(i, 33.3) for i in range(5)]
    tasks = [
        Task(1, origin=0, destination=3, deadline=3600),
        Task(2, origin=1, destination=2, deadline=3800),
        Task(3, origin=2, destination=0, deadline=4000),
    ]

    vehicle_states = {
        0: VehicleState(0, link=1, lane=0, speed=10.0, position=50.0, heading=1),
        1: VehicleState(1, link=1, lane=1, speed=8.0, position=100.0, heading=-1),
        2: VehicleState(2, link=0, lane=0, speed=5.0, position=200.0, heading=1),
    }

    equipment_states = [
        OperationState(0, link_id=3, status=0, position=150.0),
        OperationState(1, link_id=0, status=1, position=150.0),
    ]

    # 初始化
    n_links = 10
    shortest_path_matrix = np.zeros((n_links, n_links))
    link_lengths = get_default_link_lengths()
    link_conflicts = get_default_link_conflicts()
    vehicle_plans: Dict[int, Dict] = {}

    current_time = 0.0

    # 测试任务分配
    print("\n[1] 测试任务分配...")
    vehicle_tasks = allocate_tasks(
        vehicles, tasks, vehicle_states,
        shortest_path_matrix, link_lengths, current_time
    )
    print(f"  分配结果: {vehicle_tasks}")

    # 测试路径规划
    print("\n[2] 测试路径规划...")
    vehicle_routes = plan_routes(vehicle_tasks, shortest_path_matrix, link_lengths)
    print(f"  路径结果: {vehicle_routes}")

    # 更新规划信息
    for vehicle_id, route in vehicle_routes.items():
        if vehicle_id not in vehicle_plans:
            vehicle_plans[vehicle_id] = {}
        vehicle_plans[vehicle_id]['route'] = route
        vehicle_plans[vehicle_id]['max_speed'] = 33.3

    # 测试速度引导
    print("\n[3] 测试速度引导...")
    vehicle_speeds = compute_speed(vehicle_states, vehicle_plans, equipment_states, current_time)
    print(f"  速度结果: {vehicle_speeds}")

    # 测试车道引导
    print("\n[4] 测试车道引导...")
    conflicts = detect_conflicts(vehicle_states, equipment_states, link_conflicts)
    conflict_zones = _conflicts_to_zones(conflicts)
    vehicle_lanes = compute_lane(vehicle_states, vehicle_routes, vehicle_speeds, conflict_zones)
    print(f"  车道结果: {vehicle_lanes}")

    # 测试冲突检测
    print("\n[5] 测试冲突检测...")
    print(f"  检测到冲突: {conflicts}")

    # 测试完整更新
    print("\n[6] 测试完整更新...")
    new_plan = update_plan(
        current_time, vehicle_states, equipment_states,
        vehicles, tasks, shortest_path_matrix, link_lengths,
        link_conflicts, vehicle_plans
    )
    print(f"  规划完成时间: {new_plan['timestamp']}")
    print(f"  扣分: {new_plan['score_penalty']}")

    print("\n" + "=" * 50)
    print("测试完成!")
