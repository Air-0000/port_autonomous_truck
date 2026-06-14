# Port Autonomous Truck — Conflict Avoidance Algorithms

**港区自动驾驶集卡调度 — 冲突避免与路径规划算法**

This project implements conflict avoidance and scheduling algorithms for autonomous container trucks operating within a port terminal. It is designed for **Track 9: Port Autonomous Truck Dispatching** of the TESS NG simulation platform. The primary goal is to safely dispatch company-owned autonomous trucks to complete container transport tasks while minimizing scoring penalties from collisions, deadlocks, and human-machine conflicts.

---

## Description

Autonomous trucks at a port must navigate between container yards, gate areas, main roads, and quay crane zones. They share the road network with manually driven trucks and other vehicles. This project provides two layers of algorithm:

- **Baseline (`Baseline.py`)** — A complete greedy dispatching scheduler with IDM-based speed control and basic lane guidance. Suitable as a starting benchmark.
- **Conflict Avoidance (`conflict_avoidance.py`)** — A dedicated detection and resolution module that identifies deadlocks (head-on meetings on narrow roads) and human-machine conflicts, then adjusts speed or replans routes to avoid them.

---

## Algorithm Explanation

### Baseline (`Baseline.py`)

The baseline implements a six-module pipeline that runs every simulation tick:

| Step | Module | Description |
|------|--------|-------------|
| 1 | **Task Allocation** | Greedy assignment: idle vehicles nearest a task origin receive the task. Tasks are sorted by deadline (earliest deadline first). |
| 2 | **Route Planning** | Uses a precomputed shortest-path matrix to assign a route (list of link IDs) from task origin to destination. Falls back to Dijkstra/A* if the matrix is unavailable. |
| 3 | **Speed Guidance (IDM)** | An **Intelligent Driver Model (IDM)** controls vehicle acceleration/deceleration. Parameters: max acceleration `2.0 m/s²`, comfortable deceleration `3.0 m/s²`, safe time headway `1.5 s`, minimum gap `2.0 m`. The model accounts for a lead vehicle and red-light equipment ahead. |
| 4 | **Lane Guidance** | Determines target lane based on route direction (yard→quay → lane 0; quay→yard → lane 1). Triggers lane changes to avoid opposing traffic in conflict zones. |
| 5 | **Conflict Detection** | Detects three conflict types:
  - **Deadlock (顶牛)** — Two vehicles approaching head-on in the same lane within 10 m (severity 10).
  - **Human-machine conflict (人机冲突)** — Vehicle within 5 m of active loading/unloading equipment (severity 2).
  - **Lane-change conflict (变道冲突)** — Vehicles in different lanes within 3 m (severity 2). |
| 6 | **Plan Update** | Aggregates all outputs (`vehicle_tasks`, `vehicle_route`, `vehicle_speed`, `vehicle_lane`, `conflicts`, `score_penalty`) into a single `new_plan` dict returned each tick. |

**Scoring Penalties:**
- Deadlock: 10 points × occurrence count
- Human-machine conflict: 2 points × occurrence count
- Lane-change conflict: 2 points × occurrence count

### Conflict Avoidance (`conflict_avoidance.py`)

A companion module with richer data structures and safety-focused algorithms for the same TESS NG Track 9 problem.

**Key Data Structures:**
- `VehicleState` — Vehicle with position (x, y), road/lane IDs, speed, heading angle, and type (`company`, `other`, `human`).
- `RoadNetwork` — Road topology with links, nodes, adjacency, lane width, and bidirectional flags.

**Core Functions:**

| Function | Description |
|----------|-------------|
| `detect_deadlock()` | Identifies head-on meetings on **narrow roads** (width < 6 m, cannot pass). Checks both vehicles moving toward each other; classifies severity as `high`/`medium`/`low` based on direction and speed. |
| `detect_human_conflict()` | Detects company vs. human-driven truck proximity within 20 m on the same or adjacent lanes. Computes Time To Collision (TTC). |
| `check_potential_deadlock()` | **Predictive** — looks ahead `LOOKAHEAD_TIME` (5 s) along the planned route to forecast deadlock risk before it occurs. |
| `replan_route()` | When a conflict is predicted, computes a new path that avoids blocked links. Uses BFS to find detours around forbidden segments. |
| `adjust_speed_for_conflict()` | Returns recommended speed based on conflict type: -1.0 (immediate stop) for high-severity deadlocks, speed × 0.6 for medium, reduced speed for human conflicts. |
| `resolve_conflicts()` | **Master decision function**: orchestrates all detection and resolution steps, returning per-vehicle speed, route, lane-change flag, and action description. |
| `check_front_vehicle_conflict()` | Car-following safety: ensures safe headway (2-second rule + 10 m buffer). |

**Tunable Parameters** (top of file):
- `SAFE_DISTANCE_DEADLOCK = 15.0 m`
- `SAFE_DISTANCE_HUMAN = 20.0 m`
- `SAFE_DISTANCE_CAR_FOLLOWING = 25.0 m`
- `MAX_TRUCK_SPEED = 15.0 m/s`
- `SPEED_REDUCTION_RATIO = 0.6`
- `BRAKE_DECELERATION = 3.0 m/s²`
- `REACTION_TIME = 1.0 s`
- `LOOKAHEAD_TIME = 5.0 s`

---

## Key Files

| File | Description |
|------|-------------|
| `Baseline.py` | Full baseline dispatching algorithm: task allocation, A*/Dijkstra route planning, IDM speed control, lane guidance, conflict detection, and scoring penalty calculation. Includes a self-contained test harness. |
| `conflict_avoidance.py` | Dedicated conflict detection and avoidance module: deadlock detection (head-on meetings), human-machine conflict detection, predictive route replanning, speed adjustment, and a comprehensive `resolve_conflicts()` orchestrator. Includes unit tests. |
| `README.md` | This file. |

---

## Status

🟢 **Active Development** — Core algorithms are implemented and unit-tested. The modules are designed to integrate with the TESS NG simulation platform via CSV-based vehicle state loading and structured plan dicts.

### Planned Improvements
- Integration with live TESS NG API for real-time state feed
- Multi-task path merging (beyond single-task-per-vehicle)
- Cooperative lane-change negotiation between company trucks
- Machine-learning-based speed prediction for human-driven vehicles
- Route replanning with dynamic traffic congestion awareness
