#!/usr/bin/env python
# coding: utf-8

import os
import mujoco
import mujoco.viewer
import numpy as np
import heapq

# ----------------------------
# D* Lite 核心数据结构（3D 栅格）
# ----------------------------
class Node:
    def __init__(self, pos):
        self.pos = pos
        self.g = float('inf')
        self.rhs = float('inf')
        self.h = 0

    def __lt__(self, other):
        return (min(self.g, self.rhs) + self.h) < (min(other.g, other.rhs) + other.h)

class DStarLite:
    def __init__(self, start, goal, grid):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.nodes = {}
        self.U = []
        self.km = 0

        # 初始化节点
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    self.nodes[(x, y, z)] = Node((x, y, z))

        self.nodes[goal].rhs = 0
        self.nodes[goal].h = self.heuristic(start, goal)
        heapq.heappush(self.U, (self.key(self.nodes[goal]), self.nodes[goal]))

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def key(self, n):
        k1 = min(n.g, n.rhs) + self.heuristic(self.start, n.pos) + self.km
        k2 = min(n.g, n.rhs)
        return (k1, k2)

    def get_neighbors(self, u):
        neighbors = []
        x, y, z = u.pos
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1] and 0 <= nz < self.grid.shape[2]:
                if self.grid[nx, ny, nz] == 0:
                    neighbors.append(self.nodes[(nx, ny, nz)])
        return neighbors

    def cost(self, a, b):
        return np.linalg.norm(np.array(a.pos)-np.array(b.pos))

    def update_vertex(self, u):
        if u.pos != self.goal:
            nbrs = self.get_neighbors(u)
            if nbrs:
                u.rhs = min([self.cost(u,s)+s.g for s in nbrs])
        if u in [x[1] for x in self.U]:
            self.U = [(k,node) for k,node in self.U if node!=u]
            heapq.heapify(self.U)
        if u.g != u.rhs:
            heapq.heappush(self.U,(self.key(u),u))

    def compute_shortest_path(self):
        while self.U:
            k_old, u = heapq.heappop(self.U)
            if u.g > u.rhs:
                u.g = u.rhs
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                u.g = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)

    def get_path(self):
        path = []
        current = self.nodes[self.start]
        path.append(current.pos)
        while current.pos != self.goal:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break
            current = min(neighbors, key=lambda n: n.g + self.cost(current, n))
            path.append(current.pos)
        return path

# ----------------------------
# MuJoCo环境加载
# ----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(base_dir, "Fencing_agent&obstacle_description", "fencing_arm_ver3.xml")
assert os.path.exists(xml_path), "❌ XML 文件不存在"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 在首次读取任何派生状态（如 site_xpos）之前，需要先做一次前向计算
mujoco.mj_forward(model, data)

# ----------------------------
# 三维栅格化末端空间
# ----------------------------
GRID_SIZE = (40, 20, 20)
workspace_grid = np.zeros(GRID_SIZE)  # 0可行，1障碍
# TODO: 填充障碍物位置为1

# EE 范围
EE_MIN = np.array([-3, -1, -1])
EE_MAX = np.array([ 3,  1,  1])

def ee2grid(ee_pos):
    """将EE实际坐标映射到栅格坐标"""
    grid_pos = (ee_pos - EE_MIN) / (EE_MAX - EE_MIN) * (np.array(GRID_SIZE) - 1)
    return tuple(np.clip(grid_pos.astype(int), 0, np.array(GRID_SIZE)-1))

def grid2ee(grid_pos):
    """将栅格坐标映射回EE实际坐标"""
    ee_pos = np.array(grid_pos) / (np.array(GRID_SIZE) - 1) * (EE_MAX - EE_MIN) + EE_MIN
    return ee_pos

# ----------------------------
# D* Lite路径规划
# ----------------------------
tip_name = "tip8_agent"
tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_name)
assert tip_id != -1, f"❌ 未找到 site {tip_name}"

start_ee_pos = data.site_xpos[tip_id][:3]
print("起点:", start_ee_pos)
goal_ee_pos = np.array([-2, 0, 0])

start_node = ee2grid(start_ee_pos)
goal_node  = ee2grid(goal_ee_pos)

dstar = DStarLite(start_node, goal_node, workspace_grid)
dstar.compute_shortest_path()
path = dstar.get_path()
# ----------------------------
# 将路径映射回实际EE坐标
# ----------------------------
ee_path = np.array([grid2ee(node) for node in path])
print("规划路径点(EE实际坐标):", ee_path)

# ----------------------------
# 雅可比IK求解器，只求前8个可控关节
# ----------------------------
def simple_ik(model, data, target_pos, ee_site_name="tip8_agent", max_iters=100, lr=0.1, tol=1e-4, controllable_joints=8):
    qpos = data.qpos.copy()
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    if ee_site_id == -1:
        raise ValueError(f"Site {ee_site_name} not found in model")

    for _ in range(max_iters):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[ee_site_id]

        error = target_pos - ee_pos
        if np.linalg.norm(error) < tol:
            break

        Jp_full = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, Jp_full, None, ee_site_id)
        Jp = Jp_full[:, :controllable_joints]

        dq = lr * np.linalg.pinv(Jp) @ error
        qpos[:controllable_joints] += dq

        # 夹住可控关节范围
        for j in range(controllable_joints):
            qpos[j] = np.clip(qpos[j], model.jnt_range[j,0], model.jnt_range[j,1])

    return qpos[:controllable_joints]

# 求解关节路径
ee_qpos_path = [simple_ik(model, data, np.array(p)) for p in ee_path]

# ----------------------------
# PD 扭矩控制参数
# ----------------------------
Kp = 50.0
Kd = 1.5
joint_ids = np.arange(8)
actuator_ids = np.arange(8)

def step_torque_control(qpos_target):
    qpos_now = data.qpos[joint_ids]
    qvel_now = data.qvel[joint_ids]
    qvel_target = np.zeros_like(qvel_now)

    torque = Kp * (qpos_target - qpos_now) + Kd * (qvel_target - qvel_now)
    ctrl_min = model.actuator_ctrlrange[actuator_ids, 0]
    ctrl_max = model.actuator_ctrlrange[actuator_ids, 1]

    data.ctrl[actuator_ids] = np.clip(torque, ctrl_min, ctrl_max)

# ----------------------------
# MuJoCo 可视化执行
# ----------------------------
viewer = None
try:
    viewer = mujoco.viewer.launch_passive(model, data)
    for qpos_target in ee_qpos_path:
        # 每个目标点执行若干step，使机械臂逐渐逼近
        for _ in range(1000):  # 200步 × 0.002s = 0.4秒
            step_torque_control(qpos_target)
            mujoco.mj_step(model, data)
            viewer.sync()
finally:
    if viewer is not None:
        try:
            viewer.sync()
            viewer.close()
        except Exception as e:
            print("关闭viewer时出错:", e)
        viewer = None
        import time
        time.sleep(0.1)
        print("🛑 MuJoCo viewer 已关闭")
