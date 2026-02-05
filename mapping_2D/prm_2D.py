import random
import math
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

import random
import math
from scipy.spatial import KDTree

class PRM:
    def __init__(self, start, goal, num_nodes, map_array, max_sample_dist=5.0):
        self.start = start
        self.goal = goal
        self.num_nodes = num_nodes
        self.map_array = map_array
        self.h, self.w = map_array.shape
        self.max_sample_dist = max_sample_dist

        self.nodes = [start, goal]
        self.graph = {start: [], goal: []}

    @staticmethod
    def dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def in_obstacle(self, p):
        x, y = int(p[0]), int(p[1])
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return True
        return self.map_array[y, x] == 1

    def sample_nodes(self):
        # 初始随机采样
        while len(self.nodes) < self.num_nodes:
            p = (random.uniform(0, self.w-1), random.uniform(0, self.h-1))
            if not self.in_obstacle(p):
                self.nodes.append(p)
                self.graph[p] = []

        # KDTree 加速距离查询
        tree = KDTree(self.nodes)

        # 检查每个节点的邻居距离
        new_nodes = []
        for node in self.nodes:
            # 查询距离 node 最近的 k 个节点
            dists, idxs = tree.query(node, k=min(10, len(self.nodes)))
            # 去掉自己
            dists = dists[1:]
            if len(dists) == 0 or min(dists) > self.max_sample_dist:
                # 在 node 附近随机生成一个新节点
                for _ in range(10):  # 最多尝试 10 次
                    angle = random.uniform(0, 2*math.pi)
                    r = random.uniform(0.5, self.max_sample_dist)
                    new_x = node[0] + r*math.cos(angle)
                    new_y = node[1] + r*math.sin(angle)
                    new_p = (new_x, new_y)
                    if 0 <= new_x < self.w and 0 <= new_y < self.h and not self.in_obstacle(new_p):
                        new_nodes.append(new_p)
                        break
        # 加入新节点
        for p in new_nodes:
            self.nodes.append(p)
            self.graph[p] = []
        # 最后更新 KDTree
        tree = KDTree(self.nodes)
    # def sample_nodes(self):
    #     while len(self.nodes) < self.num_nodes:
    #         p = (random.uniform(0, self.w-1), random.uniform(0, self.h-1))
    #         if not self.in_obstacle(p):
    #             self.nodes.append(p)
    #             self.graph[p] = []

    # ---------------- 碰撞检测 ----------------
    def collision_free(self, a, b):
        """Bresenham 离散线碰撞检测"""
        x0, y0 = int(a[0]), int(a[1])
        x1, y1 = int(b[0]), int(b[1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1>x0 else -1
        sy = 1 if y1>y0 else -1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.in_obstacle((x, y)):
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.in_obstacle((x, y)):
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return not self.in_obstacle((x1, y1))

    # ---------------- 建图 ----------------
    def build_roadmap(self, k=5, r_max=None):
        """
        k: 每个节点最多连接 k 个最近邻
        r_max: 最大连接距离，超过不连接
        """
        tree = KDTree(self.nodes)
        for p in self.nodes:
            dists, idxs = tree.query(p, k=k + 1)
            for dist, idx in zip(dists[1:], idxs[1:]):
                if r_max is not None and dist > r_max:
                    continue  # 超过最大半径就不连
                q = self.nodes[idx]
                if self.collision_free(p, q):
                    self.graph[p].append(q)

        # 强制目标至少有几条边
        dists, idxs = tree.query(self.goal, k=k + 1)
        for dist, idx in zip(dists[1:], idxs[1:]):
            if r_max is not None and dist > r_max:
                continue
            q = self.nodes[idx]
            if self.collision_free(self.goal, q) and q not in self.graph[self.goal]:
                self.graph[self.goal].append(q)

    # ---------------- A* 搜索 ----------------
    def find_path(self):
        self.sample_nodes()
        self.build_roadmap()

        open_set = {self.start}
        came_from = {}
        g = {n: float('inf') for n in self.nodes}
        f = {n: float('inf') for n in self.nodes}

        g[self.start] = 0
        f[self.start] = self.dist(self.start, self.goal)

        while open_set:
            cur = min(open_set, key=lambda n: f[n])
            if cur == self.goal:
                return self._reconstruct(came_from, cur)

            open_set.remove(cur)
            for nb in self.graph[cur]:
                tentative = g[cur] + self.dist(cur, nb)
                if tentative < g[nb]:
                    came_from[nb] = cur
                    g[nb] = tentative
                    f[nb] = tentative + self.dist(nb, self.goal)
                    open_set.add(nb)

        return []

    def to_networkx(self):
        """
        返回 networkx.Graph
        节点: q0, q1, q2, ...
        节点属性: pos=(x,y), is_start, is_goal
        """
        G = nx.Graph()

        # 1️⃣ 建立编号映射
        node_id_map = {}
        node_list = list(self.nodes)

        # 强制 start 是 q0
        node_list.remove(self.start)
        node_list = [self.start] + node_list

        for i, p in enumerate(node_list):
            qid = f"q{i}"
            node_id_map[p] = qid
            G.add_node(
                qid,
                pos=p,  # 坐标作为属性
                #is_start=(p == self.start),
                #is_goal=(p == self.goal)
            )

        # 2️⃣ 加边
        for p, neighbors in self.graph.items():
            for q in neighbors:
                G.add_edge(
                    node_id_map[p],
                    node_id_map[q],
                    weight=self.dist(p, q)
                )

        return G, node_id_map

    @staticmethod
    def _reconstruct(came_from, cur):
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        return path[::-1]
