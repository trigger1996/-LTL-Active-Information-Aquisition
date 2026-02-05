import math
import heapq

def rolling_a_star(prm, start, goal, R=15, k=10, w_f=0.7, w_g=1.0, w_v=50.0):
    """
    带记忆的 Rolling A*
    w_v: 已访问节点惩罚权重
    """
    path_total = [start]
    current = start

    local_graph = {}            # 🧠 记忆局部地图
    visited = set([start])      # 🚫 防止震荡

    while True:
        # 1️⃣ 找半径 R 内的节点
        # local_nodes = [
        #     n for n in prm.nodes
        #     if math.hypot(n[0]-current[0], n[1]-current[1]) <= R
        # ]
        local_nodes, _, g_cost = dijkstra_ball(prm, current, R)
        local_nodes = list(local_nodes)

        if not local_nodes:
            print("No nodes in local radius!")
            break

        # 2️⃣ 更新「记忆局部图」
        for n in local_nodes:
            if n not in local_graph:
                local_graph[n] = []

            for nb in prm.graph.get(n, []):
                if nb in local_nodes and nb not in local_graph[n]:
                    local_graph[n].append(nb)

        # 3️⃣ 选 k 个离 goal 最近的候选
        local_nodes.sort(key=lambda n: math.hypot(n[0]-goal[0], n[1]-goal[1]))
        candidates = local_nodes[:k]

        # 4️⃣ 计算带记忆的 f + g + visited penalty
        f_g_values = {}
        for node in candidates:
            f_val = dijkstra_cost(local_graph, current, node)
            g_val = math.hypot(node[0]-goal[0], node[1]-goal[1])
            v_penalty = w_v if node in visited else 0.0

            f_g_values[node] = w_f * f_val + w_g * g_val + v_penalty

        sorted_subgoals = sorted(f_g_values, key=f_g_values.get)

        # 5️⃣ 在「记忆图」上局部 A*
        path = []
        for subgoal in sorted_subgoals:
            came_from, found = astar_local(current, subgoal, local_graph)
            if not found:
                continue

            path = reconstruct_path(came_from, subgoal)
            if len(path) >= 2:
                break

        if len(path) < 2:
            print("Local A* failed!")
            break

        # 6️⃣ 向前滚动一步
        current = path[1]
        path_total.append(current)
        visited.add(current)

        # 7️⃣ 是否到达 goal
        if math.hypot(current[0]-goal[0], current[1]-goal[1]) < 1e-3:
            break

    return path_total



# ----------------- 工具函数 -----------------
def astar_local(start, goal, graph):
    """标准 A* 在局部图上搜索"""
    open_set = []
    heapq.heappush(open_set, (euclid(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current == goal:
            return came_from, True

        closed_set.add(current)
        for nb in graph.get(current, []):
            if nb in closed_set:
                continue
            tentative = g + euclid(current, nb)
            if tentative < g_score.get(nb, float('inf')):
                g_score[nb] = tentative
                heapq.heappush(open_set, (tentative + euclid(nb, goal), tentative, nb))
                came_from[nb] = current

    return came_from, False


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def dijkstra_cost(graph, start, goal):
    """局部图 Dijkstra 计算 start -> goal 最短代价"""
    heap = [(0, start)]
    visited = set()
    costs = {start: 0}

    while heap:
        cost, node = heapq.heappop(heap)
        if node == goal:
            return cost
        if node in visited:
            continue
        visited.add(node)
        for nb in graph.get(node, []):
            if nb in visited:
                continue
            new_cost = cost + euclid(node, nb)
            if new_cost < costs.get(nb, float('inf')):
                costs[nb] = new_cost
                heapq.heappush(heap, (new_cost, nb))
    return float('inf')

def dijkstra_ball(prm, start, R):
    """
    返回：
    local_nodes: 所有 d(start, x) ≤ R 的节点
    local_graph: induced subgraph
    g_cost: start 到各点的最短路径代价
    """
    pq = [(0.0, start)]
    g_cost = {start: 0.0}
    visited = set()

    while pq:
        cost, u = heapq.heappop(pq)
        if cost > R:
            continue
        if u in visited:
            continue
        visited.add(u)

        for v in prm.graph[u]:
            new_cost = cost + prm.dist(u, v)
            if new_cost < g_cost.get(v, float("inf")):
                g_cost[v] = new_cost
                heapq.heappush(pq, (new_cost, v))

    local_nodes = set(g_cost.keys())

    local_graph = {
        u: [v for v in prm.graph[u] if v in local_nodes]
        for u in local_nodes
    }

    return local_nodes, local_graph, g_cost
