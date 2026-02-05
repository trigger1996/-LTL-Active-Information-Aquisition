import math
import heapq

def rolling_a_star(prm, start, goal, R=15, k=5, w_f=0.7, w_g=1.0):
    """
    改进滚动 A*:
    prm: PRM 对象
    start, goal: (x,y)
    R: 探测半径
    k: 子目标候选节点数量（距离 goal 最近）
    w_f: f(x) 权重
    w_g: g(x) 权重
    """
    path_total = [start]
    current = start

    while True:
        # 1️⃣ 找半径 R 内的节点
        local_nodes = [n for n in prm.nodes if math.hypot(n[0]-current[0], n[1]-current[1]) <= R]

        if not local_nodes:
            print("No nodes in local radius!")
            break

        # 确保 goal 包含在局部图候选中
        if goal not in local_nodes:
            local_nodes.append(goal)

        # 2️⃣ 选出 k 个离 goal 最近的节点作为候选
        local_nodes.sort(key=lambda n: math.hypot(n[0]-goal[0], n[1]-goal[1]))
        candidates = local_nodes[:k]

        # 3️⃣ 构建局部图
        local_graph = {}
        for n in local_nodes:
            neighbors = [nb for nb in prm.graph.get(n, []) if nb in local_nodes]
            local_graph[n] = neighbors

        # 4️⃣ 计算 f+g 选择 subgoal
        f_g_values = {}
        for node in candidates:
            f_val = dijkstra_cost(local_graph, current, node)
            g_val = math.hypot(node[0]-goal[0], node[1]-goal[1])
            f_g_values[node] = w_f * f_val + w_g * g_val

        subgoal = min(f_g_values, key=f_g_values.get)

        # 5️⃣ 局部 A* 搜索到 subgoal
        came_from, found = astar_local(current, subgoal, local_graph)
        if not found:
            print("Local A* failed!")
            break

        path = reconstruct_path(came_from, subgoal)
        if len(path) < 2:
            break

        # 6️⃣ 移动一步到路径下一点
        current = path[1]
        path_total.append(current)

        # 7️⃣ 检查是否到达 goal
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
