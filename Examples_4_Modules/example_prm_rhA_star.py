import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mapping_2D.prm_2D import PRM
from mapping_2D.RH_Astar import rolling_a_star

def load_map_from_png(png_path, scale=1.0, threshold=50):
    """
    scale > 1: 放大, 使用双线性插值
    scale < 1: 缩小, 最近邻
    """
    img = Image.open(png_path).convert("L")
    w, h = img.size
    new_w, new_h = int(w*scale), int(h*scale)

    if scale > 1.0:
        img = img.resize((new_w, new_h), Image.BILINEAR)
    elif scale < 1.0:
        img = img.resize((new_w, new_h), Image.NEAREST)

    arr = np.array(img)
    map_array = (arr < threshold).astype(np.uint8)
    return map_array


def random_point_in_region(map_array, region):
    x_min, y_min, x_max, y_max = region
    h, w = map_array.shape
    while True:
        x = random.uniform(x_min, min(x_max, w-1))
        y = random.uniform(y_min, min(y_max, h-1))
        if map_array[int(y), int(x)] == 0:
            return (x, y)

def snap_to_free(map_array, point):
    """
    将点 (x,y) 对齐到最近的自由栅格中心
    """
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = map_array.shape

    if map_array[y, x] == 0:  # 自由点
        return (x+0.5, y+0.5)

    # 搜索附近自由点
    for r in range(1, max(h, w)):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h and map_array[ny, nx] == 0:
                    return (nx+0.5, ny+0.5)
    raise ValueError("No free point near", point)

def plot_prm(map_array, prm, path=None):
    """
    使用 matplotlib 绘制地图、PRM 节点、边和路径
    """
    h, w = map_array.shape
    plt.figure(figsize=(8,8))
    plt.imshow(map_array, cmap='gray_r', origin='upper')

    # 绘制 PRM 边
    for p, neighbors in prm.graph.items():
        for q in neighbors:
            plt.plot([p[0], q[0]], [p[1], q[1]], color='blue', linewidth=0.5, alpha=0.5)

    # 绘制 PRM 节点
    xs, ys = zip(*prm.nodes)
    plt.scatter(xs, ys, c='blue', s=10, label='Nodes')

    # 绘制路径
    if path:
        px, py = zip(*path)
        plt.plot(px, py, color='red', linewidth=2, label='Path')

    # 绘制 start 和 goal
    plt.scatter([prm.start[0]], [prm.start[1]], c='green', s=50, label='Start')
    plt.scatter([prm.goal[0]], [prm.goal[1]], c='red', s=50, label='Goal')

    plt.legend()
    plt.gca().invert_yaxis()  # 保持 y 方向与 numpy array 一致
    plt.title("PRM Path Planning")
    # plt.show()

# ---------------- 可视化 ----------------
def plot_rolling(map_array, prm, path, start, goal):
    plt.figure(figsize=(8,8))
    plt.imshow(map_array, cmap='gray_r', origin='upper')

    # PRM 网络
    for p, neighbors in prm.graph.items():
        for q in neighbors:
            plt.plot([p[0], q[0]], [p[1], q[1]], color='blue', linewidth=0.5, alpha=0.5)

    # 节点
    xs, ys = zip(*prm.nodes)
    plt.scatter(xs, ys, c='blue', s=10, label='Nodes')

    # 滚动路径
    if path:
        px, py = zip(*path)
        plt.plot(px, py, color='red', linewidth=2, label='Rolling Path')

    # 起点/终点
    plt.scatter([start[0]], [start[1]], c='green', s=50, label='Start')
    plt.scatter([goal[0]], [goal[1]], c='red', s=50, label='Goal')

    plt.legend()
    plt.gca().invert_yaxis()
    plt.title("Rolling A* Path on PRM")

if __name__ == "__main__":
    png_map = "Examples_4_Modules/example_map_2D/map.png"

    start_region = (1, 1, 10, 10)
    goal_region = (40, 40, 50, 50)
    R = 15  # 探测半径

    # 1️⃣ 读取 PNG 并转为栅格地图
    map_array = load_map_from_png(png_map, scale=0.05)   # 注意， 在实际论文中，我们最后不讨论原图的最优性，我们只讨论建图后的结果

    start = snap_to_free(map_array, random_point_in_region(map_array, start_region))
    goal = snap_to_free(map_array, random_point_in_region(map_array, goal_region))

    # 2️⃣ 构建 PRM 并寻找路径
    prm = PRM(start=start, goal=goal, num_nodes=300, map_array=map_array, max_sample_dist=R)
    path = prm.find_path()
    lts_individual = prm.to_networkx()
    print("Path length:", len(path))
    if path:
        print("Path:", path[:5], "...", path[-5:])
    else:
        print("No path found!")
        exit()

    # 3️⃣ 绘图
    plot_prm(map_array, prm, path)

    # 4️⃣ 滚动 A*
    path_total = rolling_a_star(prm, start, goal, R=R)
    print(f"Total path length: {len(path_total)}")

    # 5️⃣ 可视化
    plot_rolling(map_array, prm, path_total, start, goal)

    plt.show()
    print("finished!")
