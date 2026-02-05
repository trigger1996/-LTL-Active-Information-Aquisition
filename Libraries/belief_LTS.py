import math
import random
import itertools
import networkx as nx
from scipy.spatial import KDTree
from mapping_2D.prm_2D import PRM

class BeliefState:
    """
    Belief state b ⊆ X
    """
    def __init__(self, states):
        self.states = set(states)

    def intersect(self, obs_set):
        self.states.intersection_update(obs_set)

    def copy(self):
        return BeliefState(self.states.copy())

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return f"BeliefState(size={len(self.states)})"

class BeliefPRM(PRM):
    """
    PRM + Belief State + Sensing Model + 自动生成观测、动作、信念空间
    """

    def __init__(
        self,
        start,
        goal,
        num_nodes,
        map_array,
        sensors,
        observation_function,
        sensing_cost_function,
        max_sample_dist=5.0,
    ):
        super().__init__(start, goal, num_nodes, map_array, max_sample_dist)
        self.sensors = list(sensors)
        self.obs_func = observation_function
        self.sensing_cost_func = sensing_cost_function
        self.X = None  # 状态空间
        self.belief_states = set()
        self.belief_map = {}  # 当前状态到belief state映射
        self.actions = {}  # 状态转移动作集
        self.sensing_actions = []  # Sens = 2^S

    # ----------------- Observation -----------------
    def observation(self, x, s):
        return self.obs_func(x, s)

    def observation_theta(self, x, theta):
        if not theta:
            return set(self.X)
        obs_sets = [self.observation(x, s) for s in theta]
        return set.intersection(*obs_sets)

    def belief_update(self, belief, x_true, theta):
        obs = self.observation_theta(x_true, theta)
        new_belief = belief.copy()
        new_belief.intersect(obs)
        return new_belief

    def sensing_cost(self, x, theta):
        return sum(self.sensing_cost_func(x, s) for s in theta)

    # ----------------- Belief Initialization -----------------
    def initialize_belief(self):
        self.X = set(self.nodes)
        b0 = BeliefState(self.X)
        self.belief_states.add(frozenset(b0.states))
        for x in self.X:
            self.belief_map[x] = b0.copy()
        return b0

    def all_sensing_actions(self):
        actions = []
        for r in range(len(self.sensors) + 1):
            for comb in itertools.combinations(self.sensors, r):
                actions.append(frozenset(comb))
        self.sensing_actions = actions
        return actions

    # ----------------- 自动生成动作集 -----------------
    def generate_state_actions(self):
        """
        自动生成状态转移动作集，动作集根据 PRM 图中邻居定义
        actions[state] = list of neighbor states
        """
        self.actions = {}
        for s in self.nodes:
            self.actions[s] = list(self.graph.get(s, []))
        return self.actions

    # ----------------- 自动生成 belief state -----------------
    def generate_belief_states(self):
        """
        对每个状态 x，生成对应的 belief state (初始为全X)
        并生成所有可能的 belief states
        """
        self.initialize_belief()
        self.all_sensing_actions()
        new_beliefs = set()
        for x in self.X:
            b0 = self.belief_map[x].copy()
            # 遍历所有 sensing actions
            for theta in self.sensing_actions:
                b_next = self.belief_update(b0, x, theta)
                if b_next.states:
                    new_beliefs.add(frozenset(b_next.states))
        self.belief_states = new_beliefs
        return self.belief_states

    # ----------------- 打印状态空间信息 -----------------
    def print_summary(self):
        print(f"Number of nodes: {len(self.nodes)}")
        print(f"Number of sensors: {len(self.sensors)}")
        print(f"Number of sensing actions: {len(self.sensing_actions)}")
        print(f"Number of states: {len(self.X)}")
        print(f"Number of belief states: {len(self.belief_states)}")
