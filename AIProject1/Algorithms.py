import numpy as np
from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

from types import TracebackType


class BFSAgent:
    class BFSNode:
        def __init__(self, state, prev=None, actions=[], cost=0, terminated=False):
            self.state = state
            self.prev = prev
            self.cost = cost
            self.terminated = terminated
            self.actions = actions

    def __init__(self):
        self.env = None

    def search(self, env: DragonBallEnv) -> Tuple[List[int], int, int]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()
        cexpanded = 0
        node = self.BFSNode(state)
        opened = [node]
        close = set()
        if self.env.is_final_state(state): return ([], 0, 0)

        while opened:
            cexpanded += 1
            node = opened.pop(0)
            if (node.terminated):
                close.add(node.state)
                continue
            env.set_state(node.state)
            close.add(node.state)
            for action in range(4):
                env.set_state_2(node.state)
                new_state, cost, terminated = self.env.step(action)
                new_cost = np.inf if cost == np.inf else node.cost + int(cost)
                child = self.BFSNode(new_state, node, node.actions + [action], new_cost, terminated)
                if self.env.is_final_state(child.state):
                    return (child.actions, child.cost, cexpanded)
                found = False
                for s in opened:
                    if (s.state == new_state): found = True
                if (not child.state in close) and (not found):
                    opened.append(child)
        return ([], 0, 0)


class Node:
    def __init__(self, state, prev=None, actions=[], cost=0, h=0):
        self.state = state
        self.f = 0
        self.prev = prev
        self.h = h
        self.cost = cost
        self.actionsList = actions


    @staticmethod
    def manhattan_calculator(n1: tuple, n2: tuple, env):
        row1, col1 = env.to_row_col(n1)
        row2, col2 = env.to_row_col(n2)
        return abs(row1 - row2) + abs(col1 - col2)

    @staticmethod
    def calc_heuristic_val(state, env: DragonBallEnv):

        if not state[1] and not state[2]:
            d1_manhattan = Node.manhattan_calculator(state, env.d1, env)
            d2_manhattan = Node.manhattan_calculator(state, env.d2, env)
            return min(d1_manhattan, d2_manhattan)

        elif not state[2]:
            d2_manhattan = Node.manhattan_calculator(state, env.d2, env)
            return d2_manhattan

        elif not state[2]:
            d1_manhattan = Node.manhattan_calculator(state, env.d1, env)
            return d1_manhattan

        else:
            min_goal = Node.manhattan_calculator(state, env.get_goal_states()[0], env)
            for goal in env.get_goal_states():
                if Node.manhattan_calculator(state, goal, env) < min_goal:
                    min_goal = Node.manhattan_calculator(state, goal, env)

            return min_goal


class WeightedAStarAgent:

    def __init__(self):
        self.env = None
        self.h_weight = 0.5

    @staticmethod
    def get_successors(current_node, env, h_weight=0.5):
        successors = []
        if current_node.state[0] == env.ncol * env.nrow - 1:  # TODO: check if it is a goal in a better way
            return successors

        for action, (child_state, child_cost, _) in env.succ(current_node.state).items():
            env.set_state_2(current_node.state)

            if child_state is None:
                continue

            state, cost, _ = env.step(action)
            if state == current_node.state:
                continue

            successor_cost = current_node.cost + cost
            successor = Node(state=state,
                             prev=current_node,
                             h=Node.calc_heuristic_val(state, env),
                             cost=successor_cost)
            successor.f = ((1 - h_weight) * successor_cost) + (h_weight * successor.h)
            successor.actionsList = current_node.actionsList + []
            successor.actionsList.append(action)

            successors.append(successor)

        return successors

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.h_weight = h_weight
        open = heapdict.heapdict()
        close = set()
        expanded = 0

        start = Node(state=env.get_initial_state(), prev=None, h=0, cost=0)
        start.f = Node.calc_heuristic_val(start.state, env)

        if env.is_final_state(start.state):
            return ([], 0, 0)
        goals = [goal_state[0] for goal_state in env.goals]
        open[start] = (start.f, start.state[0])

        while open:
            expanded += 1
            current_node, _ = open.popitem()
            close.add(current_node)

            if env.is_final_state(current_node.state):
                return (current_node.actionsList, current_node.cost, expanded)

            for child in self.get_successors(current_node, self.env, h_weight):
                if child.state[0] in goals and \
                    (child.state[1] == False or child.state[2] == False):
                    continue

                found_in_open = found_in_close = False
                element_in_close = None
                element_in_open = None
                for key in open:
                    if key.state == child.state:
                        found_in_open = True
                        element_in_open = key

                for key in close:
                    if key.state == child.state:
                        found_in_close = True
                        element_in_close = key

                if not found_in_close and not found_in_open:
                    open[child] = (child.f, child.state[0])

                elif found_in_open:
                        if child.f < open[element_in_open][0]:
                            del open[element_in_open]
                            open[child] = (child.f, child.state[0])

                elif found_in_close:
                        close.remove(element_in_close)
                        if child.f < element_in_close.f:
                            open[child] = (child.f, child.state[0])
                        else:
                            close.add(element_in_close)

        return ([], 0, 0)


class AStarEpsilonAgent:

    def __init__(self):
        self.env = None

    @staticmethod
    def popFromFocal(nodes, epsilon):
        min_f = min(nodes, key=lambda x: x.f)
        focal = [i for i in nodes if i.f <= (min_f.f * (1 + epsilon))]
        min_g = nodes[0].cost
        for node in nodes:
            if node.cost < min_g:
                min_g = node.cost
        temporary_heapdict = heapdict.heapdict()
        for f in focal:
            temporary_heapdict[f] = (f.cost, f.state)
        (del_node, priority) = temporary_heapdict.popitem()
        nodes.remove(del_node)
        return del_node

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        open = []
        close = set()
        start = Node(state=env.get_initial_state(), prev=None, h=0, cost=0)
        start.f = Node.calc_heuristic_val(start.state, env)

        if env.is_final_state(start.state):
            return ([], 0, 0)

        goals = [goal_state[0] for goal_state in env.goals]
        open.append(start)
        expanded = 0

        while len(open) != 0:
            current_node = AStarEpsilonAgent.popFromFocal(open, epsilon)
            close.add(current_node)
            expanded += 1
            if env.is_final_state(current_node.state):
                return (current_node.actionsList, current_node.cost, expanded)

            for child in WeightedAStarAgent.get_successors(current_node, env):
                if child.state[0] in goals and \
                    (child.state[1] == False or child.state[2] == False):
                    continue

                found_in_open = found_in_close = False
                index_in_open = -1
                element_in_close = None

                for index, element in enumerate(open):
                    if element.state == child.state:
                        found_in_open = True
                        index_in_open = index

                for element in close:
                    if element.state == child.state:
                        found_in_close = True
                        element_in_close = element

                if not found_in_close and not found_in_open:
                    open.append(child)

                elif found_in_open:
                    if child.f < open[index_in_open].f:
                        open.pop(index_in_open)
                        open.append(child)

                elif found_in_close:
                    close.remove(element_in_close)
                    if child.f < element_in_close.f:
                        open.append(child)
                    else:
                        close.add(element_in_close)

        return ([], 0, 0)
