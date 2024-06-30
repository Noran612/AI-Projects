from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import threading
import numpy as np

import datetime



def smart_heuristic(env: WarehouseEnv, robot_id: int):
    current_robot = env.get_robot(robot_id)

    heuristic_value = current_robot.battery + current_robot.credit

    if current_robot.package is not None:
        battery_importance = 1
        distance_to_destination = manhattan_distance(current_robot.position, current_robot.package.destination)

        battery_importance *= 1 / (distance_to_destination or 0.3)  # Avoid division by zero

        heuristic_value += battery_importance

    else:
        battery_importance = 1

        distances_to_packages = [
            manhattan_distance(current_robot.position, package.position)
            for package in env.packages
            if package.on_board
        ]

        distance_to_nearest_package = min(distances_to_packages) if distances_to_packages else 0

        battery_importance *= 1 / (distance_to_nearest_package or 0.5)

        heuristic_value += battery_importance

    return heuristic_value


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    class TimeoutError(Exception):
        pass

    def __init__(self):
        self.minimax_tree = None
        self.tree_depth = 0
        self.timed_out = False

    class TreeNode:
        def __init__(self, env: WarehouseEnv):
            self.env = env
            self.children = []
            self.op = None
            self.value = None
            self.best_next = None



    def handle_timeout(self):
        self.timed_out = True

    def build_game_tree(self, env: WarehouseEnv, depth, agent_id, heuristic_func, level):
        current_player = agent_id if level % 2 == 0 else 1 - agent_id

        if depth == 0 or env.done():
            terminal_node = self.TreeNode(env)
            terminal_node.value = heuristic_func(env, current_player)
            return terminal_node

        child_nodes = []
        root = self.TreeNode(env)
        value = -float('inf') if current_player == agent_id else float('inf')

        for operator in env.get_legal_operators(current_player):
            if self.timed_out:
                raise self.TimeoutError("Timed out")
            cloned_env = env.clone()
            cloned_env.apply_operator(current_player, operator)
            child_node = self.build_game_tree(cloned_env, depth - 1, agent_id, heuristic_func, level + 1)
            child_node.op = operator
            child_nodes.append(child_node)

            if current_player == agent_id:
                value = max(value, child_node.value)
            else:
                value = min(value, child_node.value)

            if child_node.value == value:
                root.best_next = operator

        root.value = value
        root.children = child_nodes
        return root

    def construct_minimax_tree(self, env: WarehouseEnv, agent_id, depth, heuristic_func):
        minimax_tree = self.build_game_tree(env, depth, agent_id, heuristic_func, 0)
        self.minimax_tree = minimax_tree
        self.tree_depth = depth
        self.best_next_move = minimax_tree.best_next

    def run_step_aux(self, env: WarehouseEnv, agent_id):
        legal_operators = env.get_legal_operators(agent_id)
        self.best_next_move = legal_operators[0] if legal_operators else None
        cloned_env = env.clone()
        try:
            depth = 2
            while not self.timed_out:
                self.construct_minimax_tree(cloned_env, agent_id, depth, smart_heuristic)
                depth += 1
        except:
            return

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.timed_out = False
        timer = threading.Timer(0.8 * time_limit, self.handle_timeout)
        timer.start()
        self.run_step_aux(env, agent_id)
        return self.best_next_move

class AgentAlphaBeta(Agent):
    MIN_VALUE = float('inf')
    MAX_VALUE = float('-inf')

    def __init__(self):
        self.current_agent_id = None
        self.opponent_agent_id = None
        self.end_time = None

    def set_current_agent_id(self, agent_id):
        self.current_agent_id = agent_id

    def set_opponent_agent_id(self, agent_id):
        self.opponent_agent_id = agent_id


    def get_opponent_robot(self, env):
        return env.get_robot(self.opponent_agent_id)

    def get_current_robot(self, env):
        return env.get_robot(self.current_agent_id)

    def steps_bound_achieved(self, env: WarehouseEnv):
        return env.num_steps == 0

    def evaluate_state(self, env: WarehouseEnv, agent_id):
        if self.is_terminal_state(env):
            return self.calculate_terminal_state_value(env, agent_id)

    def is_time_over(self):
        return datetime.datetime.now() >= self.end_time

    def calculate_terminal_state_value(self, env: WarehouseEnv, agent_id):
        if self.is_time_over() or self.steps_bound_achieved(env):
            return smart_heuristic(env, agent_id)
        elif self.current_agent_won(agent_id, env):
            return self.get_current_robot(env).credit - self.get_opponent_robot(env).credit
        elif self.opponent_agent_won(agent_id, env):
            return self.get_current_robot(env).credit - self.get_opponent_robot(env).credit

    def opponent_agent_won(self, agent_id, env: WarehouseEnv):
        opponent_robot = self.get_opponent_robot(env)
        return opponent_robot.credit > self.get_current_robot(env).credit

    def current_agent_won(self, agent_id, env: WarehouseEnv):
        current_robot = self.get_current_robot(env)
        return current_robot.credit >= self.get_opponent_robot(env).credit

    def is_current_agent_turn(self, agent_id):
        return agent_id == self.current_agent_id

    def is_opponent_agent_turn(self, agent_id):
        return agent_id == self.opponent_agent_id

    def get_another_agent_id(self, agent_id):
        return (agent_id + 1) % 2

    def is_terminal_state(self, env: WarehouseEnv):
        return env.done() or self.is_time_over()

    def get_best_children_result(self, agent_id, env, mode="MAX", depth=0):
        operators, children = self.successors(env, agent_id)
        children_results = [
            self.alpha_beta_pruning(child_env, self.get_another_agent_id(agent_id), depth)
            for child_env in children
        ]
        updated_children_results = [(result[0], operator) for operator, result in zip(operators + [None], children_results)]
        if mode == "MAX":
            best_child_result = max(updated_children_results, key=lambda x: x[0])
        elif mode == "MIN":
            best_child_result = min(updated_children_results, key=lambda x: -x[0])
        return best_child_result

    def alpha_beta_pruning(self, env: WarehouseEnv, agent_id, depth=0):
        if self.is_terminal_state(env):
            return (self.evaluate_state(env, agent_id), None)
        if self.is_current_agent_turn(agent_id):
            return self.get_best_children_result(agent_id, env, mode="MAX", depth=depth + 1)
        if self.is_opponent_agent_turn(agent_id):
            return self.get_best_children_result(agent_id, env, mode="MIN", depth=depth + 1)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.end_time = datetime.datetime.now() + datetime.timedelta(seconds=time_limit * 0.95)
        self.set_current_agent_id(agent_id)
        self.set_opponent_agent_id(self.get_another_agent_id(agent_id))
        result = self.alpha_beta_pruning(env, agent_id)
        return result[1]


class AgentExpectimax(Agent):
    class TimeoutError(Exception):
        pass

    def timeout_handler(self):
        self.timed_out = True

    class TreeNode:

        def __init__(self, env: WarehouseEnv):
            self.env = env
            self.children = (
                []
            )
            self.value = None
            self.best_next = (
                None
            )

    def next_operators_to_package(self, env: WarehouseEnv, robot_id):
        legal_operators = env.get_legal_operators(robot_id)
        return [operator for operator in legal_operators if operator in {'move east', 'pick up'}]

    def get_regular_probability(self, legal_operators, next_to_package_right):
        regular_operators = [op for op in legal_operators if op not in next_to_package_right]
        count_regular = len(regular_operators)
        return 1 / (count_regular + 1) if count_regular > 0 else 1

    def create_probability_dict(self, legal_operators, next_to_package_right, regular_probability):
        package_probability = 2 * regular_probability
        probability_dict = {op: package_probability if op in next_to_package_right else regular_probability for op in
                            legal_operators}
        return probability_dict

    def get_operator_probability(self, env: WarehouseEnv, robot_id):
        legal_operators = env.get_legal_operators(robot_id)
        next_to_package_operators = self.next_operators_to_package(env, robot_id)
        return self.create_probability_dict(legal_operators, next_to_package_operators, 1 / len(legal_operators))

    def __init__(self):
        self.expectimax_tree = None
        self.tree_depth = 0
        self.timed_out = False

    def evaluate_terminal_state(self, env: WarehouseEnv, player, heuristic_func):
        term = self.TreeNode(env)
        term.value = heuristic_func(env, player)
        return term

    def expand_game_tree(self, env: WarehouseEnv, depth, agent_id, heuristic_func, level):
        player = agent_id if level % 2 == 0 else 1 - agent_id
        root = self.TreeNode(env)
        value = -np.inf if player == agent_id else 0
        best_next_move = None
        child_states = []

        for operator in env.get_legal_operators(player):
            if self.timed_out:
                raise self.TimeoutError("Timed out")
            new_env = env.clone()
            new_env.apply_operator(player, operator)
            child_val = self.build_game_tree(
                new_env, depth - 1, agent_id, heuristic_func, level + 1
            )
            child_states.append(child_val)
            if player == agent_id:
                value = max(value, child_val.value)
            else:
                value += self.get_operator_probability(env, 1 - agent_id)[operator] * child_val.value
            if child_val.value == value:
                best_next_move = operator

        root.value = value
        root.best_next = best_next_move
        root.children = child_states
        return root

    def build_game_tree(self, env: WarehouseEnv, depth, agent_id, heuristic_func, level):
        if depth == 0 or env.done():
            return self.evaluate_terminal_state(env, agent_id if level % 2 == 0 else 1 - agent_id, heuristic_func)
        return self.expand_game_tree(env, depth, agent_id, heuristic_func, level)

    def expectimax_with_depth(self, env: WarehouseEnv, agent_id, depth, heuristic_func):
        expectimax_tree = self.build_game_tree(env, depth, agent_id, heuristic_func, 0)
        self.expectimax_tree = expectimax_tree
        self.tree_depth = depth
        self.best_next_move = expectimax_tree.best_next

    def run_step_aux(self, env: WarehouseEnv, agent_id):
        legal_ops = env.get_legal_operators(agent_id)
        self.best_next_move = legal_ops[0] if len(legal_ops) > 0 else None
        cloned = env.clone()
        try:
            depth = 2
            while not self.timed_out:
                self.expectimax_with_depth(cloned, agent_id, depth, smart_heuristic)
                depth += 1
        except:
            return

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.timed_out = False
        t = threading.Timer(0.8 * time_limit, self.timeout_handler)
        t.start()
        self.run_step_aux(env, agent_id)
        return self.best_next_move


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = [
            "move north",
            "move east",
            "move north",
            "move north",
            "pick_up",
            "move east",
            "move east",
            "move south",
            "move south",
            "move south",
            "move south",
            "drop_off",
        ]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
