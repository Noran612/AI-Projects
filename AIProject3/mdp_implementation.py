#NEW VERSION HERE 
from copy import deepcopy
import numpy as np
import copy
import numpy as np
import copy

# class MDP_PARAMS:
#     def __init__(self, num_row, num_col, gamma, board, actions, transition_function, terminal_states):
#         self.num_row = num_row
#         self.num_col = num_col
#         self.gamma = gamma
#         self.board = board
#         self.actions = actions
#         self.transition_function = transition_function
#         self.terminal_states = terminal_states
#
#     def step(self, state, action):
#         # Define the step function implementation
#         pass

class PolicyUpdater:
    @staticmethod
    def compute_max_expected_utility(mdp, row_index, col_index, utility_matrix, available_actions):
        state = (row_index, col_index)
        max_utility_value = float('-inf')

        for action in available_actions:
            expected_value = PolicyUpdater.compute_expected_value_for_action(mdp, state, action, utility_matrix)
            max_utility_value = max(max_utility_value, expected_value)

        return max_utility_value

    @staticmethod
    def compute_max_utility_for_state(mdp, row_index, col_index, utility_matrix, available_actions):
        state = (row_index, col_index)
        max_utility_value = float('-inf')

        for action in available_actions:
            expected_value = PolicyUpdater.compute_expected_value_for_action(mdp, state, action, utility_matrix)
            max_utility_value = max(max_utility_value, expected_value)

        return max_utility_value

    @staticmethod
    def compute_expected_value_for_action(mdp, state, action, utility_matrix):
        expected_value = 0
        for transition_index in range(len(mdp.transition_function[action])):
            next_state = mdp.step(state, list(mdp.transition_function)[transition_index])
            expected_value += mdp.transition_function[action][transition_index] * utility_matrix[next_state[0]][next_state[1]]
        return expected_value

    @staticmethod
    def calculate_transition_contribution(mdp, state, transition_index, action, utility_matrix):
        next_state = mdp.step(state, list(mdp.transition_function)[transition_index])
        return mdp.gamma * mdp.transition_function[action][transition_index] * utility_matrix[next_state[0]][next_state[1]]

    @staticmethod
    def calculate_total_contribution(mdp, state, action, utility_matrix):
        total_contribution = 0
        for transition_index in range(4):  # Assuming 4 possible transitions
            contribution = PolicyUpdater.calculate_transition_contribution(mdp, state, transition_index, action, utility_matrix)
            total_contribution += contribution
        return total_contribution


class MDPFunctions:
    def __init__(self, mdp):
        self.mdp = mdp

    def array_maker_id(self):
        rows = self.mdp.num_row
        cols = self.mdp.num_col
        all_states = rows * cols
        return np.identity(all_states)

    def get_states_num(self):
        cols = self.mdp.num_row
        rows = self.mdp.num_col
        all_states = cols * rows
        return all_states

    def array_maker(self, row, col):
        return np.zeros((row, col))

    def simulate_slow_service(self):
        for _ in range(7):
            continue

    def get_real_index(self, row, col):
        cols = self.mdp.num_col
        index = row * cols + col
        return index

    def get_next_state(self, state, action):
        cols = self.mdp.num_col
        ns = self.mdp.step(state, action)
        nsi = ns[0] * cols + ns[1]
        return ns, nsi

    def check_stop(self, policy, row, col):
        if policy[row][col] == 0 or self.is_wall(policy, row, col):
            return True

    def compute_transition_probabilities(self, policy):
        actions = self.mdp.actions
        num_of_states = self.get_states_num()
        transition_probs = self.array_maker(num_of_states, num_of_states)

        for row in range(self.mdp.num_row):
            for col in range(self.mdp.num_col):
                index = self.get_real_index(row, col)
                if self.check_stop(policy, row, col):
                    for ns in range(num_of_states):
                        transition_probs[index][ns] = 0
                    continue

                action = policy[row][col]
                transition_function = self.mdp.transition_function[action]
                for a, i in zip(actions, range(4)):
                    pair = (row, col)
                    next_state, index_ns = self.get_next_state(pair, a)
                    curr_p = transition_probs[index][index_ns]
                    new_p = curr_p + transition_function[i]
                    transition_probs[index][index_ns] = new_p

        return transition_probs

    def make_PBs(self, policy):
        gamma = self.mdp.gamma
        transition_probs = self.compute_transition_probabilities(policy)

        #print(f"transition probs are: {transition_probs}")
        num_of_states = self.get_states_num()

        array_pb = self.array_maker(num_of_states, num_of_states)
        # self.simulate_slow_service()
         
        for index in range(num_of_states):
            for ns in range(num_of_states):
                if transition_probs[index][ns]==0:
                    #print("im here")
                    continue
                array_pb[index][ns] = gamma * transition_probs[index][ns]
        # print(f"array probs is: {array_pb}")
        return array_pb

    def make_reward_board(self):
        num_rows = self.mdp.num_row
        num_cols = self.mdp.num_col
        rewards_board = self.array_maker(num_rows, num_cols)
        # self.simulate_slow_service()

        for row in range(num_rows):
            for col in range(num_cols):
                if not self.is_wall(self.mdp.board, row, col):
                    rewards_board[row][col] = self.mdp.board[row][col]
        return rewards_board

    def calculate_action_value(self, utility_matrix, row_index, col_index, action):
        actions = self.mdp.actions
        next_state_values = [
            utility_matrix[self.mdp.step((row_index, col_index), next_action)[0]][self.mdp.step((row_index, col_index), next_action)[1]]
            for next_action in actions
        ]

        transition_probabilities = self.mdp.transition_function[action]
        action_value = sum(transition_probabilities[idx] * next_state_values[idx] for idx in range(len(actions)))

        return action_value

    def is_wall(self, policy_matrix, row_index, col_index):
        return policy_matrix[row_index][col_index] in ['WALL', None]

    def update_policy(self, policy, utility_matrix):
        updated_policy = [row[:] for row in policy]

        for row in range(self.mdp.num_row):
            for col in range(self.mdp.num_col):
                if policy[row][col] == 0 or self.is_wall(policy, row, col):
                    continue

                max_action_value = float('-inf')
                for action in self.mdp.actions:
                    action_value = self.calculate_action_value(utility_matrix, row, col, action)
                    if action_value > max_action_value:
                        updated_policy[row][col] = action
                        max_action_value = action_value

        return updated_policy

    def update_policy2(self, policy, utility_matrix):
        updated_policy2 = [[[None for _ in self.mdp.actions] for _ in range(self.mdp.num_col)] for _ in
                          range(self.mdp.num_row)]
        updated_policy = [row[:] for row in policy]
        for row in range(self.mdp.num_row):
            for col in range(self.mdp.num_col):
                if policy[row][col] == 0 or self.is_wall(policy, row, col):
                    continue

                max_action_value = float('-inf')
                for i, action in enumerate(self.mdp.actions):
                    action_value = self.calculate_action_value(utility_matrix, row, col, action)
                    updated_policy2[row][col][i] = action_value
                    if action_value > max_action_value:
                        updated_policy[row][col] = action
                        max_action_value = action_value
        return updated_policy, updated_policy2
# import string
# from copy import deepcopy
# import random
import copy
import numpy as np
import matplotlib.pyplot as plt




def value_iteration(mdp, U_init, epsilon=1e-3):
    actions = mdp.actions
    num_rows = mdp.num_row
    num_cols = mdp.num_col
    gamma = mdp.gamma

    updated_utility_matrix = copy.deepcopy(U_init)

    difference = float('inf')  # Initialize difference to a large value
    while difference >= epsilon * (1 - gamma) / gamma:
        utility_matrix_in_loop = copy.deepcopy(updated_utility_matrix)
        difference = 0  # Reset difference for each iteration

        for row in range(num_rows):
            for col in range(num_cols):
                if mdp.board[row][col] != 'WALL' or mdp.board[row][col]==1  or mdp.board[row][col]==-1 or mdp.board[row][col]==None:
                  
                    max_expected_utility = PolicyUpdater.compute_max_expected_utility(mdp, row, col, utility_matrix_in_loop, actions)
                    updated_utility_matrix[row][col] = float(mdp.board[row][col]) + gamma * max_expected_utility

        for row, col in mdp.terminal_states:
            updated_utility_matrix[row][col] = float(mdp.board[row][col])

        difference = np.linalg.norm(np.array(updated_utility_matrix) - np.array(utility_matrix_in_loop), np.inf)

    return updated_utility_matrix

def get_policy(mdp, U):
    actions = mdp.actions
    num_rows = mdp.num_row
    num_cols = mdp.num_col
    policy_matrix = [copy.deepcopy(['UP'] * mdp.num_col) for _ in range(num_rows)]

    for row in range(num_rows):
        for col in range(num_cols):
            current_state = (row, col)
            max_utility = float('-inf')

            for action in actions:
                total_contribution = PolicyUpdater.calculate_total_contribution(mdp, current_state, action, U)

                if total_contribution > max_utility:
                    max_utility = total_contribution
                    policy_matrix[row][col] = action

    return policy_matrix


def policy_evaluation(mdp, policy):

    mdp_functions = MDPFunctions(mdp)
    num_rows = mdp.num_row
    num_cols = mdp.num_col
    gamma = mdp.gamma

    identity_matrix = np.identity(num_rows * num_cols)
    transition_probabilities = mdp_functions.make_PBs(policy)
    # print(f"policy evaluation: transition prob: {transition_probabilities}")
    bet_matrix = identity_matrix - gamma * transition_probabilities
    bet_inverse = np.linalg.inv(bet_matrix)
    # print(f"policy evaluation: bet_inverse prob: {transition_probabilities}")
    reward_matrix = np.reshape(mdp_functions.make_reward_board(), (num_rows * num_cols, 1)).astype('float64')

    utility_matrix = np.reshape(np.matmul(bet_inverse, reward_matrix), (num_rows, num_cols))
    # print(f"policy evaluation: utility matrix: {utility_matrix}")
    return utility_matrix



def policy_iteration(mdp, policy_init):

    updated_policy = copy.deepcopy(policy_init)
    mdp_functions = MDPFunctions(mdp)
    while True:
        utility_matrix = policy_evaluation(mdp, updated_policy)
        new_policy = mdp_functions.update_policy(updated_policy, utility_matrix)

        if new_policy == updated_policy:
            break

        updated_policy = new_policy

    return updated_policy

"""For this functions, you can import what ever you want """

def get_all_policies(mdp, U, policy_init, epsilon=10**(-0.5)):

    updated_policy = copy.deepcopy(policy_init)
    mdp_functions = MDPFunctions(mdp)
    permutations = 1
    dictt = {0:'U',
             1: 'D',
             2: 'R',
             3: 'L'}
    while True:
        utility_matrix = policy_evaluation(mdp, updated_policy)
        new_policy, new_policy_updated = mdp_functions.update_policy2(updated_policy, utility_matrix)
        if new_policy == updated_policy:
            break
        updated_policy = new_policy
    # all_policies = np.zeros(mdp.num_row, mdp.num_col)
    rows = mdp.num_row
    cols = mdp.num_col
    default_value = 0
    all_policies = [[[] for _ in range(cols)] for _ in range(rows)]
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            max_val = float('-inf')
            index = -1
            for action in range(4):
                if new_policy_updated[row][col][action] == None:
                    continue
                if new_policy_updated[row][col][action] > max_val:
                    max_val = new_policy_updated[row][col][action]
                    index = action
            if index == -1:
                continue
            all_policies[row][col].append(dictt[index])
            for action in range(4):
                if abs(new_policy_updated[row][col][action] - max_val) < epsilon:
                    all_policies[row][col].append(dictt[action])
    for row in range(len(all_policies)):
        for col in range(len(all_policies[0])):
            all_policies[row][col] = set(all_policies[row][col])
            p = len(all_policies[row][col]) if len(all_policies[row][col]) >0 else 1
            permutations *= p
    print_policy(all_policies)
    print(f"permutation is: {permutations}")
    return permutations


def print_policy(policy):
    res = ""
    for r in range(len(policy)):
        res += "|"
        for c in range(len(policy[0])):
            if len(policy[r][c]) == 0:
                val = None
            else:
                val = policy[r][c]
            if val != None:
                for i in val:
                    res += i
            else:
                res += 'None'
            res += '|'



        res += "\n"
    print(res)
    #visualize_all_policies(all_policies, len(all_policies), len(all_policies[0]))
    #return all_policies
# import matplotlib.pyplot as plt
# import numpy as np
#
# def visualize_2d_list_sets(matrix):
#     if not matrix:
#         print("Empty matrix provided.")
#         return
#
#     # Create a color map for unique elements in the sets
#     unique_elements = set()
#     for row in matrix:
#         for element_set in row:
#             unique_elements.update(element_set)
#     if not unique_elements:
#         print("No unique elements found in the sets.")
#         return
#
#     colormap = plt.cm.get_cmap('tab10', len(unique_elements))
#     color_dict = {element: np.array(colormap(i)) for i, element in enumerate(unique_elements)}
#
#     # Initialize colors_matrix with zeros
#     colors_matrix = np.zeros((len(matrix), len(matrix[0]), 3))
#
#     # Convert sets to colors
#     for i in range(len(matrix)):
#         for j in range(len(matrix[0])):
#             for element in matrix[i][j]:
#                 colors_matrix[i, j] += np.array(color_dict[element])
#
#     # Plot the colors
#     plt.imshow(colors_matrix)
#     plt.colorbar()
#     plt.show()

def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    pass
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

