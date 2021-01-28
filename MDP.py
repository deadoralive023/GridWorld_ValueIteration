import numpy as np
import math
import os

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

TERMINAL_STATE = 5
UNREACHABLE_STATE = 2
AGENT_STATE = 8
DEFAULT_STATE = 0
TERMINAL_STATES = [3, 7]
UNREACHABLE_STATES = [5]


class GridWorld:
    def __init__(self):
        self.rows = 3
        self.cols = 4
        self.S = self.create_state_space()
        self.A = [LEFT, UP, RIGHT, DOWN]
        self.R = self.create_reward_vec()
        self.T = self.create_transition_probabilities()
        self.current_state = AGENT_STATE

    def create_state_space(self):
        state_space = np.zeros([self.rows * self.cols], dtype=int)
        for state in TERMINAL_STATES:
            state_space[state] = TERMINAL_STATE
        for state in UNREACHABLE_STATES:
            state_space[state] = UNREACHABLE_STATE
        state_space[AGENT_STATE] = 1
        return state_space

    def create_reward_vec(self):
        reward_vec = np.zeros([self.rows * self.cols])
        reward_vec[3] = 1
        reward_vec[7] = -1
        return reward_vec

    def create_transition_probabilities(self):
        T = np.zeros((len(self.A), self.rows * self.cols, self.rows * self.cols))
        for action in range(0, len(self.A)):
            for state in range(0, len(self.S)):
                T[action][state][self.next_state(state, action)] += 0.8
                T[action][state][self.next_state(state, self.set_action_val(action - 1))] += 0.1
                T[action][state][self.next_state(state, self.set_action_val(action + 1))] += 0.1
        return T

    def next_state(self, state, action):
        x, y = self.get_index(state)
        if action is LEFT:
            next_state = self.map_index(max(0, x - 1), y)
        elif action is UP:
            next_state = self.map_index(x, max(0, y - 1))
        elif action is RIGHT:
            next_state = self.map_index(min(self.cols - 1, x + 1), y)
        elif action is DOWN:
            next_state = self.map_index(x, min(self.rows - 1, y + 1))
        return next_state if next_state not in UNREACHABLE_STATES else state

    def step(self, act):
        if act in self.legal_actions(self.current_state):
            reward = self.R[self.current_state]
            self.S[self.current_state] = DEFAULT_STATE
            self.S[self.next_state(self.current_state, act)] = 1
            self.current_state = self.next_state(self.current_state, act)
            return self, reward, self.is_done()
        return self, None, None

    def legal_actions(self, state):
        if state in TERMINAL_STATES:
            return [RIGHT]
        return self.A

    def is_done(self):
        if self.current_state in TERMINAL_STATES:
            return True
        return False

    def set_action_val(self, action):
        new_action = action
        if action == 4:
            new_action = 0
        elif action == -1:
            new_action = 3
        return new_action

    def get_index(self, state):
        x = state % self.cols
        y = math.floor(state / self.cols)
        return x, y

    def map_index(self, x, y):
        return y * self.cols + x

    def print(self):
        print("-----------------")
        for i in range(0, len(self.S)):
            print("| " + str(self.S[i]), end=" ")
            if ((i + 1) % self.cols) == 0:
                print("|\n-----------------")





env = GridWorld()
for i in range(0, 100):
    env.print()
    action = int(input())
    env, r, done = env.step(action)
    print(env)
env.print()
