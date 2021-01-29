import numpy as np


class MDPSolver:
    def __init__(self, env, alpha=0.9, beta=0.0001, iterations = 10000):
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.final_values = np.tile(0.0, len(self.env.S))
        self.policy = np.zeros(len(self.env.S))

    def value_iteration(self):
        values = np.tile(1.0, len(self.env.S))
        i = 0
        while np.sum(np.subtract(values, self.final_values)) > self.beta and i < self.iterations:
            for state in range(0, len(self.env.S)):
                maxQ = -1
                for action in self.env.A:
                    q = self.Q(state, action)
                    if q > maxQ:
                        maxQ = q
                        self.policy[state] = action
                values[state] = maxQ
            self.final_values = values
            self.final_values[5] = 0.0
            values = np.tile(1.0, len(self.env.S))
            i += 1

    def print_values(self):
        print("-----------------")
        for i in range(0, len(self.final_values)):
            print("| " + str(self.final_values[i]), end=" ")
            if ((i + 1) % self.env.cols) == 0:
                print("|\n-----------------")

    def print_policy(self):
        print("-----------------")
        for i in range(0, len(self.policy)):
            print("| " + str(self.env.get_action_str(self.policy[i])), end=" ")
            if ((i + 1) % self.env.cols) == 0:
                print("|\n-----------------")

    def Q(self, state, action):
        return self.env.R[state][action] + self.alpha * np.sum(np.multiply(self.env.T[action][state], self.final_values))
