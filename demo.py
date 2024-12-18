import numpy as np
import random
import matplotlib.pyplot as plt

class QLearningRobot:
    def __init__(self, n_states=10, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Q学習を用いた1次元ロボットのシミュレータ
        :param n_states: 状態数（1次元数直線の区間数）
        :param alpha: 学習率
        :param gamma: 割引率
        :param epsilon: ε-greedy法での探索率
        """
        self.n_states = n_states  # 状態数（数直線の長さ）
        self.q_table = np.zeros((n_states, 2))  # Qテーブル [状態][行動]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = [0, 1]  # 0: 左, 1: 右

    def choose_action(self, state):
        """ε-greedy法で行動を選択"""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        """Qテーブルの更新"""
        next_max = np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (reward + self.gamma * next_max - self.q_table[state, action])

    def train(self, episodes=500):
        """Q学習のトレーニング"""
        goal = self.n_states - 1  # 目標位置
        rewards = []

        for episode in range(episodes):
            state = 0  # 開始位置
            total_reward = 0

            while state != goal:
                action = self.choose_action(state)
                next_state = state + 1 if action == 1 else state - 1
                next_state = max(0, min(self.n_states - 1, next_state))

                # 報酬の設定
                reward = 1 if next_state == goal else -0.01
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            rewards.append(total_reward)

        print("Training Completed.")
        return rewards

    def show_q_table(self):
        """Qテーブルの表示"""
        print("\nQ-Table:")
        print(self.q_table)

    def plot_rewards(self, rewards):
        """エピソードごとの報酬のプロット"""
        plt.plot(rewards)
        plt.title("Q-Learning Rewards Over Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.show()

if __name__ == "__main__":
    robot = QLearningRobot(n_states=10)
    rewards = robot.train(episodes=500)
    robot.show_q_table()
    robot.plot_rewards(rewards)
