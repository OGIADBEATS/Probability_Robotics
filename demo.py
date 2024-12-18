import numpy as np
import random

# パラメータ設定
LINE_LENGTH = 10  # 数直線の長さ (0 ~ LINE_LENGTH-1)
START_POSITION = 0
GOAL_POSITION = LINE_LENGTH - 1  # ゴールの位置
ACTIONS = ["left", "right"]  # 行動

ALPHA = 0.1  # 学習率
GAMMA = 0.9  # 割引率
EPSILON = 0.2  # 探索率
EPISODES = 1000  # 学習エピソード数

# Q値の初期化
q_table = {pos: {action: 0 for action in ACTIONS} for pos in range(LINE_LENGTH)}

def take_action(position, action):
    """次の位置と報酬を返す"""
    if action == "left":
        next_position = max(0, position - 1)
    elif action == "right":
        next_position = min(LINE_LENGTH - 1, position + 1)
    else:
        raise ValueError("Invalid action")

    # ゴールに到達したら報酬+1、それ以外は-0.1
    reward = 1 if next_position == GOAL_POSITION else -0.1
    return next_position, reward

def choose_action(position):
    """ε-greedy で行動を選択"""
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)
    else:
        return max(q_table[position], key=q_table[position].get)

# 学習プロセス
for episode in range(EPISODES):
    position = START_POSITION
    total_reward = 0
    done = False

    while not done:
        if position == GOAL_POSITION:
            break  # ゴールに到達したら終了

        action = choose_action(position)
        next_position, reward = take_action(position, action)

        # Q値の更新
        if position != GOAL_POSITION:  # ゴール位置では更新しない
            best_next_action = max(q_table[next_position], key=q_table[next_position].get)
            q_table[position][action] += ALPHA * (
                reward + GAMMA * q_table[next_position][best_next_action] - q_table[position][action]
            )

        position = next_position
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# 最適政策の表示
def print_policy():
    """数直線上の最適政策を表示"""
    policy = []
    for position in range(LINE_LENGTH):
        if position == GOAL_POSITION:
            policy.append("G")  # ゴール
        elif position < GOAL_POSITION:
            best_action = max(q_table[position], key=q_table[position].get)
            policy.append(best_action[0].upper())
        else:
            policy.append(" ")  # ゴール以降は空白

    print("Optimal Policy:")
    print(" ".join(policy))

print_policy()
