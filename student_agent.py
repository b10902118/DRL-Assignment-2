# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(
            new_row, (0, self.size - len(new_row)), mode="constant"
        )  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        afterstate = self.board.copy()

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board.copy(), self.score, done, afterstate

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black"
                )
                ax.add_patch(rect)

                if value != 0:
                    ax.text(
                        j,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                        color=text_color,
                    )
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode="constant")
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode="constant")
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)


WEIGHT_SIZE = 0xFFFFFF + 1
FN = 64
# fmt: off
patterns = [
    [0, 1, 2, 4, 5, 6],    [12, 8, 4, 13, 9, 5],   [15, 14, 13, 11, 10, 9],  [3, 7, 11, 2, 6, 10],
    [3, 2, 1, 7, 6, 5],    [15, 11, 7, 14, 10, 6], [12, 13, 14, 8, 9, 10],   [0, 4, 8, 1, 5, 9],
    [1, 2, 5, 6, 9, 13],   [8, 4, 9, 5, 10, 11],   [14, 13, 10, 9, 6, 2],    [7, 11, 6, 10, 5, 4],
    [2, 1, 6, 5, 10, 14],  [11, 7, 10, 6, 9, 8],   [13, 14, 9, 10, 5, 1],    [4, 8, 5, 9, 6, 7],
    [0, 1, 2, 3, 4, 5],    [12, 8, 4, 0, 13, 9],   [15, 14, 13, 12, 11, 10], [3, 7, 11, 15, 2, 6],
    [3, 2, 1, 0, 7, 6],    [15, 11, 7, 3, 14, 10], [12, 13, 14, 15, 8, 9],   [0, 4, 8, 12, 1, 5],
    [0, 1, 5, 6, 7, 10],   [12, 8, 9, 5, 1, 6],    [15, 14, 10, 9, 8, 5],    [3, 7, 6, 10, 14, 9],
    [3, 2, 6, 5, 4, 9],    [15, 11, 10, 6, 2, 5],  [12, 13, 9, 10, 11, 6],   [0, 4, 5, 9, 13, 10],
    [0, 1, 2, 5, 9, 10],   [12, 8, 4, 9, 10, 6],   [15, 14, 13, 10, 6, 5],   [3, 7, 11, 6, 5, 9],
    [3, 2, 1, 6, 10, 9],   [15, 11, 7, 10, 9, 5],  [12, 13, 14, 9, 5, 6],    [0, 4, 8, 5, 6, 10],
    [0, 1, 5, 9, 13, 14],  [12, 8, 9, 10, 11, 7],  [15, 14, 10, 6, 2, 1],    [3, 7, 6, 5, 4, 8],
    [3, 2, 6, 10, 14, 13], [15, 11, 10, 9, 8, 4],  [12, 13, 9, 5, 1, 2],     [0, 4, 5, 6, 7, 11],
    [0, 1, 5, 8, 9, 13],   [12, 8, 9, 14, 10, 11], [15, 14, 10, 7, 6, 2],    [3, 7, 6, 1, 5, 4],
    [3, 2, 6, 11, 10, 14], [15, 11, 10, 13, 9, 8], [12, 13, 9, 4, 5, 1],     [0, 4, 5, 2, 6, 7],
    [0, 1, 2, 4, 6, 10],   [12, 8, 4, 13, 5, 6],   [15, 14, 13, 11, 9, 5],   [3, 7, 11, 2, 10, 9],
    [3, 2, 1, 7, 5, 9],    [15, 11, 7, 14, 6, 5],  [12, 13, 14, 8, 10, 6],   [0, 4, 8, 1, 9, 10],
]
# fmt: on

weights = []

LOG2 = {2**i: i for i in range(1, 16)}
LOG2[0] = 0

with open("weight.bin", "rb") as f:
    data = f.read()
    for i in range(FN):
        weights.append(
            np.frombuffer(
                data[i * (4 * WEIGHT_SIZE) : (i + 1) * (4 * WEIGHT_SIZE)], np.float32
            )
        )
        assert len(weights[i]) == WEIGHT_SIZE, f"Weight {i} size mismatch."


def get_feature(board, pattern):
    feature = 0
    for order in pattern:
        feature <<= 4
        feature |= LOG2[board[order // 4, order % 4]]
    return feature


def value(state):
    value = 0
    for weight, pattern in zip(weights, patterns):
        feature = get_feature(state, pattern)
        assert feature < WEIGHT_SIZE
        value += weight[feature]
    return value


def get_action(state, score):
    # print(state)
    # print(score)
    best_action = None
    best_value = -float("inf")
    env = Game2048Env()
    env.board = state
    # legal_actions = [a for a in range(4) if env.is_move_legal(a)]
    # print(legal_actions)
    for action in range(4):
        env.board = state.copy()
        if env.is_move_legal(action):
            _, reward, _, afterstate = env.step(action)
            v = reward + value(afterstate)
            if v > best_value:
                best_value = v
                best_action = action
    # assert best_action != None
    # print(best_action)
    return best_action
