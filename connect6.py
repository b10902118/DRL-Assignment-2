import sys
import numpy as np
import random


import copy
import random
import math
import numpy as np
from tqdm import tqdm

EMPTY = 0
BLACK = 1
WHITE = 2

DRDC = [(dr, dc) for dr in range(-1, 2) for dc in range(-1, 2) if dr != 0 or dc != 0]
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
SIZE = 19


class Turn:
    def __init__(self, who, n):
        assert n == 1 or n == 2, "n must be 1 or 2"
        self.who = who
        self.n = n

    # prevent accidently modified by ref
    def next(self):
        if self.n == 2:
            return Turn(self.who, 1)
        else:
            return Turn(3 - self.who, 2)


def get_pattern_score(board, turn: Turn, r, c):
    max_score = 0
    target_color = 0
    for dir in DIRS:
        lengths = [0, 0]
        c_offset = 1
        offset = [0, 0]
        colors = [0, 0]
        score = 0
        for d, s in enumerate([+1, -1]):
            dr, dc = dir
            dr, dc = dr * s, dc * s
            for i in range(1, 6):
                rr, cc = r + dr * i, c + dc * i
                if rr < 0 or rr >= SIZE or cc < 0 or cc >= SIZE:
                    break
                if board[rr, cc] == 0:
                    if colors[d] == 0:
                        c_offset += 1
                    else:
                        offset[d] += 1
                else:
                    if colors[d] == 0:
                        colors[d] = board[rr, cc]
                        lengths[d] += 1
                    elif board[rr, cc] == colors[d]:
                        lengths[d] += 1
                    else:
                        break
        if colors[0] == colors[1]:
            # length <= 3 here
            if lengths[0] + c_offset + offset[0] >= 6 and lengths[0] > score:
                score = lengths[0]
                local_target_color = colors[0]
            if lengths[1] + c_offset + offset[1] >= 6 and lengths[1] > score:
                score = lengths[1]
                local_target_color = colors[0]
            # length > 3 here
            if (
                lengths[0] + lengths[1] + c_offset + sum(offset) >= 6
                and 6 - c_offset > score
            ):
                score = min(6 - c_offset, lengths[0] + lengths[1])
                local_target_color = colors[0]
        else:
            for d in [0, 1]:
                if lengths[d] + c_offset >= 6:
                    if lengths[d] > score:
                        score = lengths[d]
                        local_target_color = colors[d]
        score = min(5, score)
        # print(
        #    f"dir: {dir}, lengths: {lengths}, colors: {colors}, offset: {offset}, score: {score}",
        #    file=sys.stderr,
        # )
        if score > max_score:  # must pass at least one update
            max_score = score
            target_color = local_target_color

    if max_score < 4 and target_color == turn.who:
        max_score += 0.5  # encourage attack
    elif max_score == 4 and target_color == turn.who:
        if turn.n == 2:
            max_score += 1.5
        else:
            max_score -= 0.1
    elif max_score == 5 and target_color == turn.who:
        max_score += 1
    return max_score


# TODO: prev possible version
def get_possible_positions(board, turn: Turn, debug=False):
    nonempty_positions = [
        (r, c) for r in range(SIZE) for c in range(SIZE) if board[r, c] != 0
    ]
    possible_positions = set()
    pos_score_list = []
    for np in nonempty_positions:
        for dr, dc in DRDC:
            r, c = np[0] + dr, np[1] + dc
            if 0 <= r < SIZE and 0 <= c < SIZE and board[r, c] == 0:
                if (r, c) not in possible_positions:
                    possible_positions.add((r, c))
                    pattern_score = get_pattern_score(board, turn, r, c)
                    pos_score_list.append(((r, c), pattern_score))
    # return possible_positions
    pos_score_list.sort(key=lambda x: x[1], reverse=True)
    if debug:
        pass
        # print(pos_score_list[:10], file=sys.stderr)
    return [p[0] for p in pos_score_list[:10]]


# TODO: last move version
def check_win(board, last_moves=[]):
    """Checks if a player has won.
    Returns:
    0 - No winner yet
    1 - Black wins
    2 - White wins
    """
    if last_moves:
        for r, c in reversed(last_moves):
            if board[r, c] != 0:
                current_color = board[r, c]
                for dr, dc in DIRS:
                    count = 1
                    for s in (-1, 1):
                        rr, cc = r + s * dr, c + s * dc
                        while (
                            0 <= rr < SIZE
                            and 0 <= cc < SIZE
                            and board[rr, cc] == current_color
                        ):
                            count += 1
                            rr += s * dr
                            cc += s * dc
                    if count >= 6:
                        return current_color
        return 0
    else:
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r, c] != 0:
                    current_color = board[r, c]
                    for dr, dc in DIRS:
                        prev_r, prev_c = r - dr, c - dc
                        if (
                            0 <= prev_r < SIZE
                            and 0 <= prev_c < SIZE
                            and board[prev_r, prev_c] == current_color
                        ):
                            continue
                        count = 0
                        rr, cc = r, c
                        while (
                            0 <= rr < SIZE
                            and 0 <= cc < SIZE
                            and board[rr, cc] == current_color
                        ):
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0


def index_to_label(col):
    """Converts column index to letter (skipping 'I')."""
    return chr(ord("A") + col)  # + (1 if col >= 8 else 0))  # Skips 'I'


def show_distribution(dist):
    """Displays the board as text."""
    sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    # top_n = sorted_dist[:10]
    for position, value in sorted_dist:
        print(
            f"{index_to_label(position[1])}{position[0]+1}: {value:.2f}",
            file=sys.stderr,
        )
    # for r in reversed(range(SIZE)):
    #    for c in range(SIZE):
    #        if (r, c) in dist:
    #            print(f"{dist[(r, c)]:.2f}", end=" ", file=sys.stderr)
    #        else:
    #            print("0.00", end=" ", file=sys.stderr)
    #    print("\n", file=sys.stderr)


# UCT Node for MCTS
class UCTNode:
    def __init__(self, board, turn):
        """
        state: current board state (numpy array)
        score: cumulative score at this node (2048's?)
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = board
        self.turn = turn
        self.children = {}
        self.parent = None
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = get_possible_positions(
            board, turn
        )  # List of untried actions

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


class UCTMCTS:
    def __init__(self, iterations=500, exploration_constant=1.41):
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation

    def select_child(self, node):
        # must be fully expanded
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        if len(node.untried_actions) != 0:
            return node.children[node.untried_actions[0]]
        else:
            uct_max = -np.inf
            child_max = None
            for action, child in node.children.items():
                uct_value = child.total_reward / child.visits + self.c * np.sqrt(
                    np.log(node.visits) / child.visits
                )
                if uct_value > uct_max:
                    uct_max = uct_value
                    child_max = child

            return child_max

    def rollout(self, board, turn: Turn):
        # TODO: Perform a random rollout from the current state up to the specified depth.
        # print("rollout start", file=sys.stderr)
        winner = check_win(board)
        while True:
            # print(f"turn {cnt}", file=sys.stderr)
            # if cnt > 200:
            #    print(board, file=sys.stderr)
            #    exit(0)
            if winner != 0:
                # print("rollout done", file=sys.stderr)
                # print("=" * 80, file=sys.stderr)
                return winner
            legal_moves = get_possible_positions(board, turn)
            # print("legal_moves", legal_moves, file=sys.stderr)
            if len(legal_moves) == 0:
                # print("rollout: no possible action", file=sys.stderr)
                # print(board, file=sys.stderr)
                return 0
            # if len(legal_moves) == 0:
            #    break
            action = legal_moves[0]
            board[action[0], action[1]] = turn.who
            turn = turn.next()
            winner = check_win(board, last_moves=[action])

    def backpropagate(self, node, winner):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        while node is not None:
            node.visits += 1
            if node.turn.who == winner:
                node.total_reward += 1
            elif node.turn.who != 0:
                node.total_reward -= 1
            # == 0 add 0
            node = node.parent

    # don't feed empty board
    def run(self, root):
        node = root
        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        while node.fully_expanded():
            child = self.select_child(node)
            if child is None:
                break
            node = child

        board = node.state.copy()

        # TODO: Expansion: if the node has untried actions, expand one.
        if len(node.untried_actions) != 0:
            action = node.untried_actions[0]
            node.untried_actions.remove(action)
            board[action[0], action[1]] = node.turn.who
            expanded_node = UCTNode(board.copy(), node.turn.next())
            node.children[action] = expanded_node
            expanded_node.parent = node
            node = expanded_node

        # Rollout: Simulate a random game from the expanded node.
        winner = self.rollout(board, node.turn)
        # Backpropagation: Update the tree with the rollout reward.
        self.backpropagate(node, winner)

    def best_action_distribution(self, root):
        """
        Computes the visit count distribution for each action at the root node.
        """
        # print("best_action_distribution", file=sys.stderr)
        if len(root.children) == 0:
            print("best_action_distribution: no children", file=sys.stderr)
            return (-1, -1)
        total_visits = sum(child.visits for child in root.children.values())
        best_visits = -1
        best_action = None
        dist = {}
        for action, child in root.children.items():
            dist[action] = child.visits / total_visits
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, dist


class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if (
                            0 <= prev_r < self.size
                            and 0 <= prev_c < self.size
                            and self.board[prev_r, prev_c] == current_color
                        ):
                            continue
                        count = 0
                        rr, cc = r, c
                        while (
                            0 <= rr < self.size
                            and 0 <= cc < self.size
                            and self.board[rr, cc] == current_color
                        ):
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord("A") + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= "J":  # 'I' is skipped
            return ord(col_char) - ord("A") - 1
        else:
            return ord(col_char) - ord("A")

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(",")
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == "B" else 2

        self.turn = 3 - self.turn
        print("= ", end="", flush=True)

    def generate_move(self, color):
        """Generates a random move for the computer."""
        if color == "B":
            mycolor = BLACK
        else:
            mycolor = WHITE

        # print(get_possible_positions(self.board, color, debug=True), file=sys.stderr)

        if self.game_over:
            print("? Game over")
            return

        if np.all(self.board == 0):
            # print("first move", file=sys.stderr)
            selected = [(SIZE // 2, SIZE // 2)]
        else:
            # print("MCTS", file=sys.stderr)
            # # selected = random.sample(get_possible_positions(self.board), 1)
            uct_mcts = UCTMCTS()
            # print("Black" if mycolor == BLACK else WHITE, self.turn, file=sys.stderr)
            # print(self.board, file=sys.stderr)
            myturn = Turn(mycolor, self.turn)
            root = UCTNode(
                self.board.copy(), myturn
            )  # Initialize the root node for MCTS
            for i in range(5):
                uct_mcts.run(root)
            # print(root.children.keys(), file=sys.stderr)
            best_action, dist = uct_mcts.best_action_distribution(root)
            # show_distribution(dist)
            selected = [best_action]
            # selected = [get_possible_positions(self.board)[0]]

        move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)

        self.play_move(color, move_str)

        print(f"{move_str}\n\n", end="", flush=True)
        # print(move_str, file=sys.stderr)
        return

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join(
                (
                    "X"
                    if self.board[row, col] == 1
                    else "O" if self.board[row, col] == 2 else "."
                )
                for col in range(self.size)
            )
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)

    def process_command(self, command):
        """Parses and executes GTP commands."""
        # print(command, file=sys.stderr)
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return

        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print("", flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")


if __name__ == "__main__":
    try:
        game = Connect6Game()
        game.run()
    except Exception as e:
        import traceback

        print(e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
"""
board = np.zeros((SIZE, SIZE), dtype=int)  # 0: Empty, 1: Black, 2: White
board[10, 9] = 1
board[10, 11] = 1
board[10, 12] = 1
board[10, 13] = 1
print(get_possible_positions(board, debug=True))
"""
