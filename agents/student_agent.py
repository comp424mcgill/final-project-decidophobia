# Student agent: Add your own agent here
from collections import defaultdict
from copy import deepcopy

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        start = time.time()
        board_size = len(chess_board)
        my_world = MyWorld(board_size, chess_board, max_step, my_pos, adv_pos)
        # foo = my_world.get_legal_actions()
        # bar = my_world.random_walk(my_pos, adv_pos)
        # x = 1

        root = MonteCarloTreeSearchNode(state=my_world)
        selected_node = root.best_action()
        # dummy return
        pos, dir = selected_node.parent_action
        print(time.time() - start)
        return pos, dir


class MyWorld:
    def __init__(self, board_size, chess_board, max_step, my_pos, adv_pos):
        """
        Initialize the game world
        """

        self.dir_names = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

        self.board_size = board_size
        self.chess_board = chess_board
        self.max_step = max_step
        self.p0_pos = my_pos
        self.p1_pos = adv_pos

    def check_valid_step(self, start_pos, end_pos, barrier_dir):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : np.ndarray
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False

        # Get position of the adversary
        adv_pos = self.p1_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            (r, c) = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def check_endgame(self):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                        self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(self.p0_pos))
        p1_r = find(tuple(self.p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        return True, p0_score, p1_score

    def check_boundary(self, pos):
        r, c = pos
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    # def set_barrier(self, r, c, dir):
    #     # Set the barrier to True
    #     self.chess_board[r, c, dir] = True
    #     # Set the opposite barrier to True
    #     move = self.moves[dir]
    #     self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def random_walk(self, my_pos, adv_pos):
        """
        Randomly walk to the next position in the board.

        Parameters
        ----------
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        """
        my_pos = np.asarray(my_pos)
        ori_pos = deepcopy(my_pos)
        steps = np.random.randint(0, self.max_step + 1)
        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = self.moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            adv_pos = tuple(adv_pos)
            while self.chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 200:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = self.moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 200:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while self.chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    def get_legal_actions(self):
        """
        get all valid actions of a player
        Returns
        -------
        list of ((x, y), dir)
        """

        # cur_player, cur_pos, adv_pos = self.get_current_player()

        cur_pos = self.p0_pos
        # adv_pos = self.p1_pos

        all_coor = list()
        legal_move = list()
        for i in range(self.board_size):
            for j in range(self.board_size):
                all_coor.append((i, j))
        direction = [0, 1, 2, 3]

        for pos in all_coor:

            if abs(pos[0] - self.p0_pos[0]) + abs(pos[1] - self.p1_pos[1]) > self.max_step:
                continue

            for d in direction:
                next_pos = np.asarray(pos)
                if self.check_valid_step(np.asarray(cur_pos), next_pos, d):
                    legal_move.append((pos, d))

        return legal_move

    def move(self, player, next_pos, dir):
        next_pos = np.asarray(next_pos)
        # player == 0

        p0_pos = self.p0_pos
        p1_pos = self.p1_pos
        board = deepcopy(self.chess_board)

        if not player:
            p0_pos = next_pos
        else:
            p1_pos = next_pos

        r, c = next_pos
        board[r, c, dir] = True

        move = self.moves[dir]
        board[r + move[0], c + move[1], self.opposites[dir]] = True

        new_world = MyWorld(self.board_size, board, self.max_step, p0_pos, p1_pos)

        return new_world

    def is_game_over(self):
        game, p0, p1 = self.check_endgame()
        return game

    def game_result(self):
        game, p0, p1 = self.check_endgame()

        if p0 < p1:
            return -1
        elif p0 == p1:
            return 0
        else:
            return 1

    # update world
    # def update_world(self, next_pos, dir):
    #     start_time = time()
    #     self.update_player_time(time() - start_time)
    #     next_pos = np.asarray(next_pos)
    #     if not self.turn:
    #         self.p0_pos = next_pos
    #     else:
    #         self.p1_pos = next_pos
    #     # Set the barrier to True
    #     r, c = next_pos
    #     self.set_barrier(r, c, dir)
    #
    #     # Change turn
    #     self.turn = 1 - self.turn
    #     return self


class MonteCarloTreeSearchNode:

    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        # self._untried_actions = None
        self._untried_actions = self.untried_actions()

    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        pos, dir = action
        next_state = self.state.move(0, pos, dir)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        player_turn = True

        while not current_rollout_state.is_game_over():
            # possible_moves = current_rollout_state.get_legal_actions()

            if player_turn:
                # action = self.rollout_policy(possible_moves)
                pos, dir = self.state.random_walk(self.state.p0_pos, self.state.p1_pos)
                current_rollout_state = current_rollout_state.move(0, pos, dir)
                player_turn = False
            else:
                pos, dir = self.state.random_walk(self.state.p1_pos, self.state.p0_pos)
                current_rollout_state = current_rollout_state.move(1, pos, dir)
                player_turn = True

        return current_rollout_state.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        if len(choices_weights) == 0:
            print("!!!!!!!!!!!!!!!!!!!!!!")
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):

        # play randomly
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_time = 2.2 - 0.05 * self.state.board_size

        end =time.time() + simulation_time

        while time.time() <end:
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        # for i in range(simulation_no):
        #

        return self.best_child(c_param=0.)
