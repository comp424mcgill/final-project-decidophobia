import numpy as np
from collections import defaultdict
from copy import deepcopy

class MonteCarloTreeSearchNode():

    def __init__(self, world, parent=None):
        """
        Parameters
        ----------
        world : world.py World
        parent : MonteCarloTreeSearchNode
        _untried_actions: list of ((x, y), dir)
        """
        self.world = world
        self.parent = parent
        self.children = []
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None
        # self.agent .....

    @property
    def untried_actions(self):
        """
        Returns
        -------
        _untried_actions: list of ((x, y), dir)

        """
        if self._untried_actions is None:
            self._untried_actions = self.world.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        # wins = self._results[self.parent.state.next_to_move]
        # loses = self._results[-1 * self.parent.state.next_to_move]
        # return wins - loses
        pass

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        #((x,y),dir)
        action = self.untried_actions.pop()
        next_pos, dir = action
        # update world
        world = self.world.update_world(next_pos, dir)
        child_node = MonteCarloTreeSearchNode(
            world, parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        end, _, _ = self.world.check_endgame()
        return end

    def rollout(self):
        """
        get the result of current game
        Returns
        -------
        is_end, p0_score, p1_score
        """
        current_rollout_world = self.world
        end, _, _ = current_rollout_world.check_endgame()
        while not end:
            possible_moves = current_rollout_world.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            next_pos, dir = action
            current_rollout_world = current_rollout_world.update_world(next_pos, dir)
        return current_rollout_world.check_endgame()

    def backpropagate(self, p0_score, p1_score):
        '''
        Parameters
        ----------
        result: p0_score, p1_score
        '''
        self._number_of_visits += 1.
        self._results[0] += p0_score
        self._results[1] += p1_score
        if self.parent:
            self.parent.backpropagate(p0_score, p1_score)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        '''
        Parameters
        ----------
        possible_moves: list of ((x, y), dir)
        Return
        ----------
        ((x, y), dir)
        '''
        return possible_moves[np.random.randint(len(possible_moves))]