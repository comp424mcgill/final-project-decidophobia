import numpy as np
from collections import defaultdict


class MonteCarloTreeSearchNode():

    def __init__(self, world, parent=None):
        """
        Parameters
        ----------
        world : world.py World
        parent : MonteCarloTreeSearchNode
        """
        self.world = world
        self.parent = parent
        self.children = []
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None

    @property
    def untried_actions(self):
        """
        Returns
        -------
        list of mctspy.games.common.AbstractGameAction

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
        action = self.untried_actions.pop()
        next_state = self.world.step()
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.world.is_game_over()

    def rollout(self):
        end, _, _ = self.world.check_endgame()
        while not end:
            possible_moves = self.world.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = self.world.move(action)
        return self.world.check_endgame()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        # return possible_moves[np.random.randint(len(possible_moves))]
        pass