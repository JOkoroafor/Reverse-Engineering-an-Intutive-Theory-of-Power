import sys
import csv
import math

# Get the project root by going up two levels
sys.path.insert(0, '/Users/junior/Desktop/Files/MIT/Research/Projects/Reverse-engineering an Intuitive Theory of Power/Computational Models/extensive_form_games/Games')

import numpy as np
from sharing_game import SharingGame
from game_configs_training import game_configs

Player1 = "1"
Player2 = "2"

def softmax(x, beta):
    """ Apply softmax with an inverse temperature parameter (beta) to a list of values (expected utilities). """
    x = np.array(x)
    exp_x = np.exp(beta * x)
    return exp_x / np.sum(exp_x)  # Returns a choice probability for each action that is associated with a given expected utility (x)


class LevelKSimulation_Selfish:
    "The objective is to compute the action probabilities for both players"
    "We define 3 ways of computing this which corresponds to level 0,1 and 2 reasoning"
    "For the levels of reasoning that involve utility calculations, the utility of each player"
    "... corresponds to their direct payoff in the game"
    def __init__(self, game, beta_player1, beta_player2):
        """
        Initialize the simulation with different beta parameters for Player 1 and Player 2.
        :param game: Instance of the SharingGame class.
        :param beta_player1: Beta parameter for Player 1.
        :param beta_player2: Beta parameter for Player 2.
        """
        self.game = game
        self.beta_player1 = beta_player1  # Beta for Player 1
        self.beta_player2 = beta_player2  # Beta for Player 2
        "The action probabalities below is what we aim to compute"
        self.action_probabilities = {Player1: {}, Player2: {}}  # Store action probabilities for each player

    def get_beta(self, player):
        """Return the beta parameter for the given player."""
        return self.beta_player1 if player == Player1 else self.beta_player2

    def simulate_level_0(self, player):
        """ Level 0 players choose actions uniformly at random. """
        for state in self.game.transitions.keys():
            actions = self.game.get_actions(state, player)
            if actions:
                prob = 1 / len(actions) # This is the general calculation for a player's action probabality in all states where that player can act
                self.action_probabilities[player][state] = {action: prob for action in actions}

    def simulate_level_1(self, player, state, depth=0, max_depth=3):
        if depth >= max_depth:
            return 0  # End recursion to avoid infinite depth

        for state in self.game.transitions.keys():
            actions = self.game.get_actions(state, player)
            expected_utilities = []

            for action in actions:
                next_state = self.game.get_transition(state, action)
                if self.game.is_terminal(next_state):
                    utility = self.game.get_player_utility(player, next_state)
                else:
                    # Simulate opponent's response
                    opponent = self.game.opponent(player)
                    opponent_actions = self.game.get_actions(next_state, opponent)
                    total_utility = 0
                    for opponent_action in opponent_actions:
                        opponent_next_state = self.game.get_transition(next_state, opponent_action)
                        opponent_action_prob = (
                            self.action_probabilities[opponent][next_state].get(opponent_action, 0)
                        )
                        if self.game.is_terminal(opponent_next_state):
                            utility = self.game.get_player_utility(player, opponent_next_state)
                        else:
                            utility = self.simulate_level_1(opponent, opponent_next_state, depth + 1, max_depth)
                        total_utility += utility * opponent_action_prob
                    utility = total_utility
                expected_utilities.append(utility)

            # Convert expected utilities to probabilities using softmax and the player's beta
            if expected_utilities:
                beta = self.get_beta(player)
                probabilities = softmax(expected_utilities, beta=beta)
                self.action_probabilities[player][state] = {
                    action: float(prob) for action, prob in zip(actions, probabilities)
                }

    def simulate_level_2(self, player, state, depth=0, max_depth=3):
        """Simulate Level 2 reasoning for the given player."""
        # Ensure Level 1 probabilities for the opponent are computed first
        opponent = self.game.opponent(player)
        if not self.action_probabilities[opponent]:
            self.simulate_level_1(opponent, state, depth=0, max_depth=max_depth)

        for state in self.game.transitions.keys():
            actions = self.game.get_actions(state, player)
            expected_utilities = []

            for action in actions:
                next_state = self.game.get_transition(state, action)
                if self.game.is_terminal(next_state):
                    utility = self.game.get_player_utility(player, next_state)
                else:
                    opponent_actions = self.game.get_actions(next_state, opponent)
                    total_utility = 0
                    for opponent_action in opponent_actions:
                        opponent_next_state = self.game.get_transition(next_state, opponent_action)
                        opponent_action_prob = (
                            self.action_probabilities[opponent][next_state].get(opponent_action, 0)
                        )
                        if self.game.is_terminal(opponent_next_state):
                            utility = self.game.get_player_utility(player, opponent_next_state)
                        else:
                            utility = self.simulate_level_2(opponent, opponent_next_state, depth + 1, max_depth)
                        total_utility += utility * opponent_action_prob
                    utility = total_utility
                expected_utilities.append(utility)

            # Convert expected utilities to probabilities using softmax and the player's beta
            if expected_utilities:
                beta = self.get_beta(player)
                probabilities = softmax(expected_utilities, beta=beta)
                self.action_probabilities[player][state] = {
                    action: float(prob) for action, prob in zip(actions, probabilities)
                }

class LevelKSimulation_IA:
    "The objective is to compute the action probabilities for both players"
    "We define 3 ways of computing this which corresponds to level 0,1 and 2 reasoning"
    "For the levels of reasoning that involve utility calculations, the utility of each player"
    "... corresponds to their direct payoff in the game"
    def __init__(self, game, beta_player1, beta_player2, delta_player1, delta_player2, alpha_player1, alpha_player2):
        """
        Initialize the simulation with different beta parameters for Player 1 and Player 2.
        :param game: Instance of the SharingGame class.
        :param beta_player1: Beta parameter for Player 1.
        :param beta_player2: Beta parameter for Player 2.
        """
        self.game = game
        self.beta_player1 = beta_player1  # Beta for Player 1
        self.beta_player2 = beta_player2  # Beta for Player 2

        self.delta_player1   = delta_player1   # P1’s disad‐ineq weight
        self.delta_player2   = delta_player2   # P2’s disad‐ineq weight  
        self.alpha_player1   = alpha_player1   # P1’s adV‐ineq weight
        self.alpha_player2   = alpha_player2    # P2’s adv‐ineq weight

        "The action probabalities below is what we aim to compute"
        self.action_probabilities = {Player1: {}, Player2: {}}  # Store action probabilities for each player

    def get_beta(self, player):
        """Return the beta parameter for the given player."""
        return self.beta_player1 if player == Player1 else self.beta_player2

    def get_IA_params(self, player):
        if player == Player1:
            return self.delta_player1, self.alpha_player1
        else:
            return self.delta_player2, self.alpha_player2
        
    def simulate_level_0(self, player):
        """ Level 0 players choose actions uniformly at random. """
        for state in self.game.transitions.keys():
            actions = self.game.get_actions(state, player)
            if actions:
                prob = 1 / len(actions) # This is the general calculation for a player's action probabality in all states where that player can act
                self.action_probabilities[player][state] = {action: prob for action in actions} # Stores action probabalities for each player in a given state.

    def simulate_level_1(self, player, state, depth=0, max_depth=3):
        opponent = self.game.opponent(player)
        if depth >= max_depth:
            return 0  # End recursion to avoid infinite depth

        for state in self.game.transitions.keys():
            actions = self.game.get_actions(state, player)
            expected_utilities = []
            for action in actions:
                next_state = self.game.get_transition(state, action)
                if self.game.is_terminal(next_state):
                    self_utility = self.game.get_player_utility(player, next_state)
                    opponent_utility = self.game.get_player_utility(opponent, next_state)
                    delta, alpha = self.get_IA_params(player)
                    "Setting delta (weighting over disadvantegeous equity) and alpha (weighting over advantageous equity) as free parameters"
                    disad = delta * max(opponent_utility - self_utility, 0)
                    ad = alpha * max(self_utility - opponent_utility, 0)
                    utility = self_utility - disad - ad
                else:
                    # Simulate opponent's response
                    opponent_actions = self.game.get_actions(next_state, opponent)
                    total_utility = 0
                    for opponent_action in opponent_actions:
                        opponent_next_state = self.game.get_transition(next_state, opponent_action)
                        opponent_action_prob = (
                            self.action_probabilities[opponent][next_state].get(opponent_action, 0)
                        )
                        if self.game.is_terminal(opponent_next_state):
                            self_utility = self.game.get_player_utility(player, opponent_next_state)
                            opponent_utility = self.game.get_player_utility(opponent, opponent_next_state)
                            delta, alpha = self.get_IA_params(player)
                            "Setting delta (weighting over disadvantegeous equity) and alpha (weighting over advantageous equity) as free parameters"
                            disad = delta * max(opponent_utility - self_utility, 0)
                            ad = alpha * max(self_utility - opponent_utility, 0)
                            utility = self_utility - disad - ad
                        else:
                            utility = self.simulate_level_1(opponent, opponent_next_state, depth + 1, max_depth)
                        total_utility += utility * opponent_action_prob
                    utility = total_utility
                expected_utilities.append(utility)

            # Convert expected utilities to probabilities using softmax and the player's beta
            if expected_utilities:
                beta = self.get_beta(player)
                probabilities = softmax(expected_utilities, beta=beta)
                self.action_probabilities[player][state] = {
                    action: float(prob) for action, prob in zip(actions, probabilities)
                }

    def simulate_level_2(self, player, state, depth=0, max_depth=3):
        """Simulate Level 2 reasoning for the given player."""
        # Ensure Level 1 probabilities for the opponent are computed first
        opponent = self.game.opponent(player)
        if not self.action_probabilities[opponent]:
            self.simulate_level_1(opponent, state, depth=0, max_depth=max_depth)

        for state in self.game.transitions.keys():
            actions = self.game.get_actions(state, player)
            expected_utilities = []
            for action in actions:
                next_state = self.game.get_transition(state, action)
                if self.game.is_terminal(next_state):
                    self_utility = self.game.get_player_utility(player, next_state)
                    opponent_utility = self.game.get_player_utility(opponent, next_state)
                    delta, alpha = self.get_IA_params(player)
                    "Setting delta (weighting over disadvantegeous equity) and alpha (weighting over advantageous equity) as free parameters"
                    disad = delta * max(opponent_utility - self_utility, 0)
                    ad = alpha * max(self_utility - opponent_utility, 0)
                    utility = self_utility - disad - ad
                else:
                    opponent_actions = self.game.get_actions(next_state, opponent)
                    total_utility = 0
                    for opponent_action in opponent_actions:
                        opponent_next_state = self.game.get_transition(next_state, opponent_action)
                        opponent_action_prob = (
                            self.action_probabilities[opponent][next_state].get(opponent_action, 0)
                        )
                        if self.game.is_terminal(opponent_next_state):
                            self_utility = self.game.get_player_utility(player, opponent_next_state)
                            opponent_utility = self.game.get_player_utility(opponent, opponent_next_state)
                            delta, alpha = self.get_IA_params(player)
                            "Setting delta (weighting over disadvantegeous equity) and alpha (weighting over advantageous equity) as free parameters"
                            disad = delta * max(opponent_utility - self_utility, 0)
                            ad = alpha * max(self_utility - opponent_utility, 0)
                            utility = self_utility - disad - ad
                        else:
                            utility = self.simulate_level_2(opponent, opponent_next_state, depth + 1, max_depth)
                        total_utility += utility * opponent_action_prob
                    utility = total_utility
                expected_utilities.append(utility)

            # Convert expected utilities to probabilities using softmax and the player's beta
            if expected_utilities:
                beta = self.get_beta(player)
                probabilities = softmax(expected_utilities, beta=beta)
                self.action_probabilities[player][state] = {
                    action: float(prob) for action, prob in zip(actions, probabilities)
                }

class LevelKSimulation_Reciprocity:
    def __init__(self, game: SharingGame,
                 beta1, beta2,
                 rho1, rho2,
                 sigma1, sigma2,
                 theta2):
        """
        :param game: a SharingGame
        :param beta1, beta2: softmax inverse‐temp for P1, P2
        :param rho1, rho2: CR rho for P1, P2 (envy when you earn more)
        :param sigma1, sigma2: CR sigma for P1, P2 (envy when you earn less)
        :param theta2: CR theta for P2 (negative reciprocity if P1 “misbehaved”)
        """
        self.game = game
        # softmax temperatures
        self.beta = {Player1: beta1, Player2: beta2}
        # CR parameters
        self.rho   = {Player1: rho1,  Player2: rho2}
        self.sigma = {Player1: sigma1, Player2: sigma2}
        self.theta = {Player1: 0.0,    Player2: theta2}

        # we'll fill in action_probabilities[state] → {action:prob}
        self.action_probabilities = {Player1: {}, Player2: {}}

        # precompute the joint payoffs in Out / Left / Right
        r2 = game.get_reward(2)
        r4 = game.get_reward(4)
        r5 = game.get_reward(5)
        self.joint_out   = r2[Player1] + r2[Player2]
        self.joint_left  = r4[Player1] + r4[Player2]
        self.joint_right = r5[Player1] + r5[Player2]
        self.best_joint = max(self.joint_out,
                              self.joint_left,
                              self.joint_right)

    def simulate_level_0(self, player):
        """Uniform random over all actions."""
        for s in self.game.transitions:
            acts = self.game.get_actions(s, player)
            if not acts:
                continue
            p = 1 / len(acts)
            self.action_probabilities[player][s] = {a: p for a in acts}

    def _recip_utility(self, player, x_i, x_j, via_in):
        """Charness–Rabin utility at a terminal node."""
        # social‐preference weights
        rho, sigma = self.rho[player], self.sigma[player]
        theta      = self.theta[player]

        # inequality indicators
        r = 1 if x_i > x_j else 0
        s = 1 if x_j > x_i else 0

        # only P2 punishes (q=−1) if we arrived via In *and* Out was best joint
        q = -1 if (player == Player2 and via_in and self.joint_out == self.best_joint) else 0

        w_other = rho*r + sigma*s + theta*q
        # weighted average of other's payoff vs your payoff
        return w_other * x_j + (1 - w_other) * x_i

    def simulate_level_1(self, player, state, depth=0, max_depth=2):
        """Level-1: best‐response (softmax) to opponent’s level-0."""
        if depth >= max_depth:
            return 0.0

        opponent = self.game.opponent(player)
        for s in self.game.transitions:
            acts = self.game.get_actions(s, player)
            utils = []
            for a in acts:
                ns = self.game.get_transition(s, a)

                # terminal vs non-terminal
                if self.game.is_terminal(ns):
                    x_i = self.game.get_player_utility(player, ns)
                    x_j = self.game.get_player_utility(opponent, ns)
                    # only via_in if (s, a) = (1, "In")
                    via_in = (s == self.game.initial_state and a == "In")
                    u = self._recip_utility(player, x_i, x_j, via_in)
                else:
                    # simulate opponent’s level-0 at ns
                    total = 0.0
                    for a2, p2 in self.action_probabilities[opponent][ns].items():
                        ns2 = self.game.get_transition(ns, a2)
                        if self.game.is_terminal(ns2):
                            xi = self.game.get_player_utility(player, ns2)
                            xj = self.game.get_player_utility(opponent, ns2)
                            via_in = (s == self.game.initial_state and a == "In")
                            util = self._recip_utility(player, xi, xj, via_in)
                        else:
                            util = self.simulate_level_1(
                                opponent, ns2, depth + 1, max_depth)
                        total += p2 * util
                    u = total

                utils.append(u)

            # softmax over utils
            β = self.beta[player]
            probs = softmax(utils, β)
            self.action_probabilities[player][s] = {
                a: float(p) for a, p in zip(acts, probs)
            }

    def simulate_level_2(self, player, state):
        """Level-2: best‐response (softmax) to opponent’s level-1."""
        opponent = self.game.opponent(player)
        # make sure opponent’s level-1 is filled in
        if not self.action_probabilities[opponent]:
            # we need their level-0 first
            self.simulate_level_0(opponent)
            self.simulate_level_1(opponent, self.game.initial_state)

        # now do same as level-1 but recursing to simulate_level_2
        for s in self.game.transitions:
            acts = self.game.get_actions(s, player)
            utils = []
            for a in acts:
                ns = self.game.get_transition(s, a)
                if self.game.is_terminal(ns):
                    xi = self.game.get_player_utility(player, ns)
                    xj = self.game.get_player_utility(opponent, ns)
                    via_in = (s == self.game.initial_state and a == "In")
                    u = self._recip_utility(player, xi, xj, via_in)
                else:
                    total = 0.0
                    for a2, p2 in self.action_probabilities[opponent][ns].items():
                        ns2 = self.game.get_transition(ns, a2)
                        if self.game.is_terminal(ns2):
                            xi = self.game.get_player_utility(player, ns2)
                            xj = self.game.get_player_utility(opponent, ns2)
                            via_in = (s == self.game.initial_state and a == "In")
                            util = self._recip_utility(player, xi, xj, via_in)
                        else:
                            util = self.simulate_level_2(
                                opponent, ns2)
                        total += p2 * util
                    u = total
                utils.append(u)

            β = self.beta[player]
            probs = softmax(utils, β)
            self.action_probabilities[player][s] = {
                a: float(p) for a, p in zip(acts, probs)
            }

# Example configuration for a single game
transitions = {1: {"Out": 2, "In": 3}, 3: {"Left": 4, "Right": 5}}
rewards = {
    2: {Player1: 5, Player2: 5},
    4: {Player1: 10, Player2: 10},
    5: {Player1: 0, Player2: 0},
}
actions = {1: {Player1: ["Out", "In"]}, 3: {Player2: ["Left", "Right"]}}
initial_state = 1

"Simulating Play"
game = SharingGame(
    transitions=["transitions"],
    rewards=["rewards"],
    actions=["actions"],
    initial_state=["initial_state"],
)
sim = LevelKSimulation_Reciprocity(
    game,
    beta1=0.3, beta2=0.3,
    rho1=0.5,  rho2=0.5,
    sigma1=0.2, sigma2=0.2,
    theta2=0.1,
)
# seed level-0 and level-1 for both players
sim.simulate_level_0(Player1)
sim.simulate_level_0(Player2)
sim.simulate_level_1(Player1, game.initial_state)
sim.simulate_level_1(Player2, game.initial_state)
# then level-2
sim.simulate_level_2(Player1, game.initial_state)
sim.simulate_level_2(Player2, game.initial_state)

# extract the final probabilities of interest:
p_in    = sim.action_probabilities[Player1][game.initial_state]["In"]
p_right = sim.action_probabilities[Player2][3]["Right"]
