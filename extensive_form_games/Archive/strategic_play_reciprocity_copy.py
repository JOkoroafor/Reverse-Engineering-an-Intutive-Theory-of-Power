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
    "The objective is to compute the action probabilities for both players"
    "We define 3 ways of computing this which corresponds to level 0,1 and 2 reasoning"
    def __init__(self, game, beta_player1, beta_player2, rho_player1, rho_player2, sigma_player1, sigma_player2, theta_player2):
        """
        Initialize the simulation with different parameters for Player 1 and Player 2.
        :game: Instance of the SharingGame class.
        :param beta_player1: Beta parameter for Player 1.
        :param beta_player2: Beta parameter for Player 2.
        :param rho_player1: Rho parameter for Player 1 (the players social preference for terminal states 
        where their payoff is higher than the other player)
        :param rho_player2: Rho parameter for Player 2 (the players social preference for terminal states 
        where their payoff is higher than the other player)
        :param sigma_player1: Sigma parameter for Player 1 (the players social preference for terminal states 
        where their payoff is lower than the other player)
        :param sigma_player2: Sigma parameter for Player 2 (the players social preference for terminal states 
        where their payoff is lower than the other player)
        :param theta_player2: Theta parameter for Player 2 (the players social preference in reponse to whether or not player 1
        misbehaved)
        """
        self.game = game
        self.beta_player1 = beta_player1  
        self.beta_player2 = beta_player2  
        self.rho_player1 = rho_player1 
        self.rho_player2 = rho_player2  
        self.sigma_player1 = sigma_player1  
        self.sigma_player2 = sigma_player2  
        self.theta_player2 = theta_player2

        "The action probabalities below is what we aim to compute"
        self.action_probabilities = {Player1: {}, Player2: {}}  # Store action probabilities for each player
        
        "Map the rewards at each terminal state to the player"
        self.U_P1_Out = game.get_reward(2)[Player1]
        self.U_P2_Out = game.get_reward(2)[Player2]
        self.U_P1_Left = game.get_reward(4)[Player1]
        self.U_P2_Left = game.get_reward(4)[Player2]
        self.U_P1_Right = game.get_reward(5)[Player1]
        self.U_P2_Right = game.get_reward(5)[Player2]

        self.player1_utilities = [self.U_P1_Out, self.U_P1_Left, self.U_P1_Right]
        self.player2_utilities = [self.U_P2_Out, self.U_P2_Left, self.U_P2_Right]

        "Define the notion of joint utility"
        self.joint_utility_Out = self.U_P1_Out + self.U_P2_Out
        self.joint_utility_Left = self.U_P1_Left + self.U_P2_Left
        self.joint_utility_Right = self.U_P1_Right + self.U_P2_Right

        # Collect them in one 
        self.joint_utilities = [self.joint_utility_Out, self.joint_utility_Left, self.joint_utility_Right]

        # find the maximum
        self.max_player1_payoff = max(self.player1_utilities)
        self.max_player2_payoff = max(self.player2_utilities)
        self.max_joint_payoff = max(self.joint_utilities)

    def get_beta(self, player):
        """Return the beta parameter for the given player."""
        return self.beta_player1 if player == Player1 else self.beta_player2
    
    def get_Recip_params(self, player):
        if player == Player1:
            return self.rho_player1, self.sigma_player1
        else:
            return self.rho_player2, self.sigma_player2

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
                    rho, sigma = self.get_Recip_params(player)
                    theta      = self.theta_player2 if player==Player2 else 0
                    if self_utility > opponent_utility:
                        r = 1
                    else:
                        r = 0
                    if opponent_utility > self_utility:
                        s = 1
                    else:
                        s = 0
                    if player == Player1:
                        mod_opp_reward =(rho * r) + (sigma *s)
                        mod_self_reward = 1 - (rho * r) - (sigma *s)                
                        utility = (mod_opp_reward * opponent_utility) + (mod_self_reward * self_utility)
                    else:
                        if self.joint_utility_Out == self.max_joint_payoff and self.U_P2_Out == self.max_player2_payoff:
                            q = -1
                        else:
                            q = 0     
                        mod_opp_reward =(rho * r) + (sigma *s) + (theta *q)
                        mod_self_reward = 1 - (rho * r) - (sigma *s) - (theta *q)              
                        utility = (mod_opp_reward * opponent_utility) + (mod_self_reward * self_utility) 
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
                            self_utility = self.game.get_player_utility(player, next_state)
                            opponent_utility = self.game.get_player_utility(opponent, next_state)
                            rho, sigma = self.get_Recip_params(player)
                            theta      = self.theta_player2 if player==Player2 else 0
                            if self_utility > opponent_utility:
                                r = 1
                            else:
                                r = 0
                            if opponent_utility > self_utility:
                                s = 1
                            else:
                                s = 0
                            if player == Player1:
                                mod_opp_reward =(rho * r) + (sigma *s)
                                mod_self_reward = 1 - (rho * r) - (sigma *s)                
                                utility = (mod_opp_reward * opponent_utility) + (mod_self_reward * self_utility)
                            else:
                                if self.joint_utility_Out == self.max_joint_payoff and self.U_P2_Out == self.max_player2_payoff:
                                    q = -1
                                else:
                                    q = 0     
                                mod_opp_reward =(rho * r) + (sigma *s) + (theta *q)
                                mod_self_reward = 1 - (rho * r) - (sigma *s) - (theta *q)              
                                utility = (mod_opp_reward * opponent_utility) + (mod_self_reward * self_utility)
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
                    rho, sigma = self.get_Recip_params(player)
                    theta      = self.theta_player2 if player==Player2 else 0
                    if self_utility > opponent_utility:
                        r = 1
                    else:
                        r = 0
                    if opponent_utility > self_utility:
                        s = 1
                    else:
                        s = 0
                    if player == Player1:
                        mod_opp_reward =(rho * r) + (sigma *s)
                        mod_self_reward = 1 - (rho * r) - (sigma *s)                
                        utility = (mod_opp_reward * opponent_utility) + (mod_self_reward * self_utility)
                    else:
                        if self.joint_utility_Out == self.max_joint_payoff and self.U_P2_Out == self.max_player2_payoff:
                            q = -1
                        else:
                            q = 0     
                        mod_opp_reward =(rho * r) + (sigma *s) + (theta *q)
                        mod_self_reward = 1 - (rho * r) - (sigma *s) - (theta *q)              
                        utility = (mod_opp_reward * opponent_utility) + (mod_self_reward * self_utility)
                else:
                    opponent_actions = self.game.get_actions(next_state, opponent)
                    total_utility = 0
                    for opponent_action in opponent_actions:
                        opponent_next_state = self.game.get_transition(next_state, opponent_action)
                        opponent_action_prob = (
                            self.action_probabilities[opponent][next_state].get(opponent_action, 0)
                        )
                        if self.game.is_terminal(opponent_next_state):
                            self_utility = self.game.get_player_utility(player, next_state)
                            opponent_utility = self.game.get_player_utility(opponent, next_state)
                            rho, sigma = self.get_Recip_params(player)
                            theta      = self.theta_player2 if player==Player2 else 0
                            if self_utility > opponent_utility:
                                r = 1
                            else:
                                r = 0
                            if opponent_utility > self_utility:
                                s = 1
                            else:
                                s = 0
                            if player == Player1:
                                mod_opp_reward =(rho * r) + (sigma *s)
                                mod_self_reward = 1 - (rho * r) - (sigma *s)                
                                utility = (mod_opp_reward * opponent_utility) + (mod_self_reward * self_utility)
                            else:
                                if self.joint_utility_Out == self.max_joint_payoff and self.U_P2_Out == self.max_player2_payoff:
                                    q = -1
                                else:
                                    q = 0     
                                mod_opp_reward =(rho * r) + (sigma *s) + (theta *q)
                                mod_self_reward = 1 - (rho * r) - (sigma *s) - (theta *q)              
                                utility = (mod_opp_reward * opponent_utility) + (mod_self_reward * self_utility)
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
game = SharingGame(transitions=transitions, rewards=rewards, actions=actions, initial_state=initial_state)

# Loop through all games in the game_configs dictionary
for game_name, config in game_configs.items():

    # Initialize the game for the current configuration
    game = SharingGame(
        transitions=config["transitions"],
        rewards=config["rewards"],
        actions=config["actions"],
        initial_state=config["initial_state"],
    )

    # Create a simulation object
    simulation = LevelKSimulation_Selfish(
    game=game,
    beta_player1 = 0.33567815435117143,
    beta_player2 = 0.34834185166878495,
)
    # Simulate Level 0 reasoning
    simulation.simulate_level_0(Player1)
    simulation.simulate_level_0(Player2)


     # Print Level 0 action probabilities for both players
    #print(f"Level 0 action probabilities for {game_name} - Player 1:")
    #print(simulation.action_probabilities[Player1])
    #print(f"Level 0 action probabilities for {game_name} - Player 2:")
    #print(simulation.action_probabilities[Player2])

    # Simulate Level 1 reasoning
    simulation.simulate_level_1(Player1, 1)
    simulation.simulate_level_1(Player2, 1)

    # Print Level 1 action probabilities for both players
   #print(f"Level 1 action probabilities for {game_name} - Player 1:")
   # print(simulation.action_probabilities[Player1])
   # print(f"Level 1 action probabilities for {game_name} - Player 2:")
   # print(simulation.action_probabilities[Player2])


    # Simulate Level 2 for Player 1 and Player 2
    simulation.simulate_level_2(Player1, game.get_initial_state())
    simulation.simulate_level_2(Player2, game.get_initial_state())

    # Extract and print probabilities of interest
    p_in_player1 = simulation.action_probabilities[Player1].get(1, {}).get("In", 0)
    p_right_player2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 0)

    #print(f"P(In) for Player 1: {p_in_player1}")
    #print(f"P(Right) for Player 2: {p_right_player2}")