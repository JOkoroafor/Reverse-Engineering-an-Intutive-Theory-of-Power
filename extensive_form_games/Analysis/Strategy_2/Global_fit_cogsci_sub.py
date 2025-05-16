import sys
#Get the project root by going up two levels
sys.path.insert(0, '/Users/junior/Desktop/Files/MIT/Research/Projects/Reverse-engineering an Intuitive Theory of Power/Computational Models/extensive_form_games/Games')

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sharing_game import SharingGame
from strategic_play import LevelKSimulation_Selfish
from game_configs_exp1 import game_configs

Player1 = "1"
Player2 = "2"

# Human data with probabilities
human_data = {
    "trust_game": {"p_in": 0.355, "p_right": 0.741904761904762},
    "strategic_dummy": {"p_in": 0.88190476, "p_right": 0.514523809523809},
    "safe_shot": {"p_in": 0.91833333, "p_right": 0.202142857142857},
    "rational_punish": {"p_in": 0.4997619, "p_right": 0.22547619047619},
    "near_dictator": {"p_in": 0.19071429, "p_right": 0.807619048},
    "free_punish": {"p_in": 0.5402381, "p_right": 0.451428571},
    "free_help": {"p_in": 0.38238095, "p_right": 0.071428571},
    "costly_punish": {"p_in": 0.65952381, "p_right": 0.256904762},
    "costly_help": {"p_in": 0.38047619, "p_right": 0.858571429},
    "common_interest": {"p_in": 0.76047619, "p_right": 0.051904762},
}

# Negative log-likelihood function
def negative_log_likelihood(beta, game, target_probs):
    beta_player1, beta_player2 = beta
    simulation = LevelKSimulation_Selfish(game, beta_player1, beta_player2)

    # Simulate Level 2 reasoning
    simulation.simulate_level_2(Player1, game.get_initial_state())
    simulation.simulate_level_2(Player2, game.get_initial_state())

    # Get action probabilities for player 1 player 2 for all games in game_configs
    p_in_player1 = simulation.action_probabilities[Player1].get(1, {}).get("In", 0)
    p_right_player2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 0)

    # Sets the actual human action probabilities as the target
    target_p_in = target_probs["p_in"]
    target_p_right = target_probs["p_right"]

    # Avoid invalid probabilities (e.g., log(0))
    epsilon = 1e-9
    p_in_player1 = max(min(p_in_player1, 1 - epsilon), epsilon)
    p_right_player2 = max(min(p_right_player2, 1 - epsilon), epsilon)

    # Compute log-likelihood for both Player 1 and Player 2
    log_likelihood_player1 = target_p_in * np.log(p_in_player1) + (1 - target_p_in) * np.log(1 - p_in_player1)
    log_likelihood_player2 = target_p_right * np.log(p_right_player2) + (1 - target_p_right) * np.log(1 - p_right_player2)

    # Return negative log-likelihood for both players
    return -(log_likelihood_player1 + log_likelihood_player2)

# Global Fit
result_global = minimize(
    lambda beta: sum(
        negative_log_likelihood(beta, SharingGame(
            transitions=game_configs[game]["transitions"],
            rewards=game_configs[game]["rewards"],
            actions=game_configs[game]["actions"],
            initial_state=game_configs[game]["initial_state"]),
            human_data[game])
        for game in game_configs
    ),
    x0=[0.5, 1],
    bounds=[(0.01, 10), (0.01, 10)],
    method="L-BFGS-B",
)

global_betas = {"beta_player1": result_global.x[0], "beta_player2": result_global.x[1]}
print(f"Global Betas: {global_betas}")



