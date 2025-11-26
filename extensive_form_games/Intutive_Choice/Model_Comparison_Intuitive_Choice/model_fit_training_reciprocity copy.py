import sys
#Get the project root by going up two levels
sys.path.insert(0, '/Users/junior/Desktop/Files/MIT/Research/Projects/Reverse-engineering an Intuitive Theory of Power/Computational Models/extensive_form_games/Games')
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
from sharing_game import SharingGame
from strategic_play_reciprocity import LevelKSimulation_Selfish
from strategic_play_reciprocity import LevelKSimulation_IA
from strategic_play_reciprocity import LevelKSimulation_Recip_Parameter
from game_configs_training import game_configs

Player1 = "1"
Player2 = "2"

# 1) Read the CSV
file_path = "training_data_2_summary_stats.csv"
df = pd.read_csv(file_path)

# 2) Check columns then build a small dict for your likelihood
required = [
    "Game",
    "Mean_P1_Expectation",
    "Mean_P2_Expectation",
    # (you can keep SDs around if you need CIs later)
]
if not all(col in df.columns for col in required):
    raise ValueError(f"CSV is missing one of {required}")

# index by Game
df = df.set_index("Game")
human_probs = {
    game: {
        "p_in":  df.at[game, "Mean_P1_Expectation"],
        "p_right": df.at[game, "Mean_P2_Expectation"]
    }
    for game in df.index
}

# Negative log-likelihood functions
def negative_log_likelihood_selfish(beta, game, target_probs):
    beta_player1, beta_player2 = beta
    simulation = LevelKSimulation_Selfish(game, beta_player1, beta_player2)

    simulation.simulate_level_0(Player1)
    simulation.simulate_level_0(Player2)

    simulation.simulate_level_1(Player1, game.get_initial_state())
    simulation.simulate_level_1(Player2, game.get_initial_state())

    simulation.simulate_level_2(Player1, game.get_initial_state())
    simulation.simulate_level_2(Player2, game.get_initial_state())

    # Get simulated probabilities
    p_in_player1 = simulation.action_probabilities[Player1].get(1, {}).get("In", 0)
    p_right_player2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 0)

    # Retrieve target probabilities
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


"Compute best fitting beta for self-interested model"
def total_nll_selfish(beta):
    return sum(
        negative_log_likelihood_selfish(
            beta,
            SharingGame(
                transitions   = game_configs[g]["transitions"],
                rewards       = game_configs[g]["rewards"],
                actions       = game_configs[g]["actions"],
                initial_state = game_configs[g]["initial_state"],
            ),
            human_probs[g]
        )
        for g in game_configs
        if g in human_probs       # skip any config not in your CSV
    )

result_global_selfish = minimize(
    total_nll_selfish,
    x0=[0.5, 1],
    bounds=[(0.01, 10), (0.01, 10)],
    method="L-BFGS-B",
)

global_betas_selfish = {"beta_player1": result_global_selfish.x[0], "beta_player2": result_global_selfish.x[1]}

# --- Global fit for **IA** model ---
"Compute best fitting beta for inequality aversed model"
def negative_log_likelihood_IA(params, game, target_probs):
    """
    NLL for the IA model with 6 free parameters:
      params = [beta1, beta2, delta1, delta2, alpha1, alpha2]
    beta   = softmax temperature (for both players)
    delta   = disadvantageous‐inequality aversion weighting
    alpha = advantageous‐inequality aversion weighting
    """

    beta_player1, beta_player2, delta_player1, delta_player2, alpha_player1, alpha_player2= params

    # instantiate the IA simulator with two gamma's and the FS weights
    simulation = LevelKSimulation_IA(
        game,
        beta_player1, beta_player2,
        delta_player1, delta_player2,
        alpha_player1, alpha_player2
    )

    simulation.simulate_level_0(Player1)
    simulation.simulate_level_0(Player2)

    simulation.simulate_level_1(Player1, game.get_initial_state())
    simulation.simulate_level_1(Player2, game.get_initial_state())


    simulation.simulate_level_2(Player1, game.get_initial_state())
    simulation.simulate_level_2(Player2, game.get_initial_state())

    # pull out the predicted probabilities
    p1 = simulation.action_probabilities[Player1].get(1, {}).get("In",    1e-9)
    p2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 1e-9)
    # clamp to avoid log(0)
    p1, p2 = np.clip([p1, p2], 1e-9, 1 - 1e-9)

    # human targets
    t1 = target_probs["p_in"]
    t2 = target_probs["p_right"]

    # binary log‐likelihood
    ll1 = t1 * np.log(p1) + (1 - t1) * np.log(1 - p1)
    ll2 = t2 * np.log(p2) + (1 - t2) * np.log(1 - p2)

    return -(ll1 + ll2)

def total_nll_IA(params):
    return sum(
        negative_log_likelihood_IA(
            params,
            SharingGame(
                transitions   = game_configs[g]["transitions"],
                rewards       = game_configs[g]["rewards"],
                actions       = game_configs[g]["actions"],
                initial_state = game_configs[g]["initial_state"],
            ),
            human_probs[g]
        )
        for g in game_configs
        if g in human_probs       # skip any config not in your CSV
    )

# two inequality constraints: δ1−α1 ≥ 0 and δ2−α2 ≥ 0
cons = [
    {"type": "ineq", "fun": lambda x: x[2] - x[4]},  # delta1 - alpha1 >= 0
    {"type": "ineq", "fun": lambda x: x[3] - x[5]},  # delta2 - alpha2 >= 0
]

result_global_IA = minimize(
    total_nll_IA,
    x0 = [0.5, 1.0, 0.5,   0.5, 1.0, 0.5],   # [β1, δ1, α1, β2, δ2, α2]
    bounds=[(0.00, 100),(0.00, 100),(0.00, 100),(0.00, 100),(0.00, 100),(0.00, 100) ],
    constraints=cons,
    method="SLSQP",
)

global_params_IA = {"beta_player1": result_global_IA.x[0], "beta_player2": result_global_IA.x[1], 
                    "delta_player1": result_global_IA.x[2], "delta_player2": result_global_IA.x[3],
                    "alpha_player1": result_global_IA.x[4], "alpha_player2": result_global_IA.x[5]}



# --- Global fit for **Reciprocity** model ---
"Compute best fitting beta for inequality aversed model"
def negative_log_likelihood_Reciprocity(params, game, target_probs):
    """
    NLL for the Recriprocity model with 7 free parameters:
      params = [beta1, beta2, theta2]
    beta   = softmax temperature (for both players)
    theta = joint payoff preference when the other player "misbehaves"
    """

    beta_player1, beta_player2, theta_player2= params

    # instantiate the IA simulator with two gamma's and the FS weights
    simulation = LevelKSimulation_Recip_Parameter(
        game,
        beta_player1, beta_player2,
        theta_player2
    )

    simulation.simulate_level_0(Player1)
    simulation.simulate_level_0(Player2)

    simulation.simulate_level_1(Player1, game.get_initial_state())
    simulation.simulate_level_1(Player2, game.get_initial_state())


    simulation.simulate_level_2(Player1, game.get_initial_state())
    simulation.simulate_level_2(Player2, game.get_initial_state())

    # pull out the predicted probabilities
    p1 = simulation.action_probabilities[Player1].get(1, {}).get("In",    1e-9)
    p2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 1e-9)
    # clamp to avoid log(0)
    p1, p2 = np.clip([p1, p2], 1e-9, 1 - 1e-9)

    # human targets
    t1 = target_probs["p_in"]
    t2 = target_probs["p_right"]

    # binary log‐likelihood
    ll1 = t1 * np.log(p1) + (1 - t1) * np.log(1 - p1)
    ll2 = t2 * np.log(p2) + (1 - t2) * np.log(1 - p2)

    return -(ll1 + ll2)

def total_nll_Reciprocity(params):
    return sum(
        negative_log_likelihood_Reciprocity(
            params,
            SharingGame(
                transitions   = game_configs[g]["transitions"],
                rewards       = game_configs[g]["rewards"],
                actions       = game_configs[g]["actions"],
                initial_state = game_configs[g]["initial_state"],
            ),
            human_probs[g]
        )
        for g in game_configs
        if g in human_probs       # skip any config not in your CSV
    )


result_global_Reciprocity = minimize(
    total_nll_Reciprocity,
    x0 = [0.2, 0.2, 0.2],  
    bounds=[(0.00, 1),(0.00, 1),(0.00, 1)],
    method="L-BFGS-B",
)

global_params_Reciprocity = {"beta_player1": result_global_Reciprocity.x[0], "beta_player2": result_global_Reciprocity.x[1],
                      "theta_player2": result_global_Reciprocity.x[2]}

print("Self_Interested Level-K Betas:", global_betas_selfish)
print("Inequality-Aversion Level-K Parameters:", global_params_IA)
print("Reciprocity Level-K Parameters:", global_params_Reciprocity)


all_params = {
    "selfish":   global_betas_selfish,
    "inequality": global_params_IA,
    "reciprocity": global_params_Reciprocity
}

with open("fitted_params.json", "w") as fp:
    json.dump(all_params, fp)



