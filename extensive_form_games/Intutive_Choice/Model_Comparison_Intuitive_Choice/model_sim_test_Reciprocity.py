import sys,json
# Ensure your project Games folder is on the path
sys.path.insert(0, '/Users/junior/Desktop/Files/MIT/Research/Projects/Reverse-engineering an Intuitive Theory of Power/Computational Models/extensive_form_games/Games')
import pandas as pd
from sharing_game import SharingGame
from strategic_play_reciprocity import LevelKSimulation_Selfish, LevelKSimulation_IA, LevelKSimulation_Recip_Parameter
from game_configs_exp3 import game_configs  # Test Games


Player1 = "1"
Player2 = "2"

# 1) Load the fitted parameters for models
with open("fitted_params.json", "r") as fp:
    params = json.load(fp)

BETA_SELFISH = params["selfish"]
PARAMS_IA      = params["inequality"]
PARAMS_Reciprocity = params["reciprocity"]

# Prepare a list to collect model simulations
results = []

# --- 2) Loop through each new game and simulate both models ---
for game_name, cfg in game_configs.items():
    game = SharingGame(
        transitions   = cfg["transitions"],
        rewards       = cfg["rewards"],
        actions       = cfg["actions"],
        initial_state = cfg["initial_state"],
    )
    
    # 2a) Selfish Model Prediction
    sim_s = LevelKSimulation_Selfish(
        game, 
        BETA_SELFISH["beta_player1"], 
        BETA_SELFISH["beta_player2"]
    )

    sim_s.simulate_level_0(Player1)
    sim_s.simulate_level_0(Player2)
    sim_s.simulate_level_1(Player1, game.get_initial_state())
    sim_s.simulate_level_1(Player2, game.get_initial_state())
    sim_s.simulate_level_2(Player1, game.get_initial_state())
    sim_s.simulate_level_2(Player2, game.get_initial_state())
    p_in_s   = sim_s.action_probabilities[Player1][1].get("In", None)
    p_right_s = sim_s.action_probabilities[Player2][3].get("Right", None)
    
    results.append({
        "Game":      game_name,
        "Model":     "Selfish",
        "p_in":      p_in_s,
        "p_right":   p_right_s
    })
    
    # 2b) Inequality-Aversion Model Prediction
    sim_ia = LevelKSimulation_IA(
        game, 
        PARAMS_IA ["beta_player1"], 
        PARAMS_IA ["beta_player2"],
        PARAMS_IA ["delta_player1"], 
        PARAMS_IA ["delta_player2"],
        PARAMS_IA ["alpha_player1"], 
        PARAMS_IA ["alpha_player2"]
    )
    sim_ia.simulate_level_0(Player1)
    sim_ia.simulate_level_0(Player2)
    sim_ia.simulate_level_1(Player1, game.get_initial_state())
    sim_ia.simulate_level_1(Player2, game.get_initial_state())
    sim_ia.simulate_level_2(Player1, game.get_initial_state())
    sim_ia.simulate_level_2(Player2, game.get_initial_state())
    p_in_ia    = sim_ia.action_probabilities[Player1][1].get("In", None)
    p_right_ia = sim_ia.action_probabilities[Player2][3].get("Right", None)
    
    results.append({
        "Game":      game_name,
        "Model":     "InequalityAversion",
        "p_in":      p_in_ia,
        "p_right":   p_right_ia
    })

 # 2c) Reciprocity Model Prediction
    sim_reciprocity = LevelKSimulation_Recip_Parameter(
        game, 
        PARAMS_Reciprocity ["beta_player1"], 
        PARAMS_Reciprocity ["beta_player2"],
        PARAMS_Reciprocity ["theta_player2"]
    )
    sim_reciprocity.simulate_level_0(Player1)
    sim_reciprocity.simulate_level_0(Player2)
    sim_reciprocity.simulate_level_1(Player1, game.get_initial_state())
    sim_reciprocity.simulate_level_1(Player2, game.get_initial_state())
    sim_reciprocity.simulate_level_2(Player1, game.get_initial_state())
    sim_reciprocity.simulate_level_2(Player2, game.get_initial_state())
    p_in_reciprocity    = sim_reciprocity.action_probabilities[Player1][1].get("In", None)
    p_right_reciprocity = sim_reciprocity.action_probabilities[Player2][3].get("Right", None)
    
    results.append({
        "Game":      game_name,
        "Model":     "Reciprocity",
        "p_in":      p_in_reciprocity,
        "p_right":   p_right_reciprocity
    })
# --- 3) Compile results into a table and save/print ---
df_results = pd.DataFrame(results)
print(df_results)

# Optionally, save to CSV:
df_results.to_csv("test_2_model_predictions.csv", index=False)

