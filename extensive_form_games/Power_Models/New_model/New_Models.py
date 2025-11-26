import sys
import csv
import math

# Get the project root by going up two levels
sys.path.insert(0, '/Users/junior/Desktop/Files/MIT/Research/Projects/Reverse-engineering an Intuitive Theory of Power/Computational Models/extensive_form_games/Games')

from sharing_game import SharingGame
from new_game_configs import game_configs # Import new game configurations
from strategic_play import LevelKSimulation_Selfish

# Define players as strings to match game_configs keys
Player1 = "1"
Player2 = "2"

# 1. REU: Compute the Relative Expected Utility (of Player 2 relative to Player 1)
def compute_reu_direct(game, p_in_player1, p_right_in_player2, p_right_out_player2):
 
    "Your code here"

# 2. RCR: Compute the Relative Control over Resources (RCR).
def compute_rcr_direct(game, p_in_player1, p_right_in_player2, p_right_out_player2):
    
     "Your code here"

# 3. Choice: Compute the Relative Choice model (RC).
def compute_rc_direct(game, p_in_player1, p_right_in_player2, p_right_out_player2):

    "Your code here"

# 4. Counterfactual Causal Influence over Actions: Compute the Relative Control over Actions model (RCA).
def compute_rca_direct(game, p_in_player1, p_right_in_player2, p_right_out_player2):

    "Your code here"


# Define CSV file paths for each model
csv_file_path_reu = "reu_direct_results.csv"
csv_file_path_rcr = "rcr_direct_results.csv"
csv_file_path_rc = "rc_direct_results.csv"
csv_file_path_rca = "rca_direct_results.csv" # done for you


# Initialize dictionaries to store results for each model
results_reu = {}
results_rcr = {}
results_rc = {}
results_rca = {} # done for you

# Loop through all games in the game_configs dictionary
for game_name, config in game_configs.items():
    print(f"\nSimulating {game_name}...")

    # Initialize the game instance
    game = SharingGame(
        transitions=config["transitions"],
        rewards=config["rewards"],
        actions=config["actions"],
        initial_state=config["initial_state"]
    )

    # Initialize the simulation object with beta values
    simulation = LevelKSimulation_Selfish(
        game=game,
        beta_player1=0.33567815435117143,  # Global Best fitting beta values across 10 games.
        beta_player2=0.34834185166878495
    )

    # Simulate Level 0 reasoning (if needed)
    simulation.simulate_level_0(Player1)
    simulation.simulate_level_0(Player2)

    # Simulate Level 1 reasoning
    simulation.simulate_level_1(Player1, 1)
    simulation.simulate_level_1(Player2, 1)

    # Extract action probabilities for Level 1
    level1_p_in_player1 = simulation.action_probabilities[Player1].get(1, {}).get("In", 0)
    level1_p_right_player2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 0)

    # Simulate Level 2 reasoning
    simulation.simulate_level_2(Player1, game.get_initial_state())
    simulation.simulate_level_2(Player2, game.get_initial_state())

    # Extract action probabilities for Level 2
    p_in_player1 = simulation.action_probabilities[Player1].get(1, {}).get("In", 0)
    p_right_player2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 0)

    # Compute REU (Direct)
    EU_P1, EU_P2, REU_value = compute_reu_direct(game, p_in_player1, p_right_in_player2, p_right_out_player2)
    results_reu[game_name] = {"U_P1": round(EU_P1, 2), "U_P2": round(EU_P2, 2), "Relative Utility (P2 - P1)": round(REU_value, 2)}

    # Compute RCR (Direct)
    MaxU_P1, MaxU_P2, RCR = compute_rcr_direct(game, p_in_player1, p_right_in_player2, p_right_out_player2)
    results_rcr[game_name] = {"U_P1": round(MaxU_P1, 2), "U_P2": round(MaxU_P2, 2), "Relative Utility (P2 - P1)": round(RCR, 2)}

     # Compute RC (Direct)
    RC_P1, RC_P2, RC = compute_rc_direct(game, p_in_player1, p_right_in_player2, p_right_out_player2)
    results_rc[game_name] = {"U_P1": round(RC_P1, 2), "U_P2": round(RC_P2, 2), "Relative Utility (P2 - P1)": round(RC, 2)}

     # Compute RCA (Direct) (Done for you)
    RCA_P1, RCA_P2, RCA = compute_rca_direct(game, p_in_player1, p_right_in_player2, p_right_out_player2)

    # Print results for each model
    print(f"REU - Expected Utility of Player 1: {EU_P1:.2f}, Player 2: {EU_P2:.2f}, REU: {REU_value:.2f}")
    print(f"RCR - Relative Control over Resources for Player 1: {MaxU_P1:.2f}, Player 2: {MaxU_P2:.2f}, RCR: {RCR:.2f}")
    print(f"RC - Relative Choice for Player 1: {RC_P1:.2f}, Player 2: {RC_P2:.2f}, RCR: {RC:.2f}")
    print(f"RCA - Relative Control over Actions for Player 1: {RCA_P1:.2f}, Player 2: {RCA_P2:.2f}, RCA: {RCA:.2f}") #(done for you)

# Function to save results to CSV
def save_results_to_csv(file_path, results):
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Game", "U_P1", "U_P2", "Relative Utility (P2 - P1)"])
        for game, values in results.items():
            writer.writerow([game, values["U_P1"], values["U_P2"], values["Relative Utility (P2 - P1)"]])
    print(f"Results saved to {file_path}")

# Save each model's results 
save_results_to_csv(csv_file_path_reu, results_reu)
save_results_to_csv(csv_file_path_rcr, results_rcr)
save_results_to_csv(csv_file_path_rc, results_rc)
save_results_to_csv(csv_file_path_rca, results_rca)    #(done for you)


