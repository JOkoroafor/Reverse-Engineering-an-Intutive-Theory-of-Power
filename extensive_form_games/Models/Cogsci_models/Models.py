import sys
import csv
import math

# Get the project root by going up two levels
sys.path.insert(0, '/Users/junior/Desktop/Files/MIT/Research/Projects/Reverse-engineering an Intuitive Theory of Power/Computational Models/extensive_form_games/Games')

from sharing_game import SharingGame
from game_configs_choice import game_configs
from strategic_play import LevelKSimulation

# Define players as strings to match game_configs keys
Player1 = "1"
Player2 = "2"

### 1. REU
def compute_reu_direct(game, p_in_player1, p_right_player2):
    """Compute the Relative Expected Utility (of Player 2 relative to Player 1)."""
    # Extract rewards from the game config
    U_P1_Out = game.get_reward(2)[Player1]
    U_P2_Out = game.get_reward(2)[Player2]
    U_P1_Left = game.get_reward(4)[Player1]
    U_P2_Left = game.get_reward(4)[Player2]
    U_P1_Right = game.get_reward(5)[Player1]
    U_P2_Right = game.get_reward(5)[Player2]

    # Expected Utility Calculations using precomputed probabilities
    P_out = 1 - p_in_player1  
    P_in = p_in_player1
    P_left = 1 - p_right_player2  
    P_right = p_right_player2

    # Expected utilities for both players
    EU_P1 = U_P1_Out * P_out + U_P1_Left * P_in * P_left + U_P1_Right * P_in * P_right
    EU_P2 = U_P2_Out * P_out + U_P2_Left * P_in * P_left + U_P2_Right * P_in * P_right

    # Compute Relative Expected Utility
    relative_expected_utility = EU_P2 - EU_P1
    return EU_P1, EU_P2, relative_expected_utility

### 2. RCCR
def compute_rccr_direct(game, p_in_player1, p_right_player2):
    """Compute the Relative Control over Chance Resources (RCCR)."""
    # Extract rewards from the game config
    U_P1_Out = game.get_reward(2)[Player1]
    U_P2_Out = game.get_reward(2)[Player2]
    U_P1_Left = game.get_reward(4)[Player1]
    U_P2_Left = game.get_reward(4)[Player2]
    U_P1_Right = game.get_reward(5)[Player1]
    U_P2_Right = game.get_reward(5)[Player2]

    # Expected Utility Calculations using precomputed probabilities
    P_out = 1 - p_in_player1  
    P_in = p_in_player1
    P_left = 1 - p_right_player2  
    P_right = p_right_player2

    # Probabilities if Player 1 = coin flip
    P_out_P1_chance = 0.5
    P_in_P1_chance = 0.5
    P_right_P1_chance = P_right  # or consider 0.5 if you intend a uniform chance
    P_left_P1_chance = P_left    # or consider 0.5

    # Probabilities if Player 2 = coin flip
    # NOTE: level1_p_in_player1 should be computed from Level 1 simulation.
    # Ensure that you pass level1_p_in_player1 to this function if needed.
    P_in_P2chance = level1_p_in_player1  
    P_out_P2chance = 1 - level1_p_in_player1
    P_right_P2chance = 0.5
    P_left_P2chance = 0.5

    # Expected utilities for both players (direct calculation)
    EU_P1 = U_P1_Out * P_out + U_P1_Left * P_in * P_left + U_P1_Right * P_in * P_right
    EU_P2 = U_P2_Out * P_out + U_P2_Left * P_in * P_left + U_P2_Right * P_in * P_right

    # Expected utilities  players act by chance (that is if their action was replaced by a coin flip)
    EU_P1_chance = U_P1_Out * P_out_P1_chance + U_P1_Left * P_in_P1_chance * P_left_P1_chance + U_P1_Right * P_in_P1_chance * P_right_P1_chance
    EU_P2_chance = U_P2_Out * P_out_P2chance + U_P2_Left * P_in_P2chance * P_left_P2chance + U_P2_Right * P_in_P2chance * P_right_P2chance

    # Control over Chance Resources for each player
    CCR_P1 = ((EU_P1 - EU_P1_chance)/EU_P1_chance) 
    CCR_P2 = ((EU_P2 - EU_P2_chance)/EU_P2_chance)

    # Compute Relative Control over Chance Resources
    relative_control_over_chance_resources = CCR_P2 - CCR_P1
    return CCR_P1, CCR_P2, relative_control_over_chance_resources

### 3. RCR
def compute_rcr_direct(game, p_in_player1, p_right_player2):
    """Compute the Relative Control over Resources (RCR)."""
    # Extract rewards from the game config
    U_P1_Out = game.get_reward(2)[Player1]
    U_P2_Out = game.get_reward(2)[Player2]
    U_P1_Left = game.get_reward(4)[Player1]
    U_P2_Left = game.get_reward(4)[Player2]
    U_P1_Right = game.get_reward(5)[Player1]
    U_P2_Right = game.get_reward(5)[Player2]

    # Expected Utility Calculations using precomputed probabilities
    P_out = 1 - p_in_player1  
    P_in = p_in_player1
    P_left = 1 - p_right_player2  
    P_right = p_right_player2

    # Define choice sets for players
    P1_Choice_Set = [U_P1_Out, U_P1_Left, U_P1_Right]
    P2_Choice_Set = [U_P2_Out, U_P2_Left, U_P2_Right]
    MaxU_P1 = max(P1_Choice_Set)
    MaxU_P2 = max(P2_Choice_Set)

    EU_P1 = U_P1_Out * P_out + U_P1_Left * P_in * P_left + U_P1_Right * P_in * P_right
    EU_P2 = U_P2_Out * P_out + U_P2_Left * P_in * P_left + U_P2_Right * P_in * P_right

    RCR_P1 = (abs(MaxU_P2 - EU_P2) / MaxU_P2)
    RCR_P2 = (abs(MaxU_P1 - EU_P1) / MaxU_P1) 

    relative_control_over_resources = RCR_P2 - RCR_P1
    return RCR_P1, RCR_P2, relative_control_over_resources

### 4. Choice
def compute_rc_direct(game, p_in_player1, p_right_player2):
    """Compute the Relative Choice model (RC)."""
    # Extract rewards from the game config
    U_P1_Out = game.get_reward(2)[Player1]
    U_P2_Out = game.get_reward(2)[Player2]
    U_P1_Left = game.get_reward(4)[Player1]
    U_P2_Left = game.get_reward(4)[Player2]
    U_P1_Right = game.get_reward(5)[Player1]
    U_P2_Right = game.get_reward(5)[Player2]

    # Expected Utility Calculations using precomputed probabilities
    P_out = 1 - p_in_player1  
    P_in = p_in_player1
    P_left = 1 - p_right_player2  
    P_right = p_right_player2


    # Expected utilities for both players (direct calculation)
    EU_P1 = U_P1_Out * P_out + U_P1_Left * P_in * P_left + U_P1_Right * P_in * P_right
    EU_P2 = U_P2_Out * P_out + U_P2_Left * P_in * P_left + U_P2_Right * P_in * P_right

    P1_choice = 2
    P2_choice = (P_in) * (game.no_unique_states(Player2, 3))

    relative_choice = P2_choice -  P1_choice
    return P1_choice , P2_choice, relative_choice


# Define CSV file paths for each model
csv_file_path_reu = "reu_direct_results.csv"
csv_file_path_rccr = "rccr_direct_results.csv"
csv_file_path_rcr = "rcr_direct_results.csv"
csv_file_path_rc = "rc_direct_results.csv"


# Initialize dictionaries to store results for each model
results_reu = {}
results_rccr = {}   # <-- Use a separate dictionary for RCCR!
results_rcr = {}
results_rc = {}

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
    simulation = LevelKSimulation(
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
    EU_P1, EU_P2, REU_value = compute_reu_direct(game, p_in_player1, p_right_player2)
    results_reu[game_name] = {"U_P1": round(EU_P1, 2), "U_P2": round(EU_P2, 2), "Relative Utility (P2 - P1)": round(REU_value, 2)}

    # Compute RCCR (Direct) - make sure to pass level1_p_in_player1 if required by your model
    CCR_P1, CCR_P2, RCCR = compute_rccr_direct(game, p_in_player1, p_right_player2)
    results_rccr[game_name] = {"U_P1": round(CCR_P1, 2), "U_P2": round(CCR_P2, 2), "Relative Utility (P2 - P1)": round(RCCR, 2)}

    # Compute RCR (Direct)
    MaxU_P1, MaxU_P2, RCR = compute_rcr_direct(game, p_in_player1, p_right_player2)
    results_rcr[game_name] = {"U_P1": round(MaxU_P1, 2), "U_P2": round(MaxU_P2, 2), "Relative Utility (P2 - P1)": round(RCR, 2)}

     # Compute RC (Direct)
    RC_P1, RC_P2, RC = compute_rc_direct(game, p_in_player1, p_right_player2)
    results_rc[game_name] = {"U_P1": round(RC_P1, 2), "U_P2": round(RC_P2, 2), "Relative Utility (P2 - P1)": round(RC, 2)}

    # Print results for each model
    print(f"REU - Expected Utility of Player 1: {EU_P1:.2f}, Player 2: {EU_P2:.2f}, REU: {REU_value:.2f}")
    print(f"RCCR - Relative Control over Chance Resources for Player 1: {CCR_P1:.2f}, Player 2: {CCR_P2:.2f}, RCCR: {RCCR:.2f}")
    print(f"RCR - Relative Control over Resources for Player 1: {MaxU_P1:.2f}, Player 2: {MaxU_P2:.2f}, RCR: {RCR:.2f}")
    print(f"RC - Relative Choice for Player 1: {RC_P1:.2f}, Player 2: {RC_P2:.2f}, RCR: {RC:.2f}")


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
save_results_to_csv(csv_file_path_rccr, results_rccr)
save_results_to_csv(csv_file_path_rcr, results_rcr)
save_results_to_csv(csv_file_path_rc, results_rc)


