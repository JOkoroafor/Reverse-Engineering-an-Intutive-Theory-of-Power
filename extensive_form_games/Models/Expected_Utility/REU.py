import sys
import csv
import math

# Get the project root by going up two levels
sys.path.insert(0, '/Users/junior/Desktop/Files/MIT/Research/Projects/Reverse-engineering an Intuitive Theory of Power/Computational Models/extensive_form_games/Games')

from sharing_game import SharingGame
from game_configs import game_configs
from strategic_play import LevelKSimulation

""" 
These class of models implements the theory that people's intuitive psychology of structural power concerns whether a player is in a position to get what they actually want.
    Relative Power of Person A wrt Person B = Expected Utitlity of Person A - Expected Utility of Person B.
    The expected utilities for both players are computed by a sum product of the magnitude of reward that a plyer receives at each state X the probability of reaching that state, enumerated over all states.

    The models are subdivded based on what magnitude of reward the represent at being achieved.
        1. REU (direct)
        2. GREU (uniform goal priors)
        3. REU (estimated goal priors)s
        4. REU (calibrated goal prior)

    The models are also subdivided best on how the 'probability of reaching each state' is estimated.
        A. Action Probailities generated from Level K model.
        B. Action Probabilities recovered from participant judgment.

    These models can also be subdivded best on expected relative utility or maximal potential relative utility
        A. Relative Expected Utility
        B. Max Potential Relative Utility.    
"""

# Define players as strings to match game_configs keys
Player1 = "1"
Player2 = "2"

### 1a. REU (Direct).
def compute_reu_direct(game, p_in_player1, p_right_player2):
    """This computes the Relative Expected Utility (of Player 2 relative to Player 1) with the assumption that direct payoffs in the games are what the players want ."""

    # Extract rewards from the game config
    U_P1_Out = game.get_reward(2)[Player1]
    U_P2_Out = game.get_reward(2)[Player2]
    U_P1_Left = game.get_reward(4)[Player1]
    U_P2_Left = game.get_reward(4)[Player2]
    U_P1_Right = game.get_reward(5)[Player1]
    U_P2_Right = game.get_reward(5)[Player2]

    # Expected Utility Calculations using precomputed probabilities
    P_out = 1 - p_in_player1  # Assuming Out is complementary to In
    P_in = p_in_player1
    P_left = 1 - p_right_player2  # Assuming Left is complementary to Right
    P_right = p_right_player2

    # Expected utilities for both players
    EU_P1 = U_P1_Out * P_out + U_P1_Left * P_in * P_left + U_P1_Right * P_in * P_right
    EU_P2 = U_P2_Out * P_out + U_P2_Left * P_in * P_left + U_P2_Right * P_in * P_right

    # Compute Relative Expected Utility
    relative_expected_utility = EU_P2 - EU_P1

    return EU_P1, EU_P2, relative_expected_utility

### 1a. RCCR (Direct).
def compute_rccr_direct(game, p_in_player1, p_right_player2):
    """This computes the Relative Expected Utility (of Player 2 relative to Player 1) with the assumption that direct payoffs in the games are what the players want ."""

    # Extract rewards from the game config
    U_P1_Out = game.get_reward(2)[Player1]
    U_P2_Out = game.get_reward(2)[Player2]
    U_P1_Left = game.get_reward(4)[Player1]
    U_P2_Left = game.get_reward(4)[Player2]
    U_P1_Right = game.get_reward(5)[Player1]
    U_P2_Right = game.get_reward(5)[Player2]

    # Expected Utility Calculations using precomputed probabilities
    P_out = 1 - p_in_player1  # Assuming Out is complementary to In
    P_in = p_in_player1
    P_left = 1 - p_right_player2  # Assuming Left is complementary to Right
    P_right = p_right_player2

    #Probabilities if Player 1 = coin flip
    P_out_P1_chance = 0.5
    P_in_P1_chance = 0.5
    P_right_P1_chance = P_right
    P_left_P1_chance = P_left

    # Probabilities if Player 2 = coin flip
    P_in_P2chance = level1_p_in_player1
    P_out_P2chance = 1-level1_p_in_player1
    P_right_P2chance = 0.5
    P_left_P2chance = 0.5

    # Expected utilities for both players
    EU_P1 = U_P1_Out * P_out + U_P1_Left * P_in * P_left + U_P1_Right * P_in * P_right
    EU_P2 = U_P2_Out * P_out + U_P2_Left * P_in * P_left + U_P2_Right * P_in * P_right

    # Expected utilities for 
    EU_P1_chance = U_P1_Out * P_out_P1_chance  + U_P1_Left * P_in_P1_chance * P_left_P1_chance + U_P1_Right * P_in_P1_chance * P_right_P1_chance
    EU_P2_chance = U_P2_Out * P_out_P2chance + U_P2_Left * P_in_P2chance * P_left_P2chance + U_P2_Right * P_in_P2chance * P_right_P2chance

    #Power to for P1 and P2
    CCR_P1 = EU_P1 - EU_P1_chance
    CCR_P2 = EU_P2 - EU_P2_chance


    # Compute Relative Expected Utility
    relative_control_over_chance_resources = CCR_P2 - CCR_P1

    return CCR_P1, CCR_P2, relative_control_over_chance_resources

### 1c. RCR (Direct).
def compute_rcr_direct(game, p_in_player1, p_right_player2):
    """This computes the Relative Expected Utility (of Player 2 relative to Player 1) with the assumption that direct payoffs in the games are what the players want ."""

    # Extract rewards from the game config
    U_P1_Out = game.get_reward(2)[Player1]
    U_P2_Out = game.get_reward(2)[Player2]
    U_P1_Left = game.get_reward(4)[Player1]
    U_P2_Left = game.get_reward(4)[Player2]
    U_P1_Right = game.get_reward(5)[Player1]
    U_P2_Right = game.get_reward(5)[Player2]

    # Expected Utility Calculations using precomputed probabilities
    P_out = 1 - p_in_player1  # Assuming Out is complementary to In
    P_in = p_in_player1
    P_left = 1 - p_right_player2  # Assuming Left is complementary to Right
    P_right = p_right_player2

    # Define choice set for players
    P1_Choice_Set = [U_P1_Out, U_P1_Left, U_P1_Right]
    P2_Choice_Set = [U_P2_Out, U_P2_Left, U_P2_Right]
    # Expected utilities for both players
    MaxU_P1 = max(P1_Choice_Set)
    MaxU_P2 = max(P2_Choice_Set)

    EU_P1 = U_P1_Out * P_out + U_P1_Left * P_in * P_left + U_P1_Right * P_in * P_right
    EU_P2 = U_P2_Out * P_out + U_P2_Left * P_in * P_left + U_P2_Right * P_in * P_right

    RCR_P1 = abs(MaxU_P2 - EU_P2)/MaxU_P2 
    RCR_P2 = abs(MaxU_P1 - EU_P1)/MaxU_P1

    # Compute Relative Expected Utility
    relative_control_over_resources = RCR_P2 - RCR_P1

    return RCR_P1, RCR_P2, relative_control_over_resources


### 1b. Maximal Potential Relative Utlity (Direct).
def compute_mpru_direct(game, p_in_player1, p_right_player2):
    """This computes the biggest difference in (positive) utility that could be attained if a player so wanted too. With the assumption that direct payoffs in the games are what the players want.
    P(Act) * arg Max (RU)"""

    # Extract rewards from the game config
    U_P1_Out = game.get_reward(2)[Player1]
    U_P2_Out = game.get_reward(2)[Player2]
    U_P1_Left = game.get_reward(4)[Player1]
    U_P2_Left = game.get_reward(4)[Player2]
    U_P1_Right = game.get_reward(5)[Player1]
    U_P2_Right = game.get_reward(5)[Player2]

    # Compute the relative positive utility for each player at each terminal state.
    RU_P1_Out = U_P1_Out - U_P2_Out
    RU_P2_Out = U_P2_Out - U_P1_Out
    RU_P1_Left = U_P1_Left - U_P2_Left
    RU_P2_Left = U_P2_Left - U_P1_Left
    RU_P1_Right = U_P1_Right - U_P2_Right
    RU_P2_Right = U_P2_Right - U_P1_Right  

     # Want to have some representation of the possible outcomes that a player could choose. To do this, make a list of the terminal states that could reached via each players unliteral action
    Possible_RU_P1 = [RU_P1_Out]
    Possible_RU_P2 = [RU_P2_Left, RU_P2_Right]

    # Expected Utility Calculations using precomputed probabilities
    P_out = 1 - p_in_player1  # Assuming Out is complementary to In
    P_in = p_in_player1
    P_left = 1 - p_right_player2  # Assuming Left is complementary to Right
    P_right = p_right_player2


    # Expected utilities for both players
    MPU_P1 = max(Possible_RU_P1)
    MPU_P2 = P_in * max(Possible_RU_P2)

    # Compute Relative Expected Utility
    maximal_potential_relative_utility = MPU_P2 - MPU_P1 

    return MPU_P1, MPU_P2, maximal_potential_relative_utility

### 2. GREU.
def compute_greu(game, p_in_player1, p_right_player2):
    """This computes the relative expected utility (P2 - P1) under all possible goals that each player might have"""
    # Extract Direct rewards
     # Extract rewards from the game config
    U_P1_Out = game.get_reward(2)[Player1]
    U_P2_Out = game.get_reward(2)[Player2]
    U_P1_Left = game.get_reward(4)[Player1]
    U_P2_Left = game.get_reward(4)[Player2]
    U_P1_Right = game.get_reward(5)[Player1]
    U_P2_Right = game.get_reward(5)[Player2]

    # Expected Utility Calculations using precomputed probabilities
    P_out = 1 - p_in_player1  # Assuming Out is complementary to In
    P_in = p_in_player1
    P_left = 1 - p_right_player2  # Assuming Left is complementary to Right
    P_right = p_right_player2

    "Altruism: Ua = Uother"
    Ua_P1_Out = U_P2_Out
    Ua_P2_Out =  U_P1_Out
    Ua_P1_Left = U_P2_Left 
    Ua_P2_Left = U_P1_Left
    Ua_P1_Right = U_P2_Right
    Ua_P2_Right = U_P1_Right

    # Expected utilities for both players
    EUa_P1 = Ua_P1_Out * P_out + Ua_P1_Left * P_in * P_left + Ua_P1_Right * P_in * P_right
    EUa_P2 = Ua_P2_Out * P_out + Ua_P2_Left * P_in * P_left + Ua_P2_Right * P_in * P_right

    "Prosocial: Up = 1/2 Uself + 1/2 Uother"
    # Extract rewards from the game config but modify for 5 distinct goals
    Up_P1_Out = (1/2) * (U_P1_Out + U_P2_Out)
    Up_P2_Out = (1/2) * (U_P2_Out + U_P1_Out)
    Up_P1_Left = (1/2) * (U_P1_Left + U_P2_Left)
    Up_P2_Left = (1/2) * (U_P2_Left + U_P1_Left)
    Up_P1_Right = (1/2) * (U_P1_Right + U_P2_Right)
    Up_P2_Right = (1/2) * (U_P2_Right + U_P1_Right)


    # Expected utilities for both players
    EUp_P1 = Up_P1_Out * P_out + Up_P1_Left * P_in * P_left + Up_P1_Right * P_in * P_right
    EUp_P2 = Up_P2_Out * P_out + Up_P2_Left * P_in * P_left + Up_P2_Right * P_in * P_right

    "Selish: Usel = Uself"
    # Extract rewards from the game config but modify for 5 distinct goals
    Usel_P1_Out = U_P1_Out
    Usel_P2_Out =  U_P2_Out
    Usel_P1_Left = U_P1_Left 
    Usel_P2_Left = U_P2_Left
    Usel_P1_Right = U_P1_Right
    Usel_P2_Right = U_P2_Right

    # Expected utilities for both players
    EUsel_P1 = Usel_P1_Out * P_out + Usel_P1_Left * P_in * P_left + Usel_P1_Right * P_in * P_right
    EUsel_P2 = Usel_P2_Out * P_out + Usel_P2_Left * P_in * P_left + Usel_P2_Right * P_in * P_right

    "Competitive: Uc = 1/2 Uself - 1/2 Uother"
    # Extract rewards from the game config but modify for 5 distinct goals
    Uc_P1_Out =  (1/2) * (U_P1_Out - U_P2_Out)
    Uc_P2_Out =  (1/2) * (U_P2_Out - U_P1_Out)
    Uc_P1_Left = (1/2) * (U_P1_Left - U_P2_Left)
    Uc_P2_Left = (1/2) * (U_P2_Left - U_P1_Left)
    Uc_P1_Right =(1/2) * (U_P1_Right - U_P2_Right)
    Uc_P2_Right =(1/2) * (U_P2_Right - U_P1_Right)

    # Expected utilities for both players
    EUc_P1 = Uc_P1_Out * P_out + Uc_P1_Left * P_in * P_left + Uc_P1_Right * P_in * P_right
    EUc_P2 = Uc_P2_Out * P_out + Uc_P2_Left * P_in * P_left + Uc_P2_Right * P_in * P_right

    "Sadistic: Us = -Uother"
    # Extract rewards from the game config but modify for 5 distinct goals
    Usad_P1_Out = -U_P2_Out
    Usad_P2_Out =  -U_P1_Out
    Usad_P1_Left = -U_P2_Left 
    Usad_P2_Left = -U_P1_Left
    Usad_P1_Right = -U_P2_Right
    Usad_P2_Right = -U_P1_Right

    # Expected utilities for both players
    EUsad_P1 = Usad_P1_Out * P_out + Usad_P1_Left * P_in * P_left + Usad_P1_Right * P_in * P_right
    EUsad_P2 = Usad_P2_Out * P_out + Usad_P2_Left * P_in * P_left + Usad_P2_Right * P_in * P_right

    "Average Expected Utility under possible goals"
    EU_P1 = EUa_P1 + EUp_P1 + EUsel_P1 + EUc_P1 +EUsad_P1
    EU_P2 = EUa_P2 + EUp_P2 + EUsel_P2 + EUc_P2 +EUsad_P2

    # Compute Relative Expected Utility
    relative_expected_utility = EU_P2 - EU_P1

    return EU_P1, EU_P2, relative_expected_utility

### 3. GREU (Estimated Goal Priors).
def compute_greu_estimated(game, p_in_player1, p_right_player2):
    """This computes the relative expected utility (P2 - P1) under likely goals for each player. This merely posits what the likely """
    "The model posits that observes will think that players are most likely selfish(0.7), less likely prosocial (0.2), and even less likely altrustic (0.1) but never competetive (0) or sadistic (0)"
    "The subjective utilities under possible goals are unchanged from GREU"
    # Extract Direct rewards
     # Extract rewards from the game config
    U_P1_Out = game.get_reward(2)[Player1]
    U_P2_Out = game.get_reward(2)[Player2]
    U_P1_Left = game.get_reward(4)[Player1]
    U_P2_Left = game.get_reward(4)[Player2]
    U_P1_Right = game.get_reward(5)[Player1]
    U_P2_Right = game.get_reward(5)[Player2]

    # Expected Utility Calculations using precomputed probabilities
    P_out = 1 - p_in_player1  # Assuming Out is complementary to In
    P_in = p_in_player1
    P_left = 1 - p_right_player2  # Assuming Left is complementary to Right
    P_right = p_right_player2



    "Altruism: Ua = Uother"
    # Extract rewards from the game config but modify for 5 distinct goals
    Ua_P1_Out = U_P2_Out
    Ua_P2_Out =  U_P1_Out
    Ua_P1_Left = U_P2_Left 
    Ua_P2_Left = U_P1_Left
    Ua_P1_Right = U_P2_Right
    Ua_P2_Right = U_P1_Right

    # Expected utilities for both players
    EUa_P1 = Ua_P1_Out * P_out + Ua_P1_Left * P_in * P_left + Ua_P1_Right * P_in * P_right
    EUa_P2 = Ua_P2_Out * P_out + Ua_P2_Left * P_in * P_left + Ua_P2_Right * P_in * P_right

    "Prosocial: Up = 1/2 Uself + 1/2 Uother"
    # Extract rewards from the game config but modify for 5 distinct goals
    Up_P1_Out = (1/2) * (U_P1_Out + U_P2_Out)
    Up_P2_Out =  (1/2) *  (U_P2_Out + U_P1_Out)
    Up_P1_Left = (1/2) *  (U_P1_Left + U_P2_Left)
    Up_P2_Left = (1/2) *  (U_P2_Left + U_P1_Left)
    Up_P1_Right = (1/2) *  (U_P1_Right + U_P2_Right)
    Up_P2_Right = (1/2) *  (U_P2_Right + U_P1_Right)

    # Expected utilities for both players
    EUp_P1 = Up_P1_Out * P_out + Up_P1_Left * P_in * P_left + Up_P1_Right * P_in * P_right
    EUp_P2 = Up_P2_Out * P_out + Up_P2_Left * P_in * P_left + Up_P2_Right * P_in * P_right

    "Selish: Usel = Uself"
    # Extract rewards from the game config but modify for 5 distinct goals
    Usel_P1_Out = U_P1_Out
    Usel_P2_Out =  U_P2_Out
    Usel_P1_Left = U_P1_Left 
    Usel_P2_Left = U_P2_Left
    Usel_P1_Right = U_P1_Right
    Usel_P2_Right = U_P2_Right

    # Expected utilities for both players
    EUsel_P1 = Usel_P1_Out * P_out + Usel_P1_Left * P_in * P_left + Usel_P1_Right * P_in * P_right
    EUsel_P2 = Usel_P2_Out * P_out + Usel_P2_Left * P_in * P_left + Usel_P2_Right * P_in * P_right

    "Competitive: Uc = 1/2 Uself - 1/2 Uother"
    # Extract rewards from the game config but modify for 5 distinct goals
    Uc_P1_Out = (1/2) *  (U_P1_Out - U_P2_Out)
    Uc_P2_Out =  (1/2) *  (U_P2_Out - U_P1_Out)
    Uc_P1_Left = (1/2) *  (U_P1_Left - U_P2_Left)
    Uc_P2_Left = (1/2) * (U_P2_Left - U_P1_Left)
    Uc_P1_Right = (1/2) *  (U_P1_Right - U_P2_Right)
    Uc_P2_Right = (1/2) * (U_P2_Right - U_P1_Right)

    # Expected utilities for both players
    EUc_P1 = Uc_P1_Out * P_out + Uc_P1_Left * P_in * P_left + Uc_P1_Right * P_in * P_right
    EUc_P2 = Uc_P2_Out * P_out + Uc_P2_Left * P_in * P_left + Uc_P2_Right * P_in * P_right

    "Sadistic: Us = -Uother"
    # Extract rewards from the game config but modify for 5 distinct goals
    Usad_P1_Out = -U_P2_Out
    Usad_P2_Out =  -U_P1_Out
    Usad_P1_Left = -U_P2_Left 
    Usad_P2_Left = -U_P1_Left
    Usad_P1_Right = -U_P2_Right
    Usad_P2_Right = -U_P1_Right

    # Expected utilities for both players
    EUsad_P1 = Usad_P1_Out * P_out + Usad_P1_Left * P_in * P_left + Usad_P1_Right * P_in * P_right
    EUsad_P2 = Usad_P2_Out * P_out + Usad_P2_Left * P_in * P_left + Usad_P2_Right * P_in * P_right

    "The model posits that observes will think that players are most likely selfish(0.7), less likely prosocial (0.2), and even less likely altrustic (0.1) but never competetive (0) or sadistic (0)"
    "Average Expected Utility under plausible goals implements these as goal priors"
    EU_P1 = ((0.1*EUa_P1) + (.2*EUp_P1) + (0.7*EUsel_P1) + (0*EUc_P1)+(0*EUsad_P1))
    EU_P2 = ((0.1*EUa_P2) + (.2*EUp_P2) + (0.7*EUsel_P2) + (0*EUc_P2)+(0*EUsad_P2))

    # Compute Relative Expected Utility
    relative_expected_utility = EU_P2 - EU_P1

    return EU_P1, EU_P2, relative_expected_utility


# Define CSV file paths for each model
csv_file_path_reu = "reu_direct_results.csv"
csv_file_path_rccr = "rccr_direct_results.csv"
csv_file_path_rcr = "rcr_direct_results.csv"
csv_file_path_mpru = "mpru_direct_results.csv"
csv_file_path_greu = "greu_results.csv"
csv_file_path_greu_estimated = "greu_estimated_results.csv"

# Initialize dictionaries to store results for each model
results_reu = {}
results_rccr = {}
results_rcr = {}
results_mpru = {}
results_greu = {}
results_greu_estimated = {}

# Loop through all games in the game_configs dictionary
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

# Simulate Level 0 reasoning
    simulation.simulate_level_0(Player1)
    simulation.simulate_level_0(Player2)
     # Print Level 0 action probabilities for both players
   #  print(f"Level 0 action probabilities for {game_name} - Player 1:")
   #  print(simulation.action_probabilities[Player1])
   #  print(f"Level 0 action probabilities for {game_name} - Player 2:")
   #  print(simulation.action_probabilities[Player2])

    # Simulate Level 1 reasoning
    simulation.simulate_level_1(Player1, 1)
    simulation.simulate_level_1(Player2, 1)

    # Extract action probabilities for Level 1
    level1_p_in_player1 = simulation.action_probabilities[Player1].get(1, {}).get("In", 0)
    level1_p_right_player2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 0)

    print(f"Level 1 Probabilities - Player 1 (P(In)): {level1_p_in_player1:.3f}")
    print(f"Level 1 Probabilities - Player 2 (P(Right)): {level1_p_right_player2:.3f}")

    # --- 🔹 Simulate Level 2 ---
    simulation.simulate_level_2(Player1, game.get_initial_state())
    simulation.simulate_level_2(Player2, game.get_initial_state())

    # Extract action probabilities for Level 2
    p_in_player1 = simulation.action_probabilities[Player1].get(1, {}).get("In", 0)
    p_right_player2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 0)

    print(f"Level 2 Probabilities - Player 1 (P(In)): {p_in_player1:.3f}")
    print(f"Level 2 Probabilities - Player 2 (P(Right)): {p_right_player2:.3f}")

    # Extract action probabilities
    p_in_player1 = simulation.action_probabilities[Player1].get(1, {}).get("In", 0)
    p_right_player2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 0)

    # Compute REU (Direct)
    EU_P1, EU_P2, REU = compute_reu_direct(game, p_in_player1, p_right_player2)
    results_reu[game_name] = {"U_P1": round(EU_P1, 2), "U_P2": round(EU_P2, 2), "Relative Utility (P2 - P1)": round(REU, 2)}

    # Compute RCCR (Direct)
    CCR_P1, CCR_P2, RCCR = compute_rccr_direct(game, p_in_player1, p_right_player2)
    results_reu[game_name] = {"U_P1": round(CCR_P1, 2), "U_P2": round(CCR_P2, 2), "Relative Utility (P2 - P1)": round(RCCR, 2)}

    # Compute RCR (Direct)
    MaxU_P1, MaxU_P2, RCR = compute_rcr_direct(game, p_in_player1, p_right_player2)
    results_rcr[game_name] = {"U_P1": round(MaxU_P1, 2), "U_P2": round(MaxU_P2, 2), "Relative Utility (P2 - P1)": round(RCR, 2)}

    # Compute MPRU (Direct)
    MPRU_P1, MPRU_P2, MPRU = compute_mpru_direct(game, p_in_player1, p_right_player2)
    results_mpru[game_name] = {"U_P1": round(MPRU_P1, 2), "U_P2": round(MPRU_P2, 2), "Relative Utility (P2 - P1)": round(MPRU, 2)}

    # Compute GREU (Uniform Goal Priors)
    EU_P1_greu, EU_P2_greu, GREU = compute_greu(game, p_in_player1, p_right_player2)
    results_greu[game_name] = {"U_P1": round(EU_P1_greu, 2), "U_P2": round(EU_P2_greu, 2), "Relative Utility (P2 - P1)": round(GREU, 2)}

    # Compute GREU (Estimated Goal Priors)
    EU_P1_greu_est, EU_P2_greu_est, GREU_estimated = compute_greu_estimated(game, p_in_player1, p_right_player2)
    results_greu_estimated[game_name] = {"U_P1": round(EU_P1_greu_est, 2), "U_P2": round(EU_P2_greu_est, 2), "Relative Utility (P2 - P1)": round(GREU_estimated, 2)}

    # Print results for each model
    print(f"REU - Expected Utility of Player 1: {EU_P1:.2f}, Player 2: {EU_P2:.2f}, REU: {REU:.2f}")
    print(f"RCCR - Relative Control over Chance Resources for Player 1: {CCR_P1:.2f}, Player 2: {CCR_P2:.2f}, RCCR: {RCCR:.2f}")
    print(f"RCR - Relative Control over Resources for Player 1: {MaxU_P1:.2f}, Player 2: {MaxU_P2:.2f}, RCR: {RCR:.2f}")
     #print(f"MPRU - Maximal Potential Relative Utility for Player 1: {MPRU_P1:.2f}, Player 2: {MPRU_P2:.2f}, MPRU: {MPRU:.2f}")
     #print(f"GREU - Expected Utility of Player 1: {EU_P1_greu:.2f}, Player 2: {EU_P2_greu:.2f}, GREU: {GREU:.2f}")
     #print(f"GREU Estimated - Expected Utility of Player 1: {EU_P1_greu_est:.2f}, Player 2: {EU_P2_greu_est:.2f}, GREU Estimated: {GREU_estimated:.2f}")

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
save_results_to_csv(csv_file_path_mpru, results_mpru)
save_results_to_csv(csv_file_path_greu, results_greu)
save_results_to_csv(csv_file_path_greu_estimated, results_greu_estimated)









