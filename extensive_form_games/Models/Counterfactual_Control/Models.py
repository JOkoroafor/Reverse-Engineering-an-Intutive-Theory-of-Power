import sys
import csv

# Get the project root by going up two levels
sys.path.insert(0, '/Users/junior/Desktop/Files/MIT/Research/Projects/Reverse-engineering an Intuitive Theory of Power/Computational Models/extensive_form_games/Games')

from sharing_game import SharingGame
from game_configs import game_configs
from strategic_play import LevelKSimulation

""" 
These class of models implements the theory that people's intuitive psychology of structural power concerns whether a persons could get another person to get something or do something that they otherwise would not have.
    
    The models are subdivded based on what they take to be the relevant state of affairs over which the player has influence.
        1. Behaviour: Relative Control over Behaviour (RCB)
    
        2. Desired outcomes: Relative Control over Desired Outcomes (RCR)
   
     The models are further subdivded based on what whether they consider what expected or max potential counterfactual influence a player has.
         A. Expected Counterfactual Causal Influence
         B. Max Potential Counterfactual Causal Influence

"""

# Define players as strings to match game_configs keys
Player1 = "1"
Player2 = "2"

### 1. RCB (Direct).
def compute_RCB_Expected(game, p_in_player1, p_right_player2):
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