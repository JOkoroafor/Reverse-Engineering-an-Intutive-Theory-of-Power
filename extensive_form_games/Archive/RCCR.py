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
