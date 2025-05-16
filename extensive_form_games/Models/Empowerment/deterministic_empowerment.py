from extensive_form_games.Games.sharing_game import SharingGame
import numpy as np

# Define the players
Player1 = "1"
Player2 = "2"



class GameEmpowermentCalculator:
    def __init__(self, game, horizon=1):
        self.game = game
        self.horizon = horizon
        self.empowerment_values = {}  # Store empowerment values for each player and state

    def compute_empowerment_for_player(self,player):
        """
        Compute the empowerment for a specific player, considering only the actions available
        to that player and 
        """
        # Start from the correct initial state for each player
        initial_state = self.get_player_initial_state(player)
        self.empowerment_values[player] = {}
        self._compute_empowerment_recursive(initial_state, player, self.horizon)


    def _compute_empowerment_recursive(self, state, player, steps_remaining):
        """
        Recursive function to calculate empowerment by exploring future reachable states.
        - state: current state in the game tree.
        - player: the player whose empowerment is being calculated.
        - steps_remaining: remaining number of steps in the horizon.
        """
        # If the state is terminal for the player or no more steps remaining, empowerment is 0
        if self.game.is_terminal(state, player) or steps_remaining == 0:
            self.empowerment_values[player][state] = 0
            return 0

        # If the empowerment value for this state has already been calculated, return it
        if state in self.empowerment_values[player]:
            return self.empowerment_values[player][state]

        reachable_states = set()

        # Check if it's the player's turn
        if self.game.get_player_turn(state) == player:
            # Explore all actions available to the current player in this state
            for action in self.game.get_actions(state, player):
                next_state = self.game.get_transition(state, action)
                if next_state is not None:
                    reachable_states.add(next_state)  # Only add reachable states, not sub-empowerments

        # Empowerment is the log of the number of distinct future states reachable from this state
        empowerment_value = np.log2(len(reachable_states)) if reachable_states else 0
        self.empowerment_values[player][state] = empowerment_value
        return empowerment_value


    
    # Instantiate the game and empowerment calculator

game = SharingGame()
empowerment_calculator = GameEmpowermentCalculator(game, horizon=3)

# Compute the avergae empowerment for both Player 1 and Player 2
empowerment_calculator.compute_empowerment_for_player(Player1)
empowerment_calculator.compute_empowerment_for_player(Player2)


# Print empowerment values for Player 2
print(f"Player 2's empowerment values: {empowerment_calculator.empowerment_values[Player2]}")
print(f"Player 1's empowerment values: {empowerment_calculator.empowerment_values[Player1]}")


# Compute the relative empowerment for both Player 1 and Player 2
relative_empowerment = empowerment_calculator.compute_empowerment_for_player(Player2) - empowerment_calculator.compute_empowerment_for_player(Player1)
if relative_empowerment > 0:
    print("Player 2 has more power than player 1")
else:
    print("Player 1 has more power than player 2")




