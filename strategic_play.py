import numpy as np
from sharing_game import SharingGame
from game_configs import game_configs

Player1 = "1"
Player2 = "2"

def softmax(x, beta):
    """ Apply softmax with an inverse temperature parameter (beta) to a list of values (expected utilities). """
    x = np.array(x)
    exp_x = np.exp(beta * x)
    return exp_x / np.sum(exp_x)  # Returns a choice probability for each action that is associated with a given expected utility (x)


class LevelKSimulation:
    def __init__(self, game, beta_player1, beta_player2,  transitions=None, rewards=None, actions=None, initial_state=None):
        """
        Initialize the simulation with different beta parameters for Player 1 and Player 2.
        :param game: Instance of the SharingGame class.
        :param beta_player1: Beta parameter for Player 1.
        :param beta_player2: Beta parameter for Player 2.
        """
        self.game = game
        self.beta_player1 = beta_player1  # Beta for Player 1
        self.beta_player2 = beta_player2  # Beta for Player 2
        self.action_probabilities = {Player1: {}, Player2: {}}  # Store action probabilities for each player

    def get_beta(self, player):
        """Return the beta parameter for the given player."""
        return self.beta_player1 if player == Player1 else self.beta_player2

    def simulate_level_0(self, player):
        """ Level 0 players choose actions uniformly at random. """
        for state in self.game.transitions.keys():
            actions = self.game.get_actions(state, player)
            if actions:
                prob = 1 / len(actions)
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
    print(f"Simulating {game_name}...")

    # Initialize the game for the current configuration
    game = SharingGame(
        transitions=config["transitions"],
        rewards=config["rewards"],
        actions=config["actions"],
        initial_state=config["initial_state"],
    )

    # Create a simulation object
    simulation = LevelKSimulation(
    game=game,
    beta_player1 = 0.33567815435117143,
    beta_player2 = 0.34834185166878495,
    transitions=transitions,
    rewards=rewards,
    actions=actions,
    initial_state=initial_state
)
    # Simulate Level 0 reasoning
    simulation.simulate_level_0(Player1)
    simulation.simulate_level_0(Player2)

    # Simulate Level 1 reasoning
    simulation.simulate_level_1(Player1, 1)
    simulation.simulate_level_1(Player2, 1)

    # Print Level 1 action probabilities for both players
    #print(f"Level 1 action probabilities for {game_name} - Player 1:")
    #print(simulation.action_probabilities[Player1])
    #print(f"Level 1 action probabilities for {game_name} - Player 2:")
    #print(simulation.action_probabilities[Player2])


    # Simulate Level 2 for Player 1 and Player 2
    simulation.simulate_level_2(Player1, game.get_initial_state())
    simulation.simulate_level_2(Player2, game.get_initial_state())

    # Extract and print probabilities of interest
    p_in_player1 = simulation.action_probabilities[Player1].get(1, {}).get("In", 0)
    p_right_player2 = simulation.action_probabilities[Player2].get(3, {}).get("Right", 0)

    print(f"P(In) for Player 1: {p_in_player1}")
    print(f"P(Right) for Player 2: {p_right_player2}")


