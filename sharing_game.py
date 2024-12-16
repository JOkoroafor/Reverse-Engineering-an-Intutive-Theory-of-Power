Player1 = "1"
Player2 = "2"

class SharingGame:
    """
    Generalized multi-agent environment for sharing games.
    - Players: Player1 and Player2 (These games are defined as two player)
    - States: Configurable states and initial state
    - Actions: Configurable actions for each player in each state
    - Transitions: State transition function
    - Rewards: Payoffs for reaching certain terminal states
    """

    def __init__(self, transitions, rewards, actions, initial_state):
        """
        Initialize the game with game-specific configurations.
        :param transitions: State-action-state mappings.
        :param rewards: Rewards for terminal states.
        :param actions: Actions available for each player in each state.
        :param initial_state: The starting state of the game.
        """
        self.transitions = transitions
        self.rewards = rewards
        self.actions = actions
        self.initial_state = initial_state
    
    def is_player_terminal(self, state, player):
        """Check if the given state is terminal for the specific player."""
        if len(self.get_actions(state, player)) == 0:
            return True
        else:
            return False

    def player_non_terminal_states(self, player):
        """Get a list of non-terminal states for the player."""
        return [state for state in range(1, self.get_total_states() + 1) if not self.is_player_terminal(state, player)]

    def is_terminal(self, state):
        """Check if a state is terminal for the game (i.e., no player can act)."""
        if self.is_player_terminal(state, Player1) == True and self.is_player_terminal(state, Player2)== True:
            return True
        else:
            return False
    
    def get_players(self):
        return [Player1, Player2]
    
    def opponent(self, player):
        """Return the opponent of the given player."""
        return Player2 if player == Player1 else Player1

    def get_initial_state(self):
        """Return the initial state of the game."""
        return self.initial_state
    
    def get_total_states(self):
        """Return the total number of states in the game."""
        return 5

    def get_actions(self, state, player):
        """Get the actions available to the given player at a specific state."""
        return self.actions.get(state, {}).get(player, [])

    def get_transition(self, state, action):
        """Get the next state resulting from taking a given action in a specified state."""
        return self.transitions.get(state, {}).get(action, None)

    def get_reward(self, state):
        """Return the reward for each player at a specific state."""
        return self.rewards.get(state, {Player1: 0, Player2: 0})

    def get_player_utility(self, player, state):
        """Get the utility for a player at a specific state."""
        return self.get_reward(state).get(player, 0)
    