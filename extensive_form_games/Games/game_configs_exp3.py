Player1 = "1"
Player2 = "2"

# Game configurations for all 10 games
game_configs = {
    "common_interest_A": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 5, Player2: 5},
            4: {Player1: 10, Player2: 10},
            5: {Player1: 10, Player2: 10},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "common_interest_B": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 0, Player2: 0},
            4: {Player1: 10, Player2: 10},
            5: {Player1: 0, Player2: 0},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "safe_shot_A": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 0, Player2: 0},
            4: {Player1: 5, Player2: 10},
            5: {Player1: 5, Player2: 10},
        },
        "actions": {1: {Player1: ["Out", "In"]},
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "strategic_dummy_A": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 8, Player2: 8},
            4: {Player1: 7, Player2: 7},
            5: {Player1: 7, Player2: 7},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "strategic_dummy_B": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 2, Player2: 2},
            4: {Player1: 3, Player2: 3},
            5: {Player1: 7, Player2: 7},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "near_dictator_B": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 10, Player2: 3},
            4: {Player1: 15, Player2: 10},
            5: {Player1: 5, Player2: 7},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    
    "rational_punish_A": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 10, Player2: 15},
            4: {Player1: 3, Player2: 5},
            5: {Player1: 10, Player2: 3},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "rational_punish_B": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 2, Player2: 7},
            4: {Player1: 3, Player2: 5},
            5: {Player1: 10, Player2: 3},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "costly_help_B": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 3, Player2: 3},
            4: {Player1: 10, Player2: 5},
            5: {Player1: 3, Player2: 10},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "costly_help_A": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 5, Player2: 5},
            4: {Player1: 3, Player2: 10},
            5: {Player1: 3, Player2: 10},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    }}