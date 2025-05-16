Player1 = "1"
Player2 = "2"

# Game configurations for all 10 games
game_configs = {
    "common_interest": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 5, Player2: 5},
            4: {Player1: 10, Player2: 10},
            5: {Player1: 0, Player2: 0},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "safe_shot": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 0, Player2: 0},
            4: {Player1: 5, Player2: 10},
            5: {Player1: 10, Player2: 5},
        },
        "actions": {1: {Player1: ["Out", "In"]},
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "strategic_dummy": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 2, Player2: 2},
            4: {Player1: 7, Player2: 7},
            5: {Player1: 7, Player2: 7},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "near_dictator": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 10, Player2: 3},
            4: {Player1: 7, Player2: 5},
            5: {Player1: 5, Player2: 7},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "costly_punish": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 5, Player2: 10},
            4: {Player1: 10, Player2: 5},
            5: {Player1: 3, Player2: 3},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "free_punish": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 5, Player2: 10},
            4: {Player1: 10, Player2: 5},
            5: {Player1: 3, Player2: 5},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "rational_punish": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 5, Player2: 10},
            4: {Player1: 3, Player2: 5},
            5: {Player1: 10, Player2: 3},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "costly_help": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 5, Player2: 5},
            4: {Player1: 10, Player2: 5},
            5: {Player1: 3, Player2: 10},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "free_help": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 10, Player2: 5},
            4: {Player1: 10, Player2: 10},
            5: {Player1: 5, Player2: 5},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "free_help_test": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 10, Player2: 5},
            4: {Player1: 10, Player2: 10},
            5: {Player1: 10, Player2: 5},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    },
    "trust_game": {
        "transitions": {1: {"Out": 2, "In": 3}, 
                        3: {"Left": 4, "Right": 5}},
        "rewards": {
            2: {Player1: 5, Player2: 3},
            4: {Player1: 10, Player2: 5},
            5: {Player1: 3, Player2: 10},
        },
        "actions": {1: {Player1: ["Out", "In"]}, 
                    3: {Player2: ["Left", "Right"]}},
        "initial_state": 1,
    }
    }