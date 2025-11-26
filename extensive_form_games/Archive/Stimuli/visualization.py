import sys
import csv
import math

# Get the project root by going up two levels
sys.path.insert(0, '/Users/junior/Desktop/Files/MIT/Research/Projects/Reverse-engineering an Intuitive Theory of Power/Computational Models/extensive_form_games/Games')

from game_configs_test import game_configs

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch


def visualize_games(game_configs, game_name=None):
    """Visualizes all games in game_configs by default. If a game_name is provided, only that game is visualized."""

    games_to_plot = [game_name] if game_name else game_configs.keys()  # Plot all games if none specified

    for game_name in games_to_plot:
        # Ensure the game exists
        if game_name not in game_configs:
            print(f"Error: Game '{game_name}' not found in game_configs.")
            continue

        print(f"\n📌 Visualizing {game_name}...")

        # Load the game configuration
        game = game_configs[game_name]

        # Extract components
        transitions = game["transitions"]
        rewards = game["rewards"]

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes for game states
        G.add_node("Player 1", label="Player 1")
        G.add_node("Player 2", label="Player 2")

        # Terminal states (T1, T2, T3) with dynamically updated rewards
        T1_payoff = f"${rewards[2]['1']} , ${rewards[2]['2']}" if 2 in rewards else "$?, $?"
        T2_payoff = f"${rewards[4]['1']} , ${rewards[4]['2']}" if 4 in rewards else "$?, $?"
        T3_payoff = f"${rewards[5]['1']} , ${rewards[5]['2']}" if 5 in rewards else "$?, $?"

        G.add_node("T1", label=T1_payoff)
        G.add_node("T2", label=T2_payoff)
        G.add_node("T3", label=T3_payoff)

        # Add edges for the choices
        G.add_edge("Player 1", "T1", label="Out")
        G.add_edge("Player 1", "Player 2", label="In")
        G.add_edge("Player 2", "T2", label="Left")
        G.add_edge("Player 2", "T3", label="Right")

        # Position the nodes manually for a tree layout
        pos = {
            "Player 1": (0, 2),
            "Player 2": (1, 1), 
            "T2": (0.75, 0),
            "T1": (-1, 1),
            "T3": (1.25, 0)
        }

        # Adjust node spacing manually
        distance_offset = abs(pos['Player 2'][0] - pos['T3'][0])  # Use distance between Player 2 and T3
        pos['Player 1'] = (pos['Player 2'][0] - distance_offset, pos['Player 1'][1])  # Shift Player 1 left
        pos['T1'] = (pos['Player 1'][0] - distance_offset, pos['T1'][1])  # Shift T1 left

        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, node_size=7000, node_color="white", edgecolors="black")

        # Draw the edges
        nx.draw_networkx_edges(G, pos, arrowsize=30)

        # Manually add arrows using FancyArrowPatch
        ax = plt.gca()  # Get the current axis
        for (start, end) in G.edges():
            start_pos = pos[start]
            end_pos = pos[end]
            arrow = FancyArrowPatch(
                posA=start_pos, posB=end_pos,
                arrowstyle='-|>', color='black', mutation_scale=20, lw=1.5,
                shrinkA=15, shrinkB=15
            )
            ax.add_patch(arrow)

        # Draw node labels with colors
        nx.draw_networkx_labels(G, pos, labels={"Player 1": "Player 1"}, font_size=15, font_color="blue")
        nx.draw_networkx_labels(G, pos, labels={"Player 2": "Player 2"}, font_size=15, font_color="green")

        # Draw edge labels with appropriate colors and alignment
        def draw_custom_edge_labels(pos, edge_labels):
            for (start, end), label in edge_labels.items():
                color = "blue" if start == "Player 1" else "green"  # Player 1 actions in blue, Player 2 in green
                ha = "right" if label in ["Out", "Left"] else "left"  # Align left or right based on action
                offset = -0.05 if ha == "right" else 0.05
                x_mid = (pos[start][0] + pos[end][0]) / 2 + offset
                y_mid = (pos[start][1] + pos[end][1]) / 2
                plt.text(x_mid, y_mid, label, fontsize=15, color=color, ha=ha)

        # Get edge labels and draw them
        edge_labels = nx.get_edge_attributes(G, 'label')
        draw_custom_edge_labels(pos, edge_labels)

        # Manually add colored text for payoffs at terminal states
        def draw_colored_payoff(node_pos, label, color1, color2, fontsize=20):
            part1, part2 = [x.strip() for x in label.split(",")]  # Split into two parts
            x, y = node_pos
            plt.annotate(part1, (x, y + 0.05), fontsize=fontsize, color=color1, ha='center')
            plt.annotate(part2, (x, y - 0.1), fontsize=fontsize, color=color2, ha='center')

        # Get the labels from the existing graph nodes
        node_labels = nx.get_node_attributes(G, 'label')

        # Add the colored payoffs for each terminal state
        draw_colored_payoff(pos["T1"], node_labels["T1"], "blue", "green")
        draw_colored_payoff(pos["T2"], node_labels["T2"], "blue", "green")
        draw_colored_payoff(pos["T3"], node_labels["T3"], "blue", "green")

        # Adjust plot limits to add padding
        plt.xlim(-1.5, 2)
        plt.ylim(-0.5, 2.5)

        # Display the plot
        plt.axis("off")
        plt.show()

# Example Usage
## 📌 To visualize **all games**, just call:
visualize_games(game_configs)

## 📌 To visualize **a single game**, specify the name:
# visualize_games(game_configs, "common_interest")
