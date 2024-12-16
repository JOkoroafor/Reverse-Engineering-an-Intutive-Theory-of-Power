import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

# Create a directed graph
G = nx.DiGraph()

# Add nodes for the game states
G.add_node("Player 1", label="Player 1")
G.add_node("Player 2", label="Player 2")
G.add_node("T1", label="$5 , $3")
G.add_node("T2", label="$10 , $5")
G.add_node("T3", label="$3, $10")

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

# Manually adjust the spacing between States 3, 4, and 5 to match the distance
distance_offset = abs(pos['Player 2'][0] - pos['T3'][0])  # Use the distance between State 1 and State 2
pos['Player 1'] = (pos['Player 2'][0] - distance_offset, pos['Player 1'][1])  # Adjust State 4 to the left
pos['T1'] = (pos['Player 1'][0] - distance_offset, pos['T1'][1])  # Adjust State 4 to the left

# Draw the nodes
nx.draw_networkx_nodes(G, pos, node_size=7000, node_color="white", edgecolors="black")

# Draw the edges
nx.draw_networkx_edges(G, pos,arrowsize=30)

# Manually add arrows using FancyArrowPatch
ax = plt.gca()  # Get the current axis
for (start, end) in G.edges():
    # Get the start and end positions
    start_pos = pos[start]
    end_pos = pos[end]
    # Create an arrow patch
    arrow = FancyArrowPatch(
        posA=start_pos, posB=end_pos,
        arrowstyle='-|>', color='black', mutation_scale=20, lw=1.5,
        shrinkA=15, shrinkB=15  # Shrink arrow away from nodes
    )
    ax.add_patch(arrow)

# Draw node labels with colors
nx.draw_networkx_labels(G, pos, labels={"Player 1": "Player 1"}, font_size=15, font_color="blue")
nx.draw_networkx_labels(G, pos, labels={"Player 2": "Player 2"}, font_size=15, font_color="green")

# Draw edge labels with appropriate colors and alignment
def draw_custom_edge_labels(pos, edge_labels):
    for (start, end), label in edge_labels.items():
        # Set color based on the player
        color = "blue" if start == "Player 1" else "green"
        # Determine alignment based on the edge direction
        if label in ["Out", "Left"]:
            ha = "right"  # Align to the left side of the edge
            offset = -0.05
        else:
            ha = "left"   # Align to the right side of the edge
            offset = 0.05
        # Get the midpoint of the edge
        x_mid = (pos[start][0] + pos[end][0]) / 2 + offset
        y_mid = (pos[start][1] + pos[end][1]) / 2
        # Draw the edge label with the specified color and alignment
        plt.text(x_mid, y_mid, label, fontsize=15, color=color, ha=ha)

# Get edge labels and draw them with the custom function
edge_labels = nx.get_edge_attributes(G, 'label')
draw_custom_edge_labels(pos, edge_labels)

# Manually add colored text for the payoffs based on existing labels
def draw_colored_payoff(node_pos, label, color1, color2, fontsize=20):
    # Split the label into two parts based on the comma separator
    part1, part2 = [x.strip() for x in label.split(",")]
    # Draw each part with the specified color
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
plt.xlim(-1.5, 2)  # Increase the range to ensure all text is visible
plt.ylim(-0.5, 2.5)  # Add padding above and below the nodes

# Display the plot
plt.axis("off")
plt.show()