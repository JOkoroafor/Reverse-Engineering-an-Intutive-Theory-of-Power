import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Define the key points for the preference graph
points = {
    "Altruistic (0,1)": (0, 1),
    "Prosocial (1/2, 1/2)": (0.5, 0.5),
    "Selfish (1,0)": (1, 0),
    "Competitive (1/2, -1/2)": (0.5, -0.5),
    "Sadistic (0,-1)": (0, -1),
    "Mutually Destructive (-1/2, -1/2)": (-0.5, -0.5),
    "Masochistic (-1, 0)": (-1, 0),
    "Martyr (-1/2, 1/2)": (-0.5, 0.5),
    "Apathetic (0, 0)": (0, 0)
}

# Generalized Beta PDF
def generalized_beta(x, w_O, w_S, loc=0, scale=1):
    """Generalized Beta PDF with location and scale parameters."""
    if scale <= 0:
        raise ValueError("Scale must be positive.")
    adjusted_x = (x - loc) / scale
    if w_O > 0:
        # Standard Beta behavior
        return beta.pdf(adjusted_x, w_O, w_S) / scale
    elif w_O < 0:
        # Inverted Beta behavior for negative w_O
        return -beta.pdf(adjusted_x, abs(w_O), w_S) / scale
    else:
        raise ValueError("w_O cannot be zero.")

# Define x-axis for Beta distributions
x = np.linspace(0, 1, 1000)  # Constrained to [0, 1] for Beta

# Define parameters for the generalized Beta distributions
beta_params = [
    (1.25, 1.5, 0, 1),  # Slightly skewed
    (5, 2, 0, 1),        # Right-skewed
    (-3, 1.75, 0, 1),    # Flipped
    (-1.5, 1.75, 0, 1),  # Flipped, less steep
    (2, 2, 0, 1)         # Symmetric Beta
]

# Set up the plot
fig, ax = plt.subplots(figsize=(2, 2))  # Keep the plot square to maintain proportions

# Plot the preference points
for label, (x_point, y_point) in points.items():
    ax.scatter(x_point, y_point, color='black', zorder=5)  # Add a black dot
    ax.text(x_point, y_point + 0.05, label, fontsize=8, ha='center', weight='bold')  # Bold text

# Add lines through x-axis, y-axis, and diagonals
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')  # Horizontal line (x-axis)
ax.axvline(0, color='black', linewidth=0.8, linestyle='-')  # Vertical line (y-axis)
ax.plot([0, 1], [1, 0], color='green', linestyle='--', label='$γ_S$, $γ_O$ = 1')  # Diagonal line
ax.plot([1, 0], [0, -1], color='purple', linestyle='--', label='$γ_S=1$, $γ_O$ = -1')  # Diagonal line
ax.plot([0, -1], [-1, 0], color='red', linestyle='--', label='$γ_S$, $γ_O$ = -1')  # Diagonal line
ax.plot([-1, 0], [0, 1], color='blue', linestyle='--', label='$γ_S$=-1, $γ_O$ =1')  # Diagonal line

# Overlay the generalized Beta distributions (scale y-values by 1/3)
for i, (w_O, w_S, loc, scale) in enumerate(beta_params):
    y = generalized_beta(x, w_O, w_S, loc, scale) * (1/3)  # Scale y-values
    ax.plot(x, y, label=f'Gen Beta({w_O}, {w_S})', linestyle='-', linewidth=1.5)

# Add axis labels
ax.set_xlabel('$w_S$', fontsize=12)
ax.set_ylabel('$w_O$', fontsize=12)

# Constrain axes to (-1, 1) and keep the aspect ratio equal
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal', adjustable='box')  # Ensure equal scaling of x and y axes

# Add legend
ax.legend(loc='upper left', fontsize=10)

# Refine plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(False)
plt.tight_layout()

# Display the plot
plt.show()

