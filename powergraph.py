import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import r2_score

# Model predictions
Average_REU = {
    "common_interest": 0,
    "safe_shot": 3.063735939,
    "strategic_dummy": 0,
    "near_dictator": -5.548773282,
    "costly_punish": -0.92220624,
    "free_punish": 0.948644009,
    "rational_punish": 1.83882437,
    "costly_help": 2.190784259,
    "free_help": -2.811179435,
    "trust_game": 1.031639353,
}

# Human Data
Average_Power = {
    "common_interest": 49.33333333,
    "safe_shot": 69.52380952,
    "strategic_dummy": 34.28571429,
    "near_dictator": 13.69047619,
    "costly_punish": 35.69047619,
    "free_punish": 48.14285714,
    "rational_punish": 48.64285714,
    "costly_help": 50.71428571,
    "free_help": 25.88095238,
    "trust_game": 38.57142857,
}

# Collect data for correlation and R^2 calculations
model = []
human = []

for game in Average_REU.keys():
    model.append(Average_REU[game])
    human.append(Average_Power[game])

# Compute Pearson correlations
correlation, _ = pearsonr(human, model)

# Compute R^2 and adjusted R^2
r2 = r2_score(human, model)
n = len(human)
p = 1  # Single predictor
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Compute line of best fit
slope, intercept = np.polyfit(model, human, 1)
best_fit_line = np.polyval([slope, intercept], model)

# Colors for each game
game_colors = {
    "trust_game": "blue",
    "strategic_dummy": "orange",
    "safe_shot": "green",
    "rational_punish": "red",
    "near_dictator": "purple",
    "free_punish": "brown",
    "free_help": "pink",
    "costly_punish": "cyan",
    "costly_help": "yellow",
    "common_interest": "magenta",
}

# Visualization for P(In) (Player 1)
plt.figure(figsize=(10, 6))
for game, data in Average_REU.items():
    plt.errorbar(
        data,  # Model predictions on x-axis
        Average_Power[game],  # Human data on y-axis
        yerr=0,  # Vertical error bars
        fmt="o",
        label=game,
        color=game_colors[game],
        markersize=8,
    )

# Add line of best fit
plt.plot(model, best_fit_line, linestyle="-", color="black")


plt.xlabel("Average REU (Model Predictions)", fontsize = 20)
plt.ylabel("Average Power (Human Data)", fontsize =20)

plt.legend(title="Game", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(False)
plt.tight_layout()
plt.show()

# Print correlation and adjusted R^2 results
print(f"Correlation between REU and Power: {correlation:.2f}")
print(f"Adjusted R²: {adj_r2:.2f}")
