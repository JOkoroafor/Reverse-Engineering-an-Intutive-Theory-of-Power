import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import r2_score

# Model predictions
GREU = {
    "common_interest": 0,
    "safe_shot": 6.13,
    "strategic_dummy": 0,
    "near_dictator": -11.1,
    "costly_punish": -1.84,
    "free_punish": 1.0,
    "rational_punish": 3.69,
    "costly_help": 4.38,
    "free_help": -5.62,
    "trust_game": 2.06,
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

# Standard deviations (provided values)
std_devs_power = {
    "common_interest": 31.3,
    "costly_punish": 26.7,
    "costly_help": 30.4,
    "free_help": 28.5,
    "free_punish": 30.5,
    "near_dictator": 18.8,
    "rational_punish": 31.0,
    "safe_shot": 36.2,
    "strategic_dummy": 30.0,
    "trust_game": 30.4,
}

# Confidence intervals (95% CI): CI = Z * (σ / sqrt(N))
N = 42  # Sample size
Z = 1.96  # Z-score for 95% confidence intervals
ci_power = {game: Z * (std / np.sqrt(N)) for game, std in std_devs_power.items()}

# Collect data for correlation and R^2 calculations
model = []
human = []

for game in GREU.keys():
    model.append(GREU[game])
    human.append(Average_Power[game])

# Compute Pearson correlations
correlation, _ = pearsonr(human, model)

# Compute adjusted R^2
n = len(human)
p = 1  # Single predictor
adj_r2 = 1 - (1 - (correlation * correlation)) * (n - 1) / (n - p - 1)

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

# Visualization with error bars for confidence intervals
plt.figure(figsize=(10, 6))
for game, data in GREU.items():
    plt.errorbar(
        data,  # Model predictions on x-axis
        Average_Power[game],  # Human data on y-axis
        yerr=ci_power[game],  # Vertical error bars (95% CI)
        fmt="o",
        label=game,
        color=game_colors[game],
        markersize=8,
        capsize=5,  # Add caps to the error bars
    )

# Add line of best fit
# Add line of best fit as a dotted line
    # Create a smooth trend line
x_range = np.linspace(min(model) - 1, max(model) + 1, 100)
y_range = slope * x_range + intercept

# Plot the line of best fit as a fully dashed line
plt.plot(x_range, y_range, linestyle="--", color="black", label="Best Fit Line")



# Customize axes and legend
plt.xlabel("Average Relative Resource Control (Model Predictions)", fontsize=20)
plt.ylabel("Average Power (Human Data)", fontsize=20)
plt.legend(title="Game", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(False)
plt.tight_layout()
plt.show()

# Print correlation and adjusted R^2 results
print(f"Correlation between GREU and Power: {correlation:.2f}")
print(f"Adjusted R²: {adj_r2:.2f}")
