import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

"I want to build a script that retrieves human data values [Game, p_in, p_right] and stores them into a repo."
"The script should also retrieve and store corresponding model predictions [Game, p_in, p_right]"
# Human data

human_data = {
    "common_interest": {"p_in": 0.76047619, "p_right": 0.051904762},
    "safe_shot": {"p_in": 0.91833333, "p_right": 0.202142857142857},
    "strategic_dummy": {"p_in": 0.88190476, "p_right": 0.514523809523809},
    "near_dictator": {"p_in": 0.19071429, "p_right": 0.807619048},
    "costly_punish": {"p_in": 0.65952381, "p_right": 0.256904762},
    "free_punish": {"p_in": 0.5402381, "p_right": 0.451428571},
    "rational_punish": {"p_in": 0.4997619, "p_right": 0.22547619047619},
    "costly_help": {"p_in": 0.38047619, "p_right": 0.858571429},
    "free_help": {"p_in": 0.38238095, "p_right": 0.071428571},
    "trust_game": {"p_in": 0.355, "p_right": 0.741904761904762},
}


# Standard deviations for 95% CI calculation (provided data)
std_devs = {
    "common_interest": {"p_in": 0.264, "p_right": 0.139},
    "costly_punish": {"p_in": 0.268, "p_right": 0.301},
    "costly_help": {"p_in": 0.273, "p_right": 0.154},
    "free_help": {"p_in": 0.327, "p_right": 0.155},
    "free_punish": {"p_in": 0.263, "p_right": 0.342},
    "near_dictator": {"p_in": 0.264, "p_right": 0.232},
    "rational_punish": {"p_in": 0.312, "p_right": 0.255},
    "safe_shot": {"p_in": 0.228, "p_right": 0.272},
    "strategic_dummy": {"p_in": 0.25, "p_right": 0.0583},
    "trust_game": {"p_in": 0.278, "p_right": 0.315},
}

# Model predictions
model_predictions = {
    "common_interest": {"p_in": 0.8289777838057831, "p_right": 0.029787724282501517},
    "safe_shot": {"p_in": 0.8730979312081302, "p_right": 0.149095957071859},
    "strategic_dummy": {"p_in": 0.8426913251673352, "p_right": 0.5},
    "near_dictator": {"p_in": 0.1892129044638226, "p_right": 0.6674520968500137},
    "costly_punish": {"p_in": 0.7103300000906755, "p_right": 0.3325479031499864},
    "free_punish": {"p_in": 0.6232855369540307, "p_right": 0.5},
    "rational_punish": {"p_in": 0.5274840576987666, "p_right": 0.3325479031499864},
    "costly_help": {"p_in": 0.42042754707341423, "p_right": 0.850904042928141},
    "free_help": {"p_in": 0.43776411315405667, "p_right": 0.149095957071859},
    "trust_game": {"p_in": 0.42042754707341423, "p_right": 0.850904042928141},
}

# Number of data points
n = 42

# Compute 95% confidence intervals
confidence_intervals = {
    game: {
        "p_in": 1.96 * (std_devs[game]["p_in"] / np.sqrt(n)),
        "p_right": 1.96 * (std_devs[game]["p_right"] / np.sqrt(n)),
    }
    for game in human_data
}

# Collect data for correlation and R^2 calculations
human_p_in = []
model_p_in = []
human_p_right = []
model_p_right = []

for game in human_data.keys():
    human_p_in.append(human_data[game]["p_in"])
    model_p_in.append(model_predictions[game]["p_in"])
    human_p_right.append(human_data[game]["p_right"])
    model_p_right.append(model_predictions[game]["p_right"])

# Compute Pearson correlations
correlation_p_in, _ = pearsonr(model_p_in, human_p_in)
correlation_p_right, _ = pearsonr(model_p_right, human_p_right)

# Compute R^2 and adjusted R^2
r2_p_in = r2_score(human_p_in, model_p_in)
r2_p_right = r2_score(human_p_right, model_p_right)
adj_r2_p_in = 1 - (1 - r2_p_in) * (len(human_p_in) - 1) / (len(human_p_in) - 2)
adj_r2_p_right = 1 - (1 - r2_p_right) * (len(human_p_right) - 1) / (len(human_p_right) - 2)

# Mean Squared Error
MSE_p_in = mean_squared_error(human_p_in, model_p_in)
MSE_p_right = mean_squared_error(human_p_right, model_p_right)

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
for game, data in human_data.items():
    ci_p_in = confidence_intervals[game]["p_in"]
    plt.errorbar(
        model_predictions[game]["p_in"],  # Model predictions on x-axis
        data["p_in"],  # Human data on y-axis
        yerr=ci_p_in,  # Vertical error bars
        fmt="o",
        label=game,
        color=game_colors[game],
        markersize=8,
    )
plt.plot([0, 1], [0, 1], linestyle="--", color="black")
plt.xlabel("Model Predictions", fontsize = 20)
plt.ylabel("Human Judgment", fontsize = 20)
plt.legend(title="Game", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(False)
plt.tight_layout()
plt.show()

# Visualization for P(Right) (Player 2)
plt.figure(figsize=(10, 6))
for game, data in human_data.items():
    ci_p_right = confidence_intervals[game]["p_right"]
    plt.errorbar(
        model_predictions[game]["p_right"],  # Model predictions on x-axis
        data["p_right"],  # Human data on y-axis
        yerr=ci_p_right,  # Vertical error bars
        fmt="o",
        label=game,
        color=game_colors[game],
        markersize=8,
    )
plt.plot([0, 1], [0, 1], linestyle="--", color="black")
plt.grid(False)
plt.xlabel("Model Predictions", fontsize = 20)
plt.ylabel("Human Judgment", fontsize = 20)
plt.legend(title="Game", bbox_to_anchor=(1.05, 1), loc="upper left")  # Ensure legend appears
plt.grid(False)
plt.tight_layout()
plt.show()

# Print correlation and adjusted R^2 results
print(f"Correlation between Model and Human P(In) for Player 1: {correlation_p_in}")
print(f"Adjusted R^2 for P(In) for Player 1: {adj_r2_p_in}")
print(f"Correlation between Model and Human P(Right) for Player 2: {correlation_p_right}")
print(f"Adjusted R^2 for P(Right) for Player 2: {adj_r2_p_right}")
print(f"MSE for P(In) for Player 1: {MSE_p_in}")
print(f"MSE for P(Right) for Player 2: {MSE_p_right}")


 