import csv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import r2_score

# File paths for different models
file_paths = [
    "reu_direct_results.csv",
    "rcr_direct_results.csv",
    "rccr_direct_results.csv",
    "rc_direct_results.csv",
]

# Dictionary to store results for each model
model_results = {}

# Function to open and read CSV files
def open_csv(file_path):
    REU = {}
    try:
        with open(file_path, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                game = row["Game"]
                REU[game] = float(row["Relative Utility (P2 - P1)"])
        print(f"Successfully read data from: {file_path}")
        print(f"Contents of {file_path}: {REU}")  # Debugging log
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    return REU

# Load results from all model files
for file_path in file_paths:
    model_name = file_path.replace(".csv", "")  # Extract model name from filename
    REU = open_csv(file_path)
    if REU:
        model_results[model_name] = REU

# Human Data
Average_Power = {
    "common_interest_1": 54.4583333333333,
    "common_interest_2": 52.2708333333333,
    "costly_help_2": 47.375,
    "costly_punish_1": 48.7083333333333,
    "free_help_2": 37.9375,
    "near_dictator_1": 20.6458333333333,
    "safe_shot_1": 77.2708333333333,
    "safe_shot_2": 77.9791666666667,
    "strategic_dummy_1": 46.5416666666667,
    "trust_game_2": 52.0833333333333,
}

# Standard deviations
std_devs_power = {
    "common_interest_1": 27.9429192746329,
    "common_interest_2": 29.2609241890471,
    "costly_help_2": 29.1668895634847,
    "costly_punish_1": 29.661894986403,
    "free_help_2": 31.1362183918452,
    "near_dictator_1": 28.2259044921002,
    "safe_shot_1": 30.2259150352196,
    "safe_shot_2": 26.0126725117907,
    "strategic_dummy_1": 32.477133253986,
    "trust_game_2": 29.4349866607123,
}

# Confidence intervals (95% CI): CI = Z * (σ / sqrt(N))
N = 48  # Sample size
Z = 1.96  # Z-score for 95% confidence intervals
ci_power = {game: Z * (std / np.sqrt(N)) for game, std in std_devs_power.items()}

# Colors for each game
game_colors = {
    "common_interest_1": "blue",
    "common_interest_2": "orange",
    "costly_help_2": "green",
    "costly_punish_1": "red",
    "free_help_2": "purple",
    "near_dictator_1": "brown",
    "safe_shot_1": "pink",
    "safe_shot_2": "cyan",
    "strategic_dummy_1": "yellow",
    "trust_game_2": "magenta",
}

# Process each model
for model_name, REU in model_results.items():
    print(f"\nAnalyzing model: {model_name}")
    print(f"Loaded model values: {REU}")  # Debugging log

    # Collect data for correlation and R^2 calculations
    model = []
    human = []

    for game in REU.keys():
        model.append(REU[game])
        human.append(Average_Power[game])

    print(f"Model values for correlation: {model}")  # Debugging log
    print(f"Human data values for correlation: {human}")  # Debugging log

    # Compute Pearson correlations
    correlation, _ = pearsonr(human, model)

    # Compute R^2 and adjusted R^2
    n = len(human)
    p = 1  # Single predictor
    adj_r2 = 1 - (1 - (correlation * correlation)) * (n - 1) / (n - p - 1)

    # Compute line of best fit
    slope, intercept = np.polyfit(model, human, 1)
    x_range = np.linspace(min(model) - 1, max(model) + 1, 100)
    y_range = slope * x_range + intercept

    # Visualization with error bars for confidence intervals
    plt.figure(figsize=(10, 6))
    for game, data in REU.items():
        plt.errorbar(
            data,  # Model predictions on x-axis
            Average_Power[game],  # Human data on y-axis
            yerr=ci_power[game],  # Vertical error bars (95% CI)
            fmt="o",
            label=game,
            color=game_colors.get(game, "gray"),  # Default color if not found
            markersize=8,
            capsize=5,  # Add caps to the error bars
        )

    # Plot the line of best fit as a fully dashed line
    plt.plot(x_range, y_range, linestyle="--", color="black", label="Best Fit Line")

    # Customize axes and legend
    plt.xlabel("Model Predictions", fontsize=20)
    plt.ylabel("Average Power (Human Data)", fontsize=20)
    plt.title(f"{model_name} Model: Correlation {correlation:.2f}")
    plt.legend(title="Game", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Print correlation and adjusted R² results
    print(f"Correlation between {model_name} and Power: {correlation:.2f}")
    print(f"Adjusted R²: {adj_r2:.2f}")