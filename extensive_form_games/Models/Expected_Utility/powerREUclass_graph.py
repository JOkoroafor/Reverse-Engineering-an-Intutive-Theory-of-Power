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
    "mpru_direct_results.csv",
    "greu_results.csv",
    "greu_estimated_results.csv"
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
    plt.xlabel("Relative Expected Utility (Model Predictions)", fontsize=20)
    plt.ylabel("Average Power (Human Data)", fontsize=20)
    plt.title(f"{model_name} Model: Correlation {correlation:.2f}")
    plt.legend(title="Game", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Print correlation and adjusted R² results
    print(f"Correlation between {model_name} and Power: {correlation:.2f}")
    print(f"Adjusted R²: {adj_r2:.2f}")



