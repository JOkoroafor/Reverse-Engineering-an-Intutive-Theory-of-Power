import csv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd


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

# Load the CSV file
file_path = "power_stats_by_game_exp3_choice.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Ensure the necessary columns exist
if all(col in data.columns for col in ["Game", "Mean_Power", "SD_Power"]):
    # Create Average_Power dictionary
    Average_Power = data.set_index("Game")["Mean_Power"].to_dict()

    # Create std_devs_power dictionary
    std_devs_power = data.set_index("Game")["SD_Power"].to_dict()

    # Display the dictionaries
    print("Average_Power =", Average_Power)
    print("\nstd_devs_power =", std_devs_power)

else:
    print("Error: The file does not contain the required columns: 'Game', 'Mean_Power', and 'SD_Power'.")

# Confidence intervals (95% CI): CI = Z * (σ / sqrt(N))
N = 39  # Sample size
Z = 1.96  # Z-score for 95% confidence intervals
ci_power = {game: Z * (std / np.sqrt(N)) for game, std in std_devs_power.items()}

# Colors for each game
# Get unique game names
game_names = data['Game'].unique()

# Assign colors
color_palette = plt.cm.get_cmap('tab10', len(game_names))
game_colors = {game: color_palette(i) for i, game in enumerate(game_names)}

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