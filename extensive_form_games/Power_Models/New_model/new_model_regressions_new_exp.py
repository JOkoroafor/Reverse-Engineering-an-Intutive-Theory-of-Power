import csv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd

# File paths for different models
file_paths = [
    "reu_direct_results.csv",
    "rcr_direct_results.csv",
    "rccr_direct_results.csv",
    "rc_direct_results.csv",
    "rca_direct_results.csv",
]

# Dictionary to store results for each model
model_results = {}

# Function to open and read CSV files
def open_csv(file_path):
    model_data = {}
    try:
        with open(file_path, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                game = row["Game"]
                model_data[game] = float(row["Relative Utility (P2 - P1)"])
        print(f"Successfully read data from: {file_path}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    return model_data

# Load results from all model files
for file_path in file_paths:
    model_name = file_path.replace(".csv", "")  # Extract model name from filename
    data = open_csv(file_path)
    if data:
        model_results[model_name] = data

# Load the human data
file_path = "power_stats_by_game_new_exp.csv"
data = pd.read_csv(file_path)

# Ensure the necessary columns exist
if all(col in data.columns for col in ["Game", "Mean_Power", "SD_Power"]):
    Average_Power = data.set_index("Game")["Mean_Power"].to_dict()
    std_devs_power = data.set_index("Game")["SD_Power"].to_dict()
else:
    print("Error: The file does not contain the required columns: 'Game', 'Mean_Power', and 'SD_Power'.")

# Confidence intervals (95% CI)
N = 44
Z = 1.96
ci_power = {game: Z * (std / np.sqrt(N)) for game, std in std_devs_power.items()}

# Assign colors
game_names = data['Game'].unique()
color_palette = plt.cm.get_cmap('tab10', len(game_names))
game_colors = {game: color_palette(i) for i, game in enumerate(game_names)}

# --------- MULTIPLE LINEAR REGRESSION --------- #

# Define model combinations
combinations = {
    "REU": ["reu_direct_results"],
    "RC ": ["rc_direct_results"],
    "RCR ": ["rcr_direct_results"],
    "RCCR": ["rccr_direct_results"],
    "RCA": ["rca_direct_results"]
}

for combo_name, models in combinations.items():
    print(f"\nRunning Multiple Linear Regression for: {combo_name}")
    
    # Prepare data
    X = []
    Y = []
    for game in Average_Power.keys():
        try:
            predictors = [model_results[model][game] for model in models]
            X.append(predictors)
            Y.append(Average_Power[game])
        except KeyError:
            continue  # Skip games not found in all models

    X = np.array(X)
    Y = np.array(Y)

    # Fit the multiple linear regression model
    reg = LinearRegression()
    reg.fit(X, Y)
    Y_pred = reg.predict(X)

    # Confidence intervals (95% CI)
    N = 44
    Z = 1.96
    # Compute correlation and R-squared
    correlation, _ = pearsonr(Y, Y_pred)
    adj_r2 = 1 - (1 - r2_score(Y, Y_pred)) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)

    # Visualization with error bars
    plt.figure(figsize=(10, 6))
    for i, game in enumerate(Average_Power.keys()):
        if i < len(Y_pred):
            plt.errorbar(
                Y_pred[i],
                Y[i],
                yerr=ci_power.get(game, 0),
                fmt="o",
                label=game,
                color=game_colors.get(game, "gray"),
                markersize=8,
                capsize=5,
            )
    
        # … your existing error‐bar loop here …

    #  ——> add this right before you draw your best‐fit line
    # horizontal “equal power” line
    plt.axhline(50, linestyle="-", color="black")
    # get current x‐axis limits so we can position our labels
    xmin, xmax = plt.xlim()
    # a little bit in from the left, at y=52
    plt.text(xmin + 0.02*(xmax-xmin), 52, "P₂ > P₁", va="bottom", fontsize=12)
    # same x, at y=48
    plt.text(xmin + 0.02*(xmax-xmin), 48, "P₁ > P₂", va="top",    fontsize=12)

    # now draw your 45° line of “perfect fit”
    plt.plot([min(Y_pred), max(Y_pred)],
             [min(Y_pred), max(Y_pred)],
             linestyle="--", color="black", label="Perfect Fit")

    # … the rest of your labeling / legend / show() …


    # Customize plot
    plt.xlabel("Model Predictions", fontsize=20)
    plt.ylabel("Average Human Response", fontsize=20)
    plt.title(f"{combo_name}: r = {correlation:.3f}")
    plt.legend(title="Game", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Print results
    print(f"Correlation for {combo_name}: {correlation:.2f}")
    print(f"Adjusted R² for {combo_name}: {adj_r2:.2f}")
    print(f"Model Coefficients: {reg.coef_}")
    print(f"Intercept: {reg.intercept_}")
