import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.stats import pearsonr

# File paths for model outputs: The same code should work for both New_model and Cogsci_models, make sure to check for file names
reu_file = "reu_direct_results.csv"
rc_file = "rc_direct_results.csv"

# Function to read CSV and store values
def read_csv(file_path):
    scores = {}
    try:
        with open(file_path, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                game = row["Game"]
                scores[game] = float(row["Relative Utility (P2 - P1)"])
        print(f"Successfully read data from: {file_path}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    return scores

# Load REU and RC results from CSV files
REU_scores = read_csv(reu_file)
RC_scores = read_csv(rc_file)

# Ensure both dictionaries contain the same games
common_games = sorted(set(REU_scores.keys()).intersection(set(RC_scores.keys())))

# Normalize REU and RC scores to a 0-100 scale for power prediction
def normalize_to_100_scale(scores):
    values = list(scores.values())
    scaled_values = minmax_scale(values, feature_range=(0, 100))
    return {game: scaled_values[i] for i, game in enumerate(scores.keys())}

# Get model-based predictions for power judgments
Predicted_Power_REU = normalize_to_100_scale(REU_scores)
Predicted_Power_RC = normalize_to_100_scale(RC_scores)

# Extract values for correlation and plotting
reu_power_values = [Predicted_Power_REU[game] for game in common_games]
rc_power_values = [Predicted_Power_RC[game] for game in common_games]

# Compute Pearson correlation
correlation, p_value = pearsonr(reu_power_values, rc_power_values)

# Print correlation result
print(f"\nPearson Correlation between Predicted Power (REU) and Predicted Power (RC): {correlation:.2f}")
print(f"P-value: {p_value:.4f}")

# Create scatter plot: REU Predicted Power vs. RC Predicted Power
plt.figure(figsize=(10, 6))
plt.scatter(reu_power_values, rc_power_values, color="blue", alpha=0.7)

# Label each point with the game name
for i, game in enumerate(common_games):
    plt.annotate(game, (reu_power_values[i], rc_power_values[i]), fontsize=10, alpha=0.7)

# Compute line of best fit
slope, intercept = np.polyfit(reu_power_values, rc_power_values, 1)
x_range = np.linspace(min(reu_power_values), max(reu_power_values), 100)
y_range = slope * x_range + intercept
plt.plot(x_range, y_range, linestyle="--", color="red", label="Best Fit Line")

# Customize plot
plt.xlabel("Predicted Power (REU Model) [0-100]", fontsize=14)
plt.ylabel("Predicted Power (RC Model) [0-100]", fontsize=14)
plt.title(f"Scatter Plot: REU vs. RC Power\nCorrelation: {correlation:.2f}", fontsize=16)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


