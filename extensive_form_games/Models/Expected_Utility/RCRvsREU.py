import csv
import matplotlib.pyplot as plt
import numpy as np

# File paths for model outputs
reu_file = "reu_direct_results.csv"
rcr_file = "rcr_direct_results.csv"

# Dictionary to store model results
REU_scores = {}
RCR_scores = {}

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
        print(scores)  # Debugging log
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    return scores

# Load REU and RCR results from CSV files
REU_scores = read_csv(reu_file)
RCR_scores = read_csv(rcr_file)

# Ensure both dictionaries contain the same games
common_games = set(REU_scores.keys()).intersection(set(RCR_scores.keys()))

# Extract values for plotting
reu_values = [REU_scores[game] for game in common_games]
rcr_values = [RCR_scores[game] for game in common_games]
game_labels = list(common_games)

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(reu_values, rcr_values, color="blue", alpha=0.7)

# Label each point with game names
for i, game in enumerate(game_labels):
    plt.annotate(game, (reu_values[i], rcr_values[i]), fontsize=10, alpha=0.7)

# Customize plot
plt.xlabel("Relative Expected Utility (REU)", fontsize=14)
plt.ylabel("Relative Control over Resources (RCR)", fontsize=14)
plt.title("Scatter Plot of REU vs. RCR Across Games", fontsize=16)
plt.grid(False)
plt.show()
