import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from scipy.stats import pearsonr

# Load participant data
participant_data = pd.read_csv("final_cleaned_data_2.csv", delimiter=",")

# Load model predictions
reu_predictions = pd.read_csv("reu_direct_results.csv")
rcr_predictions = pd.read_csv("rcr_direct_results.csv")

# Ensure games match between participant data and model predictions
games = reu_predictions['Game'].tolist()
participant_data = participant_data[participant_data['Game'].isin(games)]

# Group participants by ID
participants = participant_data['ID'].unique()

# Store groups
reu_group = []
rcr_group = []

# Compare REU and RCR fits for each participant
for participant in participants:
    participant_subset = participant_data[participant_data['ID'] == participant]
    
    # Match participant data with model predictions
    human_scores = participant_subset['Power'].values
    reu_scores = reu_predictions[reu_predictions['Game'].isin(participant_subset['Game'])]['Relative Utility (P2 - P1)'].values
    rcr_scores = rcr_predictions[rcr_predictions['Game'].isin(participant_subset['Game'])]['Relative Utility (P2 - P1)'].values
    
    # Calculate Pearson correlation for REU and RCR models
    if len(human_scores) > 1:  # Ensure there are enough data points for correlation
        reu_corr, _ = pearsonr(human_scores, reu_scores)
        rcr_corr, _ = pearsonr(human_scores, rcr_scores)
    
        # Assign participant to the better-fitting model group
        if reu_corr > rcr_corr:
            reu_group.append(participant)
        else:
            rcr_group.append(participant)

# Normalize REU and RCR scores to a 0-100 scale
def normalize_to_100_scale(df):
    scaled_values = minmax_scale(df['Relative Utility (P2 - P1)'].values, feature_range=(0, 100))
    return pd.Series(scaled_values, index=df['Game'].values)

# Get normalized model-based predictions for power judgments
Predicted_Power_REU = normalize_to_100_scale(reu_predictions)
Predicted_Power_RCR = normalize_to_100_scale(rcr_predictions)

# Function to create bar plots for each group
def plot_model_vs_human(group, model_predictions, model_name):
    group_data = participant_data[participant_data['ID'].isin(group)]
    
    avg_human_scores = group_data.groupby('Game')['Power'].mean()
    model_scores = model_predictions.loc[avg_human_scores.index]
    
    # Bar plot
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(avg_human_scores))

    plt.bar(index, avg_human_scores.values, bar_width, label='Human Data')
    plt.bar(index + bar_width, model_scores.values, bar_width, label=f'{model_name} Predictions')

    plt.xlabel('Games')
    plt.ylabel('Scores')
    plt.title(f'Human Data vs. {model_name} Predictions ({model_name} Cluster)')
    plt.xticks(index + bar_width / 2, avg_human_scores.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot for REU group
plot_model_vs_human(reu_group, Predicted_Power_REU, "REU")

# Plot for RCR group
plot_model_vs_human(rcr_group, Predicted_Power_RCR, "RCR")

# Print the number of participants in each group
print(f"REU Group: {len(reu_group)} participants")
print(f"RCR Group: {len(rcr_group)} participants")


