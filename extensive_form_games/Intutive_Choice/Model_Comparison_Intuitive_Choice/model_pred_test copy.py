import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

# Constants [To do: Generalise to take N from Human Data n(rows) excluding title]
N = 42  # number of participants for CI

# 1) Load model predictions
df_model = pd.read_csv("test_2_model_predictions.csv")  # columns: Game, Model, p_in, p_right

# Split into seperate DataFrames
df_selfish = df_model[df_model["Model"] == "Selfish"]
df_ia = df_model[df_model["Model"] == "InequalityAversion"]
df_reciprocity = df_model[df_model["Model"] == "Reciprocity"]

# 2) Load human test data
df_human = pd.read_csv("exp3_summary_stats.csv")  # columns: Game, Mean_P1_Expectation, SD_P1_Expectation, Mean_P2_Expectation, SD_P2_Expectation

# Compute 95% CIs
df_human["CI_p_in"] = 1.96 * df_human["SD_P1_Expectation"] / np.sqrt(N)
df_human["CI_p_right"] = 1.96 * df_human["SD_P2_Expectation"] / np.sqrt(N)

def evaluate_and_plot(df_pred, model_name):
    # Merge predictions with human
    df = pd.merge(df_pred, df_human, on="Game", how="inner")
    
    # Compute stats
    pred_p_in    = df["p_in"].values
    human_p_in   = df["Mean_P1_Expectation"].values
    ci_in        = df["CI_p_in"].values
    
    pred_p_right  = df["p_right"].values
    human_p_right = df["Mean_P2_Expectation"].values
    ci_right      = df["CI_p_right"].values
    
    corr_in, _  = pearsonr(pred_p_in, human_p_in)
    corr_right, _ = pearsonr(pred_p_right, human_p_right)

    r2_in     = r2_score(human_p_in, pred_p_in)
    r2_right  = r2_score(human_p_right, pred_p_right)
    adj_r2_in  = 1 - (1 - r2_in)  * (len(human_p_in) - 1)  / (len(human_p_in) - 2)
    adj_r2_right = 1 - (1 - r2_right) * (len(human_p_right) - 1) / (len(human_p_right) - 2)

    mse_in      = mean_squared_error(human_p_in, pred_p_in)
    mse_right   = mean_squared_error(human_p_right, pred_p_right)
    
    # Color mapping
    games = df["Game"].unique()
    cmap  = plt.cm.get_cmap('tab10', len(games))
    color_map = {g: cmap(i) for i, g in enumerate(games)}
    
    # Marker size
    msize = 50
    
    # Plot P(In)
    plt.figure(figsize=(8,6))
    for game in games:
        sub = df[df["Game"] == game]
        plt.errorbar(
            sub["p_in"], sub["Mean_P1_Expectation"],
            yerr=sub["CI_p_in"],
            fmt='o',
            color=color_map[game],
            label=game,
            markersize=msize/len(games)
        )
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("Model P(In)")
    plt.ylabel("Human P(In)")
    plt.title(f"{model_name}\nP(In)  r={corr_in:.3f}, adj R²={adj_r2_in:.3f}, MSE={mse_in:.3f}")
    plt.legend(title="Game", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    # Plot P(Right)
    plt.figure(figsize=(8,6))
    for game in games:
        sub = df[df["Game"] == game]
        plt.errorbar(
            sub["p_right"], sub["Mean_P2_Expectation"],
            yerr=sub["CI_p_right"],
            fmt='o',
            color=color_map[game],
            label=game,
            markersize=msize/len(games)
        )
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("Model P(Right)")
    plt.ylabel("Human P(Right)")
    plt.title(f"{model_name}\nP(Right)  r={corr_right:.3f}, adj R²={adj_r2_right:.3f}, MSE={mse_right:.3f}")
    plt.legend(title="Game", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    # Print stats
    print(f"{model_name} Model Statistics:")
    print(f"  P(In):  r={corr_in:.3f}, adj R²={adj_r2_in:.3f}, MSE={mse_in:.3f}")
    print(f"  P(Right): r={corr_right:.3f}, adj R²={adj_r2_right:.3f}, MSE={mse_right:.3f}\n")

# 4) Run for both models
evaluate_and_plot(df_selfish,"Selfish Level-K")
evaluate_and_plot(df_ia,"Inequality Aversion Level-K")
evaluate_and_plot(df_reciprocity,"Reciprocity Level-K")



 