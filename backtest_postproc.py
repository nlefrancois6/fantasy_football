import pandas as pd
import sqlite3
import k_ppf
import matplotlib.pyplot as plt

db_path = "Data/nfl_data.db"
conn = sqlite3.connect(db_path)

print('Loading backtest data')
with sqlite3.connect(db_path) as conn:
    # Load all weekly data
    df = pd.read_sql("SELECT * FROM k_backtest_results", conn)

#Calculate accuracy metrics
mae = k_ppf.mean_absolute_error(df['actual_fp'], df['predicted_fp'])
print(f"Mean Absolute Error (MAE): {mae:.2f}")

corrs = k_ppf.weekly_spearman_corr(df)
# Filter out None values before computing average correlation
valid_corrs = [c for c in corrs if c is not None]
if valid_corrs:
    avg_corr = sum(valid_corrs) / len(valid_corrs)
    print(f"Average Spearman Correlation: {avg_corr:.3f}")
else:
    print("No valid weekly correlations to average.")

top5_acc = k_ppf.top_k_accuracy(df, k=5)
print(f"Top-5 Accuracy: {top5_acc:.2%}")

#Plot model performance
k_ppf.plot_weekly_spearman_corr(corrs)
k_ppf.plot_rel_error_hist(df)
