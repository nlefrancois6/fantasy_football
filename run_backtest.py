import pandas as pd
import sqlite3
import k_ppf

db_path = "Data/nfl_data.db"
conn = sqlite3.connect(db_path)

# Define output storage
backtest_results = []

# Connect to database
print('Loading kicker & matchup data')
with sqlite3.connect(db_path) as conn:
    # Load all weekly data
    df_k = pd.read_sql("SELECT * FROM weekly_kicker_stats_2024", conn)
    df_def = pd.read_sql("SELECT * FROM weekly_KDEF_stats_2024", conn)

# Determine all weeks in the season
all_weeks = sorted(df_k['week'].unique())

# Begin backtesting
yr = 2024 #TODO: pull the year directly from the data
for week in all_weeks[1:]:  # skip week 1, no prior data to predict with
    print(f"Backtesting Week {week}")
    # Historical data up to the previous week
    k_hist = df_k[df_k['week'] < week].copy()
    def_hist = df_def[df_def['week'] < week].copy()

    # Aggregate history to get cumulative per-game stats per kicker and defteam
    k_agg = k_hist.groupby("kicker_player_name", observed=True).mean(numeric_only=True).reset_index()
    def_agg = def_hist.groupby("defteam", observed=True).mean(numeric_only=True).reset_index()
    #Note: This might distort the accuracy values (mean of sums, losing weighting)
    
    # Get the actual matchups for this week
    week_schedule = k_ppf.get_weekly_schedule(year=yr, week=week) #Note: contains stadium, weather, etc

    # Generate the matchups (returns dataframe w/'kicker_player_name','defteam'
    matchups = k_ppf.generate_weekly_matchups(df_k, week_schedule)

    # Predict for each matchup
    for _, row in matchups.iterrows():
        kicker = row['kicker_player_name']
        defense = row['defteam']

        # Check both exist in historical data
        k_row = k_agg[k_agg['kicker_player_name'] == kicker]
        d_row = def_agg[def_agg['defteam'] == defense]
        if k_row.empty or d_row.empty:
            continue  # Skip if we don't have data

        # Predict stats using your model
        pred = k_ppf.estimate_KDEF_matchup(kicker, defense, k_row.iloc[0], d_row.iloc[0])

        # Get actual stats
        actual = df_k[(df_k['kicker_player_name'] == kicker) & (df_k['week'] == week)]
        if actual.empty:
            continue

        actual_pts = actual.iloc[0]['total_fantasy_pts']
        pred_pts = pred['total_expected_fantasy_points']  # make sure this matches your prediction dict

        backtest_results.append({
            "week": week,
            "kicker": kicker,
            "defteam": defense,
            "predicted_fp": pred_pts,
            "actual_fp": actual_pts,
            "abs_error": abs(pred_pts - actual_pts),
            "rel_error": pred_pts - actual_pts,
            "rel_error_std": (pred_pts - actual_pts)/actual_pts
        })

# Convert results to DataFrame
results_df = pd.DataFrame(backtest_results)

print('Backtest complete.')
print(results_df.tail())

save_files = True
if save_files:
    print('Saving backtest results to db')
    with sqlite3.connect(db_path) as conn:
        results_df.to_sql("k_backtest_results", conn, if_exists="replace", index=False)
    print("Results saved.")
