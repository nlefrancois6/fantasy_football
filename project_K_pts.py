import pandas as pd
import k_ppf
import nfl_data_py as nfl

print('Loading player & defense data')
# Load the player & defense data
df_k = pd.read_csv('Data/kicker_stats_2024.csv')
df_def = pd.read_csv('Data/KDEF_stats_2024.csv')

print('Predicting player performance (All matchups)')
# Step 1: Load the current week's schedule
yr = 2024 #Year
wk = 10 #Week
week_schedule = k_ppf.get_weekly_schedule(year=yr, week=wk)
print('Schedule for,',yr,'Week',wk)
print(week_schedule.head()) #Note: stadium info available here

# Step 2: Generate the matchups (returns list of tuples)
matchups = k_ppf.generate_weekly_matchups(df_k, week_schedule)

# Step 3: Run projections
weekly_kicker_projections = k_ppf.compare_kicker_matchups(matchups, df_k, df_def)

# Step 4: Display or export
print(weekly_kicker_projections.iloc[0:10])
save_files=True
if save_files:
    print('Saving kicker & matchup statistics to csv')
    #TODO: set filename dynamically with week value
    weekly_kicker_projections.to_csv("week10_kicker_projections.csv", index=False)
    print('Files saved')
