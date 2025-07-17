import pandas as pd
import sqlite3
import k_ppf

print('Loading play-by-play data')
# Load the weekly data
db_path = 'Data/nfl_data.db'

# Connect to the database, read the data, and close the connection
with sqlite3.connect(db_path) as conn:
    # SQL query to load data from the pbp_2024 table
    query = "SELECT * FROM pbp_2024"

    # Read the data into a pandas DataFrame
    pbp = pd.read_sql_query(query, conn)

print('Extracting kicker & matchup statistics from play-by-play data')
#List of columns to keep for kicking data
kicking_cols = [
    'play_id', 'game_id', 'week', 'posteam', 'defteam', 'game_date', 'qtr', 'down',
    'play_type', 'field_goal_result', 'kick_distance', 'extra_point_result',
    'extra_point_attempt', 'field_goal_attempt', 'two_point_attempt',
    'kicker_player_id', 'kicker_player_name'
]

# Filter to only plays that are kick-related
kick_plays = pbp[pbp['play_type'].isin(['field_goal', 'extra_point'])]

# Select only relevant columns
kick_df = kick_plays[kicking_cols]
fg_df = kick_df[kick_df['play_type']=='field_goal'].copy()
pat_df = kick_df[kick_df['play_type']=='extra_point'].copy()

#Calculate fantasy points scored on each play
fg_df['fantasy_pts'] = fg_df.apply(k_ppf.fg_points, axis=1)
pat_df['fantasy_pts'] = pat_df['extra_point_result'].apply(k_ppf.pat_points)

# --- Process stats per week ---
weekly_kicker_stats = []
weekly_def_stats = []

for week in sorted(kick_df['week'].unique()):
    wk_fg_df = fg_df[fg_df['week'] == week].copy()
    wk_pat_df = pat_df[pat_df['week'] == week].copy()
    wk_kick_df = kick_df[kick_df['week'] == week].copy()

    # Kicker stats
    fg_summary = k_ppf.get_fg_summary_k(wk_fg_df) #Get fg stats by kicker
    pat_summary = k_ppf.get_pat_summary_k(wk_pat_df) #Get pat stats by kicker
    fantasy_summary = k_ppf.get_fantasy_summary_k(wk_fg_df, wk_pat_df, wk_kick_df) #Get fantasy points by kicker
    kicker_summary = k_ppf.get_kicker_summary(fg_summary, pat_summary, fantasy_summary) #Get combined stats by kicker

    kicker_summary['week'] = week
    weekly_kicker_stats.append(kicker_summary)

    #Defense stats
    fg_summary_d = k_ppf.get_fg_summary_def(wk_fg_df) #Get fg stats by def
    pat_summary_d = k_ppf.get_pat_summary_def(wk_pat_df) #Get pat stats by def
    fantasy_summary_d = k_ppf.get_fantasy_summary_def(wk_fg_df,wk_pat_df,wk_kick_df) #Get fantasy points by def
    def_summary = k_ppf.get_def_summary(fg_summary_d,pat_summary_d,fantasy_summary_d) #Get combined stats by def
    
    def_summary['week'] = week
    weekly_def_stats.append(def_summary)
    
# --- Combine all weeks ---
all_weekly_k_stats = pd.concat(weekly_kicker_stats, ignore_index=True)
all_weekly_kdef_stats = pd.concat(weekly_def_stats, ignore_index=True)

print(all_weekly_k_stats.iloc[0:10])
print(all_weekly_kdef_stats.iloc[0:10])

save_files = True
if save_files:
    print('Saving kicker & matchup statistics to db')
    with sqlite3.connect(db_path) as conn:
        all_weekly_k_stats.columns = all_weekly_k_stats.columns.str.strip().str.replace(' ', '_')
        all_weekly_kdef_stats.columns = all_weekly_kdef_stats.columns.str.strip().str.replace(' ', '_')
        
        all_weekly_k_stats.to_sql("weekly_kicker_stats_2024", conn, if_exists="replace", index=False)
        all_weekly_kdef_stats.to_sql("weekly_KDEF_stats_2024", conn, if_exists="replace", index=False)
    print('Files saved')

print('Script complete')

