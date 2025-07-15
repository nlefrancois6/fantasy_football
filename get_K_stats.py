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

print('Extracting kicker statistics from play-by-play data')
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
fg_df = kick_df[kick_df['play_type']=='field_goal']
pat_df = kick_df[kick_df['play_type']=='extra_point']

#Calculate fantasy points scored on each play
fg_df['fantasy_pts'] = fg_df.apply(k_ppf.fg_points, axis=1)
pat_df['fantasy_pts'] = pat_df['extra_point_result'].apply(k_ppf.pat_points)

#Get stats by kicker
fg_summary_k = k_ppf.get_fg_summary_k(fg_df) #Get fg stats by kicker
pat_summary_k = k_ppf.get_pat_summary_k(pat_df) #Get pat stats by kicker
fantasy_summary_k = k_ppf.get_fantasy_summary_k(fg_df,pat_df,kick_df) #Get fantasy points by kicker

kicker_summary = k_ppf.get_kicker_summary(fg_summary_k,pat_summary_k,fantasy_summary_k) #Get combined stats by kicker
kicker_summary = k_ppf.get_pergame_k(kicker_summary) #Normalize stats per game

print(kicker_summary[['kicker_player_name','posteam','games_played','total_attempts','attempts','total_fantasy_pts']].head())

#Get stats by defense
fg_summary_d = k_ppf.get_fg_summary_def(fg_df) #Get fg stats by kicker
pat_summary_d = k_ppf.get_pat_summary_def(pat_df) #Get pat stats by kicker
fantasy_summary_d = k_ppf.get_fantasy_summary_def(fg_df,pat_df,kick_df) #Get fantasy points by kicker

def_summary = k_ppf.get_def_summary(fg_summary_d,pat_summary_d,fantasy_summary_d) #Get combined stats by kicker
def_summary = k_ppf.get_pergame_k(def_summary) #Normalize stats per game
print(def_summary[['defteam','games_played','total_attempts','attempts','total_fantasy_pts']].head())

save_files = True
if save_files:
    print('Saving kicker & matchup statistics to db')
    with sqlite3.connect(db_path) as conn:
        kicker_summary.to_sql("kicker_stats_2024", conn, if_exists="replace", index=False)
        def_summary.to_sql("KDEF_stats_2024", conn, if_exists="replace", index=False)
    print('Files saved')

print('Script complete')

