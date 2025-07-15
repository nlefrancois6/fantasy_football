import pandas as pd
import sqlite3
import k_ppf

db_path = "Data/nfl_data.db"
conn = sqlite3.connect(db_path)

# List all tables in the database
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables:\n", tables)

print('Loading player & defense data')
# Load the player & defense data
db_path = 'Data/nfl_data.db'

# Connect to the database, read the data, and close the connection
with sqlite3.connect(db_path) as conn:
    # SQL query to load data from the pbp_2024 table
    query = "SELECT * FROM kicker_stats_2024"
    # Read the data into a pandas DataFrame
    df_k = pd.read_sql_query(query, conn)

    # SQL query to load data from the pbp_2024 table
    query = "SELECT * FROM KDEF_stats_2024"
    # Read the data into a pandas DataFrame
    df_def = pd.read_sql_query(query, conn)

print('Predicting player performance (All matchups)')
# Step 1: Load the current week's schedule
yr = 2024 #Year
wk = 12 #Week
week_schedule = k_ppf.get_weekly_schedule(year=yr, week=wk)
print('Schedule for,',yr,'Week',wk)
print(week_schedule.head()) #Note: stadium info available here

# Step 2: Generate the matchups (returns list of tuples)
matchups = k_ppf.generate_weekly_matchups(df_k, week_schedule)

# Step 3: Run projections
weekly_kicker_projections = k_ppf.compare_kicker_matchups(matchups, df_k, df_def)

# Step 4: Display & export
print(weekly_kicker_projections.iloc[0:10])
save_files=True
if save_files:
    print('Saving kicker & matchup statistics to db')
    with sqlite3.connect(db_path) as conn:
        #TODO: Instead of separate tables, create & append a single table for all projections with a week column
        weekly_kicker_projections.to_sql(f"week{wk}_kicker_projections", conn, if_exists="replace", index=False)
    print('Files saved')
