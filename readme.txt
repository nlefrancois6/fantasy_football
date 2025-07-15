Project: Kicker Fantasy Projections
Predict kicker performance based on player statistics and defensive matchup

Workflow:
1) load_data.py: Obtain raw pbp data
2) get_K_stats.py: Calculate K player stats and K matchup stats
3) project_K_pts: Calculate projected K points based on player + matchup stats

Store all data in nfl_data.db using sqlite3

Scripts:
load_data.py - Download play-by-play and weekly player data from nfl_data_py and save to db

get_K_stats.py - Load play-by-play data, extract player statistics for FGs and PATs, calculate fantasy points scored, and save to db

project_K_pts.py - Load K player and matchup stats, calculate expected number of attempts & makes by distance, then calculate expected fantasy points.

k_ppf.py - Preprocessing functions for kicking player statistics

Kicker Scoping.ipynb - Workspace for developing ideas before adding them as scripts to the workflow

Data:
nfl_data.db - sqlite3 database containing all data tables

pbp_2024.csv - Play-by-play data for 2024 season
weekly_2024.csv - Player stat lines for each week in the 2024 season
ftn_2024.csv - For The Numbers manual charting data. Not entirely sure what this contains, could be interesting for play predictor models or monte carlo simulation.
pbp_short_2024.csv - First 50 rows of pbp_2024.csv for easier inspection due to large file size
kicker_stats_2024.csv - Kicker player statistics for 2024 season, with FGs binned by distance and calculated fantasy points
KDEF_stats_2024.csv - Kicker matchup statistics for 2024 season, with FGs binned by distance

Issues:
- Resolve SettingWithCopyWarnings in k_ppf.fg_summary, pat_summary, and fantasy_summary functions
- Some sort of script to run through the workflow scripts in order?
- Get statistics for past 5 (or more) years
- Set up a backtest to see how good my predictions are. PBP dataset can be cut off at week 12 and then predictions for week 13-17 can be compared to actual results
- Include a volatility or boom/bust score using standard deviation of K & DEF attempts per game
- Add home/away splits to stats, and maybe indoor/outdoor splits
