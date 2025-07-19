Project: Kicker Fantasy Projections
Predict kicker performance based on player statistics and defensive matchup

Workflow:
1) load_data.py: Obtain raw pbp data
2) get_K_stats.py: Calculate weekly K player stats and K matchup stats
3) run_backtest.py : Calculate projected K points based on player + matchup stats
4) backtest_postproc.py: Evaluate results of backtest

Scripts:
load_data.py - Download play-by-play and weekly player data from nfl_data_py and save to csv

get_K_stats.py - Load play-by-play data, extract player statistics for FGs and PATs, calculate fantasy points scored, and save to csv

project_K_pts.py - Load K player and matchup stats, calculate expected number of attempts & makes by distance, then calculate expected fantasy points and save to csv

k_ppf.py - Preprocessing functions for kicking player statistics

Kicker Scoping.ipynb - Workspace for developing ideas before adding them as scripts to the workflow

Data:
pbp_2024.csv - Play-by-play data for 2024 season
weekly_2024.csv - Player stat lines for each week in the 2024 season
ftn_2024.csv - For The Numbers manual charting data. Not entirely sure what this contains, could be interesting for play predictor models or monte carlo simulation.
pbp_short_2024.csv - First 50 rows of pbp_2024.csv for easier inspection due to large file size
kicker_stats_2024.csv - Kicker player statistics for 2024 season, with FGs binned by distance and calculated fantasy points
KDEF_stats_2024.csv - Kicker matchup statistics for 2024 season, with FGs binned by distance

Issues:
- I'm getting a bunch of bad predictions from kickers who aren't actually playing; maybe handle this by dropping kickers who aren't the leading scorer on their own team that week
- Make estimate_KDEF_matchup() modular so that different models can be inserted and compared. Also change name to something like predict_KDEF_matchup()
- Get statistics for past 5 (or more) years
- Maybe drop 'games_played' column
- Include a volatility or boom/bust score using standard deviation of K & DEF attempts per game
- Add home/away splits to stats, and maybe indoor/outdoor splits
- Some sort of script to run through the workflow scripts in order?
