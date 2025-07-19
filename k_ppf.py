import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nfl_data_py as nfl
from scipy.stats import spearmanr
import os

# Assign points based on FG made
def fg_points(row):
    if row['field_goal_result'] == 'made':
        if row['kick_distance'] <= 39:
            return 3
        elif row['kick_distance'] <= 49:
            return 4
        elif row['kick_distance'] <= 59:
            return 5
        else:
            return 6
    elif row['field_goal_result'] == 'missed':
        return -1
    else:
        return 0  # for blocked or null results
    
# Assign points based on PAT result
def pat_points(result):
    if result == 'good':
        return 1
    elif result == 'failed':
        return -1
    else:
        return 0  # e.g., blocked or null

def get_pat_summary_k(pat_df):

    # ---- 1. Extra Point Attempts ----
    pat_attempts = (
        pat_df
        .groupby(['kicker_player_id', 'kicker_player_name'], observed=True)
        .size()
        .reset_index(name='attempts')
    )

    # ---- 2. Extra Points Made ----
    pat_made = (
        pat_df[pat_df['extra_point_result'] == 'good']
        .groupby(['kicker_player_id', 'kicker_player_name'], observed=True)
        .size()
        .reset_index(name='made')
    )

    # ---- 3. Merge and Calculate XP% ----
    pat_summary = pd.merge(pat_attempts, pat_made, on=['kicker_player_id', 'kicker_player_name'], how='left')
    pat_summary['made'] = pat_summary['made'].fillna(0).astype(int)
    pat_summary['xp_pct'] = pat_summary['made'] / pat_summary['attempts']

    # ---- 4. Sort ----
    pat_summary = pat_summary.sort_values('attempts', ascending=False)

    #Rename 'made' and 'attempts' to avoid confusion with field goals
    pat_summary = pat_summary.rename(columns={"made": "xp_made", "attempts": "xp_attempts"})

    return pat_summary

def get_fg_summary_k(fg_df):
    # Define the bins and labels
    bins = [0, 39, 49, 59, float('inf')]
    labels = ['0-39', '40-49', '50-59', '60+']

    # Bin distances
    fg_df['distance_range'] = pd.cut(fg_df['kick_distance'], bins=bins, labels=labels, right=True)

    # ---- 1. Field Goal Attempts ----
    fg_attempts = (
        fg_df
        .groupby(['kicker_player_id', 'kicker_player_name', 'distance_range'], observed=True)
        .size()
        .reset_index(name='attempts')
    )

    # ---- 2. Field Goals Made ----
    fg_made = (
        fg_df[fg_df['field_goal_result'] == 'made']
        .groupby(['kicker_player_id', 'kicker_player_name', 'distance_range'], observed=True)
        .size()
        .reset_index(name='made')
    )

    # ---- 3. Merge Attempts and Makes ----
    fg_stats = pd.merge(fg_attempts, fg_made,
                        on=['kicker_player_id', 'kicker_player_name', 'distance_range'],
                        how='left')

    fg_stats['made'] = fg_stats['made'].fillna(0).astype(int)
    fg_stats['fg_pct'] = fg_stats['made'] / fg_stats['attempts']

    # ---- 4. Pivot Summary Tables ----
    # Attempts table
    fg_summary = fg_stats.pivot_table(
        index=['kicker_player_id', 'kicker_player_name'],
        columns='distance_range',
        values='attempts',
        fill_value=0
    ).reset_index()

    # Ensure all distance columns exist
    for label in labels:
        if label not in fg_summary.columns:
            fg_summary[label] = 0
    
    fg_summary['fg_attempts'] = fg_summary[labels].sum(axis=1)

    # FG% table
    fg_pct = fg_stats.pivot_table(
        index=['kicker_player_id', 'kicker_player_name'],
        columns='distance_range',
        values='fg_pct'
    ).reset_index()

    # Ensure all FG% columns exist
    for label in labels:
        if label not in fg_pct.columns:
            fg_pct[label] = 0

    # Rename FG% columns for clarity
    fg_pct.columns = ['kicker_player_id', 'kicker_player_name'] + [f'{label}_FG%' for label in labels]

    # ---- 5. Combine Attempts and FG% ----
    fg_summary = pd.merge(fg_summary, fg_pct, on=['kicker_player_id', 'kicker_player_name'])

    # ---- 6. Sort ----
    fg_summary = fg_summary.sort_values('fg_attempts', ascending=False)

    return fg_summary

def get_kicker_summary(fg_summary,pat_summary,fantasy_summary):
    # Merge the fg & pat summary tables
    kicker_summary = pd.merge(
        fg_summary,
        pat_summary,
        on=['kicker_player_id', 'kicker_player_name'],
        how='outer'  # use 'outer' to keep all kickers even if they only kicked PATs or FGs
    )

    # Optional: Fill missing values with 0 (e.g., for kickers who didn't attempt PATs or FGs)
    kicker_summary = kicker_summary.fillna({
        '0-39': 0, '40-49': 0, '50-59': 0, '60+': 0,
        'fg_attempts': 0,
        '0-39 FG%': 0, '40-49 FG%': 0, '50-59 FG%': 0, '60+ FG%': 0,
        'xp_attempts': 0, 'xp_made': 0, 'xp_pct': 0
    }).sort_values('fg_attempts', ascending=False)
    
    # Merge fantasy points with kicker summary
    full_kicker_summary = pd.merge(
        kicker_summary,
        fantasy_summary,
        on=['kicker_player_id', 'kicker_player_name'],
        how='left'  # Keep all kickers, even if they had 0 fantasy points
    )

    # Fill NaNs for fantasy point columns with 0
    full_kicker_summary[['fg_fantasy_pts', 'pat_fantasy_pts', 'total_fantasy_pts']] = (
        full_kicker_summary[['fg_fantasy_pts', 'pat_fantasy_pts', 'total_fantasy_pts']].fillna(0)
    )

    # Sort by total fantasy points
    full_kicker_summary = full_kicker_summary.sort_values('total_fantasy_pts', ascending=False)

    return full_kicker_summary

def get_fantasy_summary_k(fg_df,pat_df,kick_df):
    # Summarize per kicker
    fg_fantasy = (
        fg_df
        .groupby(['kicker_player_id', 'kicker_player_name'], observed=True)['fantasy_pts']
        .sum()
        .reset_index()
        .rename(columns={'fantasy_pts': 'fg_fantasy_pts'})
    )

    # Summarize per kicker
    pat_fantasy = (
        pat_df
        .groupby(['kicker_player_id', 'kicker_player_name'], observed=True)['fantasy_pts']
        .sum()
        .reset_index()
        .rename(columns={'fantasy_pts': 'pat_fantasy_pts'})
    )

    # Count number of unique games per kicker (using both FG and PAT plays)
    kicker_games = (
        kick_df[['kicker_player_id', 'kicker_player_name', 'game_id']]
        .dropna(subset=['kicker_player_id'])  # exclude rows without kicker info
        .drop_duplicates()
        .groupby(['kicker_player_id', 'kicker_player_name'])
        .size()
        .reset_index(name='games_played')
    )

    # Determine team for each kicker (most common team in case of multiple)
    kicker_teams = (
        kick_df
        .dropna(subset=['kicker_player_id', 'posteam'])
        .groupby(['kicker_player_id', 'kicker_player_name'])['posteam']
        .agg(lambda x: x.mode().iloc[0])  # Use mode (most common team)
        .reset_index()
    )

    # --- Merge all ---
    fantasy_summary = (
        pd.merge(fg_fantasy, pat_fantasy, on=['kicker_player_id', 'kicker_player_name'], how='outer')
        .merge(kicker_games, on=['kicker_player_id', 'kicker_player_name'], how='left')
        .merge(kicker_teams, on=['kicker_player_id', 'kicker_player_name'], how='left')
        .fillna(0)
    )

    fantasy_summary['total_fantasy_pts'] = fantasy_summary['fg_fantasy_pts'] + fantasy_summary['pat_fantasy_pts']
    
    
    # Sort
    fantasy_summary = fantasy_summary.sort_values('total_fantasy_pts', ascending=False)

    return fantasy_summary

def get_pergame_k(df_k):
    # Safeguard: avoid division by zero
    df_k['games_played'] = df_k['games_played'].replace(0, pd.NA)

    # Add per-game attempt rates
    df_k['0-39 pg']   = df_k['0-39'] / df_k['games_played']
    df_k['40-49 pg']  = df_k['40-49'] / df_k['games_played']
    df_k['50-59 pg']  = df_k['50-59'] / df_k['games_played']
    df_k['60+ pg']    = df_k['60+'] / df_k['games_played']
    df_k['attempts pg'] = df_k['attempts'] / df_k['games_played']

    # Optional: round for readability
    df_k[['0-39 pg', '40-49 pg', '50-59 pg', '60+ pg', 'attempts pg']] = df_k[
        ['0-39 pg', '40-49 pg', '50-59 pg', '60+ pg', 'attempts pg']
    ].round(2)

    # Preview
    return df_k


def get_pat_summary_def(pat_df):
    # ---- 1. Extra Point Attempts ----
    pat_attempts = (
        pat_df
        .groupby(['defteam'], observed=True)
        .size()
        .reset_index(name='attempts')
    )

    # ---- 2. Extra Points Made ----
    pat_made = (
        pat_df[pat_df['extra_point_result'] == 'good']
        .groupby(['defteam'], observed=True)
        .size()
        .reset_index(name='made')
    )

    # ---- 3. Merge and Calculate XP% ----
    pat_def = pd.merge(pat_attempts, pat_made, on=['defteam'], how='left')
    pat_def['made'] = pat_def['made'].fillna(0).astype(int)
    pat_def['xp_pct'] = pat_def['made'] / pat_def['attempts']

    # ---- 4. Sort ----
    pat_def = pat_def.sort_values('attempts', ascending=False)

    #Rename 'made' and 'attempts' to avoid confusion with field goals
    pat_def = pat_def.rename(columns={"made": "xp_made", "attempts": "xp_attempts"})

    
    return pat_def

def get_fg_summary_def(fg_df):
    # Define the bins and labels
    bins = [0, 39, 49, 59, float('inf')]
    labels = ['0-39', '40-49', '50-59', '60+']

    # Bin distances
    fg_df['distance_range'] = pd.cut(fg_df['kick_distance'], bins=bins, labels=labels, right=True)

    # ---- 1. Field Goal Attempts ----
    fg_attempts = (
        fg_df
        .groupby(['defteam', 'distance_range'], observed=True)
        .size()
        .reset_index(name='attempts')
    )

    # ---- 2. Field Goals Made ----
    fg_made = (
        fg_df[fg_df['field_goal_result'] == 'made']
        .groupby(['defteam', 'distance_range'], observed=True)
        .size()
        .reset_index(name='made')
    )

    # ---- 3. Merge Attempts and Makes ----
    fg_stats = pd.merge(fg_attempts, fg_made,
                        on=['defteam', 'distance_range'],
                        how='left')

    fg_stats['made'] = fg_stats['made'].fillna(0).astype(int)
    fg_stats['fg_pct'] = fg_stats['made'] / fg_stats['attempts']

    # ---- 4. Pivot Summary Tables ----
    # Attempts table
    fg_summary = fg_stats.pivot_table(
        index=['defteam'],
        columns='distance_range',
        values='attempts',
        fill_value=0
    ).reset_index()

    # Ensure all distance columns exist
    for label in labels:
        if label not in fg_summary.columns:
            fg_summary[label] = 0
            
    fg_summary['fg_attempts'] = fg_summary[labels].sum(axis=1)

    # FG% table
    fg_pct = fg_stats.pivot_table(
        index=['defteam'],
        columns='distance_range',
        values='fg_pct'
    ).reset_index()

    # Ensure all FG% columns exist
    for label in labels:
        if label not in fg_pct.columns:
            fg_pct[label] = 0

    # Rename FG% columns for clarity
    fg_pct.columns = ['defteam'] + [f'{label}_FG%' for label in labels]

    # ---- 5. Combine Attempts and FG% ----
    fg_summary = pd.merge(fg_summary, fg_pct, on=['defteam'])

    # ---- 6. Sort ----
    fg_summary = fg_summary.sort_values('fg_attempts', ascending=False)

    return fg_summary

def get_def_summary(fg_summary,pat_summary,fantasy_summary):
    # Merge the fg & pat summary tables
    def_summary = pd.merge(
        fg_summary,
        pat_summary,
        on=['defteam'],
        how='outer'  # use 'outer' to keep all defs even if they only kicked PATs or FGs
    )

    # Optional: Fill missing values with 0 (e.g., for teams who didn't attempt PATs or FGs)
    def_summary = def_summary.fillna({
        '0-39': 0, '40-49': 0, '50-59': 0, '60+': 0,
        'fg_attempts': 0,
        '0-39 FG%': 0, '40-49 FG%': 0, '50-59 FG%': 0, '60+ FG%': 0,
        'xp_attempts': 0, 'xp_made': 0, 'xp_pct': 0
    }).sort_values('fg_attempts', ascending=False)
    
    # Merge fantasy points with defense summary
    full_def_summary = pd.merge(
        def_summary,
        fantasy_summary,
        on=['defteam'],
        how='left'  # Keep all defs, even if they had 0 fantasy points
    )

    # Fill NaNs for fantasy point columns with 0
    full_def_summary[['fg_fantasy_pts', 'pat_fantasy_pts', 'total_fantasy_pts']] = (
        full_def_summary[['fg_fantasy_pts', 'pat_fantasy_pts', 'total_fantasy_pts']].fillna(0)
    )

    # Sort by total fantasy points
    full_def_summary = full_def_summary.sort_values('total_fantasy_pts', ascending=False)

    return full_def_summary

def get_fantasy_summary_def(fg_df,pat_df,kick_df):
    # Summarize per defense
    fg_fantasy = (
        fg_df
        .groupby(['defteam'], observed=True)['fantasy_pts']
        .sum()
        .reset_index()
        .rename(columns={'fantasy_pts': 'fg_fantasy_pts'})
    )

    # Summarize per defense
    pat_fantasy = (
        pat_df
        .groupby(['defteam'], observed=True)['fantasy_pts']
        .sum()
        .reset_index()
        .rename(columns={'fantasy_pts': 'pat_fantasy_pts'})
    )

    # Count number of unique games per defense
    defense_games = (
        kick_df[['defteam', 'game_id']]
        .dropna(subset=['defteam'])  # remove rows without defensive team
        .drop_duplicates()
        .groupby('defteam')
        .size()
        .reset_index(name='games_played')
    )


    # --- Combine FG and PAT fantasy points ---
    fantasy_summary = pd.merge(fg_fantasy, pat_fantasy,
                               on=['defteam'],
                               how='outer').fillna(0)

    # Add games played
    fantasy_summary = pd.merge(fantasy_summary, defense_games,
                               on='defteam',
                               how='left')

    fantasy_summary['total_fantasy_pts'] = fantasy_summary['fg_fantasy_pts'] + fantasy_summary['pat_fantasy_pts']

    # Sort
    fantasy_summary = fantasy_summary.sort_values('total_fantasy_pts', ascending=False)

    return fantasy_summary

def get_pergame_def(df_k):
    # Safeguard: avoid division by zero
    df_k['games_played'] = df_k['games_played'].replace(0, pd.NA)

    # Add per-game attempt rates
    df_k['0-39 pg']   = df_k['0-39'] / df_k['games_played']
    df_k['40-49 pg']  = df_k['40-49'] / df_k['games_played']
    df_k['50-59 pg']  = df_k['50-59'] / df_k['games_played']
    df_k['60+ pg']    = df_k['60+'] / df_k['games_played']
    df_k['attempts pg'] = df_k['attempts'] / df_k['games_played']

    # Optional: round for readability
    df_k[['0-39 pg', '40-49 pg', '50-59 pg', '60+ pg', 'attempts pg']] = df_k[
        ['0-39 pg', '40-49 pg', '50-59 pg', '60+ pg', 'attempts pg']
    ].round(2)

    # Preview
    return df_k

def estimate_KDEF_matchup(kicker_name, defteam_name, kicker_row, def_row):
    #Predict the fantasy points scored by a kicker given the K & DEF per game stats
    #Naive average model: Expected attempts = avg(kicker + def), expected makes = kicker % * expected attempts
    
    # FG distance ranges and point values
    ranges = ['0-39', '40-49', '50-59', '60+']
    points_per_make = {'0-39': 3, '40-49': 4, '50-59': 5, '60+': 6}
    #TODO: need to account for -1 pts for missed FG
    #TODO: expected misses = expected_attempts - expected_makes
    
    expected_fg_attempts = {}
    expected_fg_makes = {}
    expected_fg_points = {}
    for r in ranges:
        kicker_pg = kicker_row[f"{r}"]
        def_pg = def_row[f"{r}"]
        avg_attempts = (kicker_pg + def_pg) / 2
        expected_fg_attempts[f"{r} expected"] = round(avg_attempts, 3)
        
        fg_pct = kicker_row[f"{r}_FG%"]
        makes = avg_attempts * fg_pct
        misses = avg_attempts * (1 - fg_pct)
        
        expected_fg_makes[f"{r} makes"] = round(makes, 3)
        fantasy_pts = (makes * points_per_make[r]) + (misses * -1)
        expected_fg_points[f"{r} fantasy"] = round(fantasy_pts, 3)
        
    # --- PAT projection ---
    kicker_pat_pg = kicker_row['xp_attempts']
    def_pat_pg = def_row['xp_attempts']
    avg_pat_attempts = (kicker_pat_pg + def_pat_pg) / 2
    
    kicker_pat_pct = kicker_row['xp_pct']
    makes = avg_pat_attempts * kicker_pat_pct
    misses = avg_pat_attempts * (1 - kicker_pat_pct)

    # 1 pt per make, -1 per miss
    expected_pat_points = round(makes * 1 - misses * 1, 3)
    

    # --- Total ---
    total_fg_points = round(sum(expected_fg_points.values()), 2)
    total_expected_points = round(total_fg_points + expected_pat_points, 2)

    return {
        'expected_fg_attempts': expected_fg_attempts,
        'expected_fg_makes': expected_fg_makes,
        'expected_fg_points': expected_fg_points,
        'expected_pat_attempts': round(avg_pat_attempts, 3),
        'expected_pat_points': expected_pat_points,
        'total_expected_fantasy_points': total_expected_points
    }

def compare_kicker_matchups(matchups, df_k, df_def):
    results = []

    for kicker_name, defteam_name in matchups:
        try:
            proj = estimate_KDEF_matchup(kicker_name, defteam_name, df_k, df_def)
            result = {
                'Kicker': kicker_name,
                'Opponent': defteam_name,
                '0-39 FGA': proj['expected_fg_attempts']['0-39 expected'],
                '40-49 FGA': proj['expected_fg_attempts']['40-49 expected'],
                '50-59 FGA': proj['expected_fg_attempts']['50-59 expected'],
                '60+ FGA': proj['expected_fg_attempts']['60+ expected'],
                'PAT Attempts': proj['expected_pat_attempts'],
                'FG Points': sum(proj['expected_fg_points'].values()),
                'PAT Points': proj['expected_pat_points'],
                'Total Fantasy Points': proj['total_expected_fantasy_points']
            }
            results.append(result)
        except Exception as e:
            print(f"Skipping matchup {kicker_name} vs {defteam_name}: {e}")
    
    return pd.DataFrame(results).sort_values('Total Fantasy Points', ascending=False).reset_index(drop=True)

def get_weekly_schedule(year, week):
    """Fetch the schedule and return matchups for a given week."""
    schedule = nfl.import_schedules([year])
    return schedule[schedule['week'] == week]

def generate_weekly_matchups(df_k, schedule_df, verbose=False):
    """Build (kicker_name, defteam_abbr) matchups based on weekly schedule and output as df."""
    matchups = []

    # Keep only unique kickers and teams for the week
    unique_kickers = df_k[['kicker_player_name', 'posteam']].drop_duplicates()
    
    for _, kicker in unique_kickers.iterrows():
        kicker_name = kicker['kicker_player_name']
        team = kicker['posteam']

        # Find the game this team is playing in
        game = schedule_df[
            (schedule_df['home_team'] == team) | 
            (schedule_df['away_team'] == team)
        ]

        if not game.empty:
            game = game.iloc[0]
            opponent = game['away_team'] if team == game['home_team'] else game['home_team']
            matchups.append((kicker_name, opponent))
        else:
            if verbose:
                print(f"️ No game found for kicker {kicker_name} ({team})")
    
    return pd.DataFrame(matchups, columns=['kicker_player_name', 'defteam'])

def top_k_accuracy(df, k=5):
    #Check how often predicted top k players are actual top k players
    correct_count = 0
    total_weeks = df['week'].nunique()
    
    for week in df['week'].unique():
        week_df = df[df['week'] == week]

        top_k_pred = week_df.nlargest(k, 'predicted_fp')['kicker'].tolist()
        top_k_actual = week_df.nlargest(k, 'actual_fp')['kicker'].tolist()
        
        correct = len(set(top_k_pred) & set(top_k_actual))
        correct_count += correct / k  # Normalize per week

    return correct_count / total_weeks

def weekly_spearman_corr(df):
    #Calculate correlation between prediction & actual rankings
    results = []

    for week in df['week'].unique():
        week_df = df[df['week'] == week][['predicted_fp', 'actual_fp']].dropna()
        if len(week_df) >= 2:
            corr, _ = spearmanr(week_df['predicted_fp'], week_df['actual_fp'])
            results.append(corr)
        else:
            results.append(None)  # Not enough data for correlation
    return results

def mean_absolute_error(y_true, y_pred):
    filtered = [(a, p) for a, p in zip(y_true, y_pred) if pd.notna(a) and pd.notna(p)]
    if not filtered:
        return None
    errors = [abs(a - p) for a, p in filtered]
    return sum(errors) / len(errors)

def plot_weekly_spearman_corr(corrs, save_path='Fig/weekly_spearman_corr.png'):
    # Remove None values and track their indices (weeks)
    valid_corrs = [(i + 1, c) for i, c in enumerate(corrs) if c is not None]
    if not valid_corrs:
        print("No valid correlations to plot.")
        return

    weeks, values = zip(*valid_corrs)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(weeks, values, marker='o', linestyle='-')
    plt.title("Weekly Spearman Correlation (Predicted vs Actual FP Rankings)")
    plt.xlabel("Week")
    plt.ylabel("Spearman Correlation")
    plt.ylim(-1.05, 1.05)
    plt.xticks(range(min(weeks), max(weeks) + 1)) 
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to {save_path}")

def plot_rel_error_hist(df, n_bins=30, save_path='Fig/rel_error_hist.png'):
    # Drop NaN or infinite values to avoid plotting issues
    rel_errors = df['rel_error'].replace([float('inf'), -float('inf')], pd.NA).dropna()

    if rel_errors.empty:
        print("No valid relative error data to plot.")
        return

    # Compute stats
    mean_val = rel_errors.mean()
    std_val = rel_errors.std()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.hist(rel_errors, bins=n_bins, color='skyblue', edgecolor='black')

    # Add mean and std lines
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_val:.3f}')
    plt.axvline(mean_val + std_val, color='green', linestyle='dotted', linewidth=2, label=f'+1 SD = {mean_val + std_val:.3f}')
    plt.axvline(mean_val - std_val, color='green', linestyle='dotted', linewidth=2, label=f'-1 SD = {mean_val - std_val:.3f}')

    plt.title("Distribution of Relative Error")
    plt.xlabel("Relative Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Histogram saved to {save_path}")
