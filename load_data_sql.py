import ssl
import certifi
import urllib.request
import sqlite3
import os

print('Setting up certificates')
#Handle certificate issue when accessing nfl-data-py url
ssl_context = ssl.create_default_context(cafile=certifi.where())
# Then monkeypatch urllib to use the context (if using urllib directly)
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
urllib.request.install_opener(opener)

import nfl_data_py as nfl

print('Loading NFL-data-py data')

print('Loading play-by-play data')
pbp = nfl.import_pbp_data([2024])
print(pbp)

pbp_short = pbp.iloc[:50] #Get small example of pbp data for inspection

# Save to CSV
save_files = True
if save_files:
    print('Saving data to SQLite database')
    # Ensure Data directory exists
    os.makedirs("Data", exist_ok=True)

    # Connect to SQLite database (it will be created if it doesn't exist)
    db_path = "Data/nfl_data.db"
    with sqlite3.connect(db_path) as conn:
        # Save full and short PBP data to two different tables
        pbp.to_sql("pbp_2024", conn, if_exists="replace", index=False)
        pbp_short.to_sql("pbp_short_2024", conn, if_exists="replace", index=False)

    print(f"Data saved to {db_path}")

print("Script complete")
