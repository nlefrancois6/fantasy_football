import ssl
import certifi
import urllib.request

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

print('Loading weekly data')

wk = nfl.import_weekly_data([2024])
print(wk)

print('Loading FTN data')
ftn = nfl.import_ftn_data([2024], thread_requests=False)

# Save to CSV
save_files = True
if save_files:
    print('Saving data to csv')
    pbp.to_csv("Data/pbp_2024.csv", index=False)
    pbp_short.to_csv("Data/pbp_short_2024.csv", index=False)
    wk.to_csv("Data/weekly_2024.csv", index=False)
    ftn.to_csv("Data/ftn_2024.csv", index=False)

print("Files saved")
