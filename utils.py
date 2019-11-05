import numpy as np
import pandas as pd

def mae(pred, targ): return (pred-targ).abs().mean()

def normalize(x,m,s):
    return (x-m)/s

def denormalize(x,m,s):
    return (x*s)+m

def create_ts_dataset(data, lookback=10):
    xs,ys = [],[]
    for i in range(lookback, len(data)):
        xs.append(data[i-lookback:i])
        ys.append(data[i])
    X = np.array(xs)
    y = np.array(ys)
    return X, y

def get_sin():
    return np.sin(np.arange(50,step=0.1))

# from feature_engineering.ipynb

def get_features(player_df):
    player_df = player_df.assign(
        target=player_df['PTS'].shift(-1),
        home=player_df['MATCHUP'].apply(lambda x: x if type(x)==np.float else int(x.split()[1]=='vs.')).shift(-1),
    )

    features = player_df[['target', 'home', 'MIN', 'FGA', 'FG_PCT', 'PTS']].copy()

    ewma = features[['MIN', 'FGA', 'FG_PCT', 'PTS']].ewm(alpha=0.1).mean()
    ewma.columns = ['MIN_ewma', 'FGA_ewma', 'FG_PCT_ewma', 'PTS_ewma']
    features = pd.concat([features, ewma], axis=1)

    features.dropna(axis=0, inplace=True)

    return features