import pandas as pd
import numpy as np
import os
import requests
import json
import datetime
import time
import glob
import collections
import matplotlib.pyplot as plt


json_dir = '/home/harrison/School/rl/project/kaggle/episodes'

def exists(x):
    path = os.path.join(json_dir, "episode_%i.json" % x)
    try:
        open(path, 'r') 
        return True
    except:
        return False

episodes_df = pd.read_csv("scripts/final_solution_files/agent_selection_20211201.csv")


episodes_df['exists'] = episodes_df['EpisodeId'].apply(exists)

episodes_df = episodes_df[episodes_df['exists']]
print(episodes_df)

episodes_df.to_csv("episodes_list.csv")

