import csv
from matplotlib import pyplot as plt

files = ['dropout_log/progress_total_timesteps_5000000_num_levels_50'] #input string of files to plot here
for f in files:
    csv_f = csv.reader(f)

    training_rew = []
    testing_rew = []
    timesteps = []
    for row in csv_f:
        training_rew.append(row[1])
        testing_rew.append(row[3])
        timesteps.append(row[-1])

    
