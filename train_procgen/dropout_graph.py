import csv
from matplotlib import pyplot as plt

dropout_file = 'dropout_log/progress_total_timesteps_5000000_num_levels_50.csv'
baseline_file = 'log/progress_total_timesteps_5000000_num_levels_50.csv'

f = open(dropout_file)
csv_f = csv.reader(f)

idx = 0
d_training_rew = []
d_testing_rew = []
d_timesteps = []
for row in csv_f:
    if idx != 0 :
        d_training_rew.append(float(row[1]))
        d_testing_rew.append(float(row[3]))
        d_timesteps.append(float(row[-1]))
    else:
        idx += 1

f = open(baseline_file)
csv_f = csv.reader(f)

idx = 0
b_training_rew = []
b_testing_rew = []
b_timesteps = []
for row in csv_f:
    if idx != 0 :
        b_training_rew.append(float(row[1]))
        b_testing_rew.append(float(row[3]))
        b_timesteps.append(float(row[-1]))
    else:
        idx += 1


plt.figure()
plt.title("Testing Reward of Dropout vs Baseline")
plt.xlabel("timesteps")
plt.ylabel("reward")
plt.plot(d_timesteps,d_testing_rew,label="dropout reward")
plt.plot(b_timesteps,b_testing_rew,label="baseline reward")
plt.legend()
plt.show()
