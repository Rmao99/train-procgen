import csv
from matplotlib import pyplot as plt

files_5M = ['log/progress_total_timesteps_5000000_num_levels_50.csv', 'log/progress_total_timesteps_5000000_num_levels_100.csv','log/progress_total_timesteps_5000000_num_levels_250.csv','log/progress_total_timesteps_5000000_num_levels_500.csv']
files_10M   = ['log/progress_total_timesteps_10000000_num_levels_50.csv', 'log/progress_total_timesteps_10000000_num_levels_100.csv','log/progress_total_timesteps_10000000_num_levels_250.csv', 'log/progress_total_timesteps_10000000_num_levels_500.csv']
files_50M =  ['log/progress_total_timesteps_50000000_num_levels_50.csv', 'log/progress_total_timesteps_50000000_num_levels_100.csv','log/progress_total_timesteps_50000000_num_levels_250.csv', 'log/progress_total_timesteps_50000000_num_levels_500.csv']


idx = [5,10,50]
i = 0
total_files = [files_5M, files_10M,files_50M]
num_levels = [50,100,250,500] 

for files in total_files:
    training_rew = []
    testing_rew = []
    for fs in files:
        f = open(fs)
        csv_f = csv.reader(f)
        temp_train = 0
        temp_test = 0
        for row in csv_f:
            if row[1] != 'eprewmean':
                temp_train = float(row[1])
                temp_test = float(row[3])
        training_rew.append(temp_train)
        testing_rew.append(temp_test)
    
    plt.title("Training/Testing Rewards of training levels for " + str(idx[i]) + "M pretraining steps")
    plt.xlabel("Num training levels")
    plt.ylabel("Reward")
    plt.plot(num_levels,training_rew,label="training reward")
    plt.plot(num_levels,testing_rew,label="testing reward")
    plt.legend()
    plt.savefig("graphs/numlevels_" + str(idx[i]) + "M.png")
    plt.close()
    i += 1
