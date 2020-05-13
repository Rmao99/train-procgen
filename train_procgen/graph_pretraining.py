import csv
from matplotlib import pyplot as plt

files_50 = ['log/progress_total_timesteps_5000000_num_levels_50.csv', 'log/progress_total_timesteps_10000000_num_levels_50.csv','log/progress_total_timesteps_50000000_num_levels_50.csv']
files_100 = ['log/progress_total_timesteps_5000000_num_levels_100.csv', 'log/progress_total_timesteps_10000000_num_levels_100.csv','log/progress_total_timesteps_50000000_num_levels_100.csv']
files_250 =  ['log/progress_total_timesteps_5000000_num_levels_250.csv', 'log/progress_total_timesteps_10000000_num_levels_250.csv','log/progress_total_timesteps_50000000_num_levels_250.csv']
files_500 =  ['log/progress_total_timesteps_5000000_num_levels_500.csv', 'log/progress_total_timesteps_10000000_num_levels_500.csv','log/progress_total_timesteps_50000000_num_levels_500.csv']


idx = [50,100,250,500]
i = 0
total_files = [files_50,files_100,files_250,files_500]
pretraining_steps = [5,10,50] #in millions

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
    
    plt.title("Training/Testing Rewards of pretraining steps for " + str(idx[i]) + " training levels")
    plt.xlabel("Pretraining steps in millions")
    plt.ylabel("Reward")
    plt.plot(pretraining_steps,training_rew,label="training reward")
    plt.plot(pretraining_steps,testing_rew,label="testing reward")
    plt.legend()
    plt.savefig("graphs/pretraining_" + str(idx[i]) + ".png")
    plt.close()
    i += 1
