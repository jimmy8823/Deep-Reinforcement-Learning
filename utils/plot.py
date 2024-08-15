import pandas as pd
import matplotlib.pyplot as plt
from typing import List

def plot_D3QN():
    episodes = []
    steps = []
    mean_rewards = []
    collisions = []
    eps_per100 = range(1,31)
    fail_rate = []
    with open("./result/leanring_result.txt", "r") as fp:
        i = 1
        fail = 0
        for contents in fp:
            contents = contents.replace(" ","")
            contents = contents.strip("\n")
            episode,r_s_c = contents.split(":")
            reward, step, collision = r_s_c.split(",")
            episodes.append(episode)
            mean_rewards.append(float(reward)/int(step))
            steps.append(int(step))
            collisions.append(collision)
            # print(contents)
            if collision=="True" or int(step)==2000:
                fail+=1
            if i==100:
                print(fail)
                fail_rate.append(fail)
                i = 0
                fail = 0
            i+=1


    plot_learning_curve(episodes,mean_rewards,"Mean_rewards","mean_reward",figur_file="./result/mean_rewards.png")

def plot_cmvae_loss():
    itera = []
    t_loss = []
    with open("result.txt", "r") as fp:
        for contents in fp:
            contents = contents.replace(" ","")
            contents = contents.strip("\n")
            iteration, loss = contents.split(",")
            itera.append(int(iteration))
            t_loss.append(float(loss))
    plot_learning_curve(itera,t_loss,"Training Loss","loss",figur_file="training_loss.png")

def plot_two_record(file1,file2,label1,label2,column,xlabel,ylabel,title,s=True):
    file1_csv = pd.read_csv(file1)
    file2_csv = pd.read_csv(file2)
    #step = [x*3 for x in range(1,1001)]
    file1_value = file1_csv[column]
    file2_value = file2_csv[column]
    step = file1_csv['Step']

    if s:
        smoothed_1 = smooth(file1_value,0.99)
        smoothed_2 = smooth(file2_value,0.99)
        plt.plot(step,smoothed_1,label=label1)
        
        plt.plot(step,smoothed_2,label=label2)
    else:
        plt.plot(step,file1_value,label=label1)
        
        plt.plot(step,file2_value,label=label2)
    # naming the x axis
    plt.xlabel(xlabel)
    # naming the y axis
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_three_record(file1,file2,file3,label1,label2,label3,column,xlabel,ylabel,title,s=True):
    file1_csv = pd.read_csv(file1)
    file2_csv = pd.read_csv(file2)
    file3_csv = pd.read_csv(file3)
    #step = [x*3 for x in range(1,1001)]
    file1_value = file1_csv[column]
    file2_value = file2_csv[column]
    file3_value = file3_csv[column]

    step = file1_csv['Step']
    if s:
        smoothed_1 = smooth(file1_value,0.99)
        smoothed_2 = smooth(file2_value,0.99)
        smoothed_3 = smooth(file2_value,0.99)
        plt.plot(step,smoothed_1,label=label1)
        plt.plot(step,smoothed_2,label=label2)
        plt.plot(step,smoothed_3,label=label3)
    else:
        plt.plot(step,file1_value,label=label1)
        plt.plot(step,file2_value,label=label2)
        plt.plot(step,file3_value,label=label3)

    # naming the x axis
    plt.xlabel(xlabel)
    # naming the y axis
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed
"""
plot_two_record(
    file1="D:\\CODE\\Python\\AirSim\\PPO\\run-cmave_.-tag-rollout_ep_rew_mean.csv",
    file2="D:\\CODE\\Python\\AirSim\\PPO\\run-cnn_.-tag-rollout_ep_rew_mean.csv",
    label1="PPO with CmVAE",
    label2="PPO with CNN",
    column="Value",
    xlabel="Timestep",
    ylabel="Reward",
    title="Reward comparison",
    s=False
)

"""

plot_two_record(
    file1="D:\\CODE\\Python\\AirSim\\run-cmave_.-tag-reward.csv",
    file2="D:\\CODE\\Python\\AirSim\\run-cnn_.-tag-reward.csv",
    label1="D3QN with CMVAE",
    label2="D3QN with CNN",
    column="Value",
    xlabel="Per 10 Episodes",
    ylabel="Avg. Reward",
    title="Reward",
    s=False
)
"""
plot_three_record(
    file1="D:\\CODE\\Python\\AirSim\\D3QN_with_cmvae\\run-cmave_terminal_state_Sucess-tag-terminal_state.csv",
    file2="D:\\CODE\\Python\\AirSim\\D3QN_with_cmvae\\run-Collision-tag-terminal_state.csv",
    file3="D:\\CODE\\Python\\AirSim\\D3QN_with_cmvae\\run-Time exceed-tag-terminal_state.csv",
    label1="Success",
    label2="Collision",
    label3="Time Exceed",
    column="Value",
    xlabel="Per 50 Episodes",
    ylabel="Success Times",
    title="Success Rate",
    s=False
)

"""

