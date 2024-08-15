import os
import matplotlib.pyplot as plt

def create_dir(path:str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + "is already exist!")
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + "create successfully!")

def plot_learning_curve(episodes, records, title, ylabel, figur_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color = 'r')
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figur_file)

