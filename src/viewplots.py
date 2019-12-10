import pickle
import random
import matplotlib.pyplot as plt

global_returns = []
with open("out_returns.pkl", 'rb') as f:
    global_returns = pickle.load(f)

plt.plot(global_returns)
for i in range(int(len(global_returns)/3)):
    plt.axvline(x=i*3, c='orange', ls='--')

plt.show(block=True)