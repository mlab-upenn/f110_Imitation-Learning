import pickle
import random
import matplotlib.pyplot as plt

global_returns = []
with open("out_returns.pkl", 'rb') as f:
    global_returns = pickle.load(f)

plt.plot(global_returns, c='purple')
for i in range(int(len(global_returns)/3)):
    plt.axvline(x=i*3, c='orange', ls=':')

plt.legend(['Reward Curve', 'Dagger Rollout'])
plt.xlabel("Episode")
plt.ylabel("Reward / Time until crash")
plt.show(block=True)