import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("LOSS.txt", sep=',')

plt.plot(1/df["best_fitness"])
plt.semilogy()
plt.show()
