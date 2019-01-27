import numpy as np
import matplotlib.pyplot as plt

d = np.load("./data/train.npy", mmap_mode="r")
ds = d[:70000000:100]

fig, ax1 = plt.subplots()
ax1.plot(range(len(ds)), ds[:,0], "k-")
ax1.set_xlabel("")
ax1.set_ylabel("Acoustic value", color="k")
ax1.tick_params("y", colors="k")

ax2 = ax1.twinx()
ax2.plot(range(len(ds)), ds[:,1], "r--")
ax2.set_xlabel("")
ax2.set_ylabel("Time to failure (s)", color="r")
ax2.tick_params("y", colors="k")

fig.tight_layout()
plt.show()
