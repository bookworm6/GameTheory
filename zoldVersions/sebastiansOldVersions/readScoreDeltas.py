import numpy as np
import matplotlib.pyplot as plt

def load_csv(filename):
    """Load a CSV file and return a 2D numpy array."""
    data = np.loadtxt(filename, delimiter=',')
    return data

data = load_csv("scoreDeltas.csv")
payoffData = data[1]
scoreDeltas = data[0]
y = scoreDeltas

unique_x = sorted(set(payoffData))
x_positions = list(range(len(unique_x)))
x_mapped = [unique_x.index(x) for x in payoffData]

# plot
plt.figure(figsize=(8,5))
plt.scatter(x_mapped, y, s=60, color='steelblue')

# custom tick positions and labels
plt.xticks(x_positions, [f"{val:g}" for val in unique_x])  # shows 3, 3.01, 3.1, 3.5, 4

plt.xlabel("Original X values (displayed)")
plt.ylabel("Y values")
plt.title("Scatter with custom X tick labels")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

#plt.scatter(payoffData, scoreDeltas)
#plt.xscale("log")
#plt.show()