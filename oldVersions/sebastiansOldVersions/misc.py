import numpy as np
from Perlin_numpy import *
import pandas as pd
import matplotlib.pyplot as plt

data = generate_fractal_noise_2d((32,32),(4,4),1,0.5,2,(False, False))[0]
#data = data*1.3/2 # Current processing
data += 1
data /= 2
#data /= np.nanmax(data)

df = pd.DataFrame(data.ravel())
print(df.describe())

plt.hist(data.ravel(), 50)
plt.show()