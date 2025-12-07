import subprocess
import os
from datetime import datetime
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from display import *
import numpy.random as random

# Path to your compiled C++ executable
EXEC_PATH = "./simTest"

# Path to the CSV log file
LOG_FILE = "experiment_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "p00", "p01","p10","p11","gridN","res0","res1","maxN", "rounds", "iters", "snaps", "evolutionRate", "mutationRate", "evolutionChance", "seed", "gridseed", "result_file", "status", "scoreDelta"])

def payoffSweep(start, end, step, index, sweep=None):
    if sweep == None:
        sweep = np.arange(start, end, step)
    print(sweep)
    base = np.array([[1,5],[0,3]], dtype = np.float64)
    baseMatrix = np.vstack([base]*len(sweep))
    baseMatrix = np.reshape(baseMatrix, (len(sweep),2,2))
    baseMatrix[:,index[0], index[1]] = sweep
    trials = np.hstack([[1]]*len(sweep))
    print(trials)
    return baseMatrix, trials

payoffMatrices = [
    [[1,5],[0,3]],[[1,5],[0,3.01]],[[1,5],[0,3.1]],[[1,5],[0,3.5]],[[1,5],[0,4]]
]
sampleRepeats = [1,1,1,1,1]

gridN = 32
res = (4,4)
maxN = 1
rounds = 15000
iters = 60
snaps = 100
evolutionRate = 0.01
evolutionChance = 0.2
mutationRate = 0.001

def load_csv(filename):
    """Load a CSV file and return a 2D numpy array."""
    data = np.loadtxt(filename, delimiter=',')
    return data

scoreDeltas = []
seed2 = 897#int(random.rand()*1000)

payoffMatrices, sampleRepeats = payoffSweep(3.17,3.25,0.001, (1,1))

for matrix, samples in zip(payoffMatrices, sampleRepeats):
    for n in range(samples):
        seed = seed2#int(random.rand()*1000)
        timestamp = datetime.now().isoformat(timespec='seconds')
        result_file = f"results_{timestamp.replace(':', '-')}.csv"
        try:
            # Run your simulation (pass parameters via argv)
            subprocess.run(
                [EXEC_PATH, str(matrix[0][0]), str(matrix[0][1]), str(matrix[1][0]), 
                 str(matrix[1][1]), str(gridN), str(res[0]), str(res[1]),
                 str(maxN), str(rounds), str(iters), str(snaps),
                 str(evolutionRate), str(mutationRate), str(evolutionChance), str(seed), str(seed2), result_file],
                check=True
            )
            status = "success"
        except subprocess.CalledProcessError:
            status = "failed"

        # Analyze Data
        df = pd.read_csv("params.csv", index_col=0)
        params = df['Value'].apply(pd.to_numeric).to_dict()
        snaps = int(params["snaps"])
        N = int(params["gridN"])
        maxN = int(params["maxN"])
        rounds = int(params["rounds"])
        print(params)
        scoreSnaps = load_csv("nonCumulativeScore.csv")  # shape: (frames, grid_y, grid_x)
        totalScore = load_csv("totalScore.csv")      # shape: (grid_y, grid_x)
        ruleSnaps  = load_csv("ruleSnaps.csv")   # shape: (frames, grid_y, grid_x*4)

        # compile with g++ -O3 -std=c++17 sim.cpp -o sim -pthread

        # Reshape ruleSnaps if stored flattened
        yLen, xLen = params["gridN"], params["gridN"]
        ruleSnaps = ruleSnaps.reshape(snaps, N, N, 4**maxN)
        scoreSnaps = scoreSnaps.reshape(snaps, N, N)
        
        scoreDelta = np.mean(scoreSnaps[-1]) - np.mean(scoreSnaps[0])
        scoreDeltas.append(scoreDelta)
        
        displayAsImage(scoreSnaps, totalScore, ruleSnaps, matrix)
        contourPlots(scoreSnaps, totalScore, ruleSnaps, params["iters"])
        
        # Log it
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                matrix[0][0], matrix[0][1], matrix[1][0], 
                matrix[1][1], gridN, res[0], res[1],
                maxN, rounds, iters, snaps,
                evolutionRate, mutationRate, evolutionChance, seed,
                result_file, status, scoreDelta
            ])
payoffs = np.array(payoffMatrices)[:,1,1]
payoffData = []
for payoff, times in zip(payoffs, sampleRepeats):
    for time in range(times):
        payoffData.append(payoff)
with open("scoreDeltas.csv", "a", newline="") as f:
    f.truncate(0)
    writer = csv.writer(f)
    writer.writerow(scoreDeltas)
    writer.writerow(payoffData)
# print(payoffData)
# plt.scatter(payoffData, scoreDeltas)
# plt.show