import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim
import matplotlib.colors as colors
import numpy as np
import pandas as pd

def load_csv(filename):
    """Load a CSV file and return a 2D numpy array."""
    data = np.loadtxt(filename, delimiter=',')
    return data

def displayAsImage(scoreSnaps, totalScore, ruleSnaps, matrix):
    fig1, ((ax1,ax2,fax1,fax2)
    ,(ax3,ax4,fax3,fax4)
    ,(ax5,ax6,fax5,fax6)
    ,(ax7,ax8,ax9,ax10)) = plt.subplots(4,4,figsize=(16,8))

    mspf = 120 # ms per frame
    im = ax1.imshow(scoreSnaps[0], cmap="viridis", vmin=0, vmax=np.nanmax(scoreSnaps[-1]))
    ax1.set_title("Score Evolution")
    plt.colorbar(im, ax=ax1)
    ani = anim.FuncAnimation(fig=fig1, func=lambda frame: im.set_data(scoreSnaps[frame]), frames=len(scoreSnaps), interval=mspf)
    ax2.imshow(totalScore, cmap="magma", vmin=0, vmax=np.nanmax(totalScore))
    ax2.set_title("Final Score")

    ruleSnaps = np.array(ruleSnaps)
    #print(ruleSnaps.shape)

    # Agent snaps:
    ruleR1s = ruleSnaps[:,:,:,0]
    #print(ruleR1s)
    ruleR2s = ruleSnaps[:,:,:,1]
    ruleR3s = ruleSnaps[:,:,:,2]
    ruleR4s = ruleSnaps[:,:,:,3]
    mutationRate = ruleSnaps[:,:,:,4]
    # N = 1:
    # R1 = DD, R2 = DC, R3 = CD, R4 = CC
    fig1.suptitle(f"Rule structure: Their move, My move (both from previous round). Color: Bluer => more likely to defect, Yellower => more likely to cooperate\nPayoff Matrix: \n{matrix}")
    im1 = ax3.imshow(ruleR1s[0], cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im1, ax=ax3)
    ani1 = anim.FuncAnimation(fig=fig1, func=lambda frame: im1.set_data(ruleR1s[frame]), frames=len(scoreSnaps), interval=mspf)
    ax3.set_title("R1(DD) evolution")

    im2 = ax4.imshow(ruleR2s[0], cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax4)
    ani2 = anim.FuncAnimation(fig=fig1, func=lambda frame: im2.set_data(ruleR2s[frame]), frames=len(scoreSnaps), interval=mspf)
    ax4.set_title("R2(DC) evolution")

    im3 = ax5.imshow(ruleR3s[0], cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im3, ax=ax5)
    ani3 = anim.FuncAnimation(fig=fig1, func=lambda frame: im3.set_data(ruleR3s[frame]), frames=len(scoreSnaps), interval=mspf)
    ax5.set_title("R3(CD) evolution")

    im4 = ax6.imshow(ruleR4s[0], cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im4, ax=ax6)
    ani4 = anim.FuncAnimation(fig=fig1, func=lambda frame: im4.set_data(ruleR4s[frame]), frames=len(scoreSnaps), interval=mspf)
    ax6.set_title("R4(CC) evolution")
    
    im5 = fax1.imshow(mutationRate[0], cmap="viridis", vmin=0, vmax=np.nanmax(mutationRate[-1]))
    plt.colorbar(im4, ax=fax1)
    animr = anim.FuncAnimation(fig=fig1, func=lambda frame: im5.set_data(mutationRate[frame]), frames=len(scoreSnaps), interval=mspf)
    fax1.set_title("Mutation rate evolution")
    frm = fax2.imshow(mutationRate[-1], cmap="viridis", vmin=0, vmax=np.nanmax(mutationRate[-1]))
    fax2.set_title("Mutation rate final")

    fr1 = fax3.imshow(ruleR1s[-1], cmap="viridis", vmin=0, vmax=1)
    fax3.set_title("R1(DD) final")
    fr2 = fax4.imshow(ruleR2s[-1], cmap="viridis", vmin=0, vmax=1)
    fax4.set_title("R2(DC) final")
    fr3 = fax5.imshow(ruleR3s[-1], cmap="viridis", vmin=0, vmax=1)
    fax5.set_title("R3(CD) final")
    fr4 = fax6.imshow(ruleR4s[-1], cmap="viridis", vmin=0, vmax=1)
    fax6.set_title("R4(CC) final")
    
    ax7.imshow(ruleR1s[0], cmap="viridis", vmin=0, vmax=1)
    ax7.set_title("R1(DD) initial")
    ax8.imshow(ruleR2s[0], cmap="viridis", vmin=0, vmax=1)
    ax8.set_title("R2(DC) initial")
    ax9.imshow(ruleR3s[0], cmap="viridis", vmin=0, vmax=1)
    ax9.set_title("R3(CD) initial")
    ax10.imshow(ruleR4s[0], cmap="viridis", vmin=0, vmax=1)
    ax10.set_title("R4(CC) initial")

    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")
    ax5.axis("off")
    ax6.axis("off")
    ax7.axis("off")
    ax8.axis("off")
    ax9.axis("off")
    ax10.axis("off")
    fax1.axis("off")
    fax2.axis("off")
    fax3.axis("off")
    fax4.axis("off")
    fax5.axis("off")
    fax6.axis("off")

    plt.show()


def flattenDataToHist(data, max=None):
    histogramX = list(range(len(data)))
    histogramZ = []
    maxScore = max
    if maxScore == None: maxScore = np.nanmax(data[-1])
    bins = np.linspace(0,maxScore,200)
    for snap in data:
        hist, _ = np.histogram(snap, bins)
        histogramZ.append(hist)
    histogramY = []
    for n in range(len(bins)-1):
        histogramY.append((bins[n]+bins[n+1])/2)
    x = np.array(histogramX)
    y = np.array(histogramY)
    z = np.array(histogramZ)
    y, x = np.meshgrid(y, x)
    return x, y, z

def heightmaps(scoreSnaps, totalScore, ruleSnaps, iters):
    # Generate score 3d data
    # x axis is round
    # y is histogram buckets
    # z is height of histogram
    ### Score evolution:
    x, y, z = flattenDataToHist(scoreSnaps, iters*5)
    z_masked = np.ma.masked_where(z == 0, z)
    fig = plt.figure(figsize=(12,8))
    nrows, ncols = 2,3
    score = fig.add_subplot(nrows, ncols, 1, projection="3d")
    score.grid(False)
    score.plot_surface(x, y, z_masked, cmap="gist_gray", lw=0.5, rstride=1, cstride=1)
    score.set_xlabel("round")
    score.set_ylabel("score bin")
    score.view_init(elev=90, azim=0)
    
    ### Rule evolution
    ruleComps = [ruleSnaps[:,:,:,n] for n in range(ruleSnaps.shape[-1])]
    axes = []
    labels = ["Rule 1", "Rule 2", "Rule 3", "Rule 4"]
    for id, (rule, label) in enumerate(zip(ruleComps,labels)):
        x, y, z = flattenDataToHist(rule, max=1)
        z_masked = np.ma.masked_where(z == 0, z)
        plot = fig.add_subplot(nrows, ncols, 2+id, projection="3d")
        plot.grid(False)
        plot.plot_surface(x, y, z_masked, cmap="gist_gray", lw=0.5, rstride=1, cstride=1)
        plot.set_xlabel("round")
        plot.set_ylabel("rule value")
        plot.set_title(label)
        plot.view_init(elev=90, azim=0)
        axes.append(plot)
    plt.show()
    
def stateSpace4d(ruleSnaps):
    x = []
    y = []
    z = []
    colors = []
    
    # R1 = DD, R2 = DC, R3 = CD, R4 = CC
    ruleR1s = ruleSnaps[:,:,:,0]
    ruleR2s = ruleSnaps[:,:,:,1]
    ruleR3s = ruleSnaps[:,:,:,2]
    ruleR4s = ruleSnaps[:,:,:,3]
    
    x = np.array([np.mean(el) for el in ruleR1s])
    print(x)
    y = np.array([np.mean(el) for el in ruleR2s])
    z = np.array([np.mean(el) for el in ruleR3s])
    r4mean = np.array([np.mean(el) for el in ruleR4s])
    cmap = plt.colormaps['viridis']
    colors = cmap(r4mean)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') # Create a 3D subplot
    phase = ax.scatter(x, y, z, c=colors)
    ax.scatter(x[0],y[0],z[0],c="black", linewidths=5)
    ax.set_xlabel("DD mean")
    ax.set_ylabel("DC mean")
    ax.set_zlabel("CD mean")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    ax.set_title("color: CC mean, yellower = closer to 1")
    
    ## PLOT PROJECTIONS ON THE PLANES
    
    plt.show()
    
if __name__ == "__main__":
    # Example: load from CSV files exported by your C++ simulation
    df = pd.read_csv("params.csv", index_col=0)
    params = df['Value'].apply(pd.to_numeric).to_dict()
    snaps = int(params["snaps"])
    N = int(params["gridN"])
    rounds = int(params["rounds"])
    print(params)
    scoreSnaps = load_csv("nonCumulativeScore.csv")  # shape: (frames, grid_y, grid_x)
    totalScore = load_csv("totalScore.csv")      # shape: (grid_y, grid_x)
    ruleSnaps  = load_csv("ruleSnaps.csv")   # shape: (frames, grid_y, grid_x*4)

    # Reshape ruleSnaps if stored flattened
    yLen, xLen = params["gridN"], params["gridN"]
    ruleSnaps = ruleSnaps.reshape(snaps, N, N, 5)
    scoreSnaps = scoreSnaps.reshape(snaps, N, N)
    
    scoreDelta = np.mean(scoreSnaps[-1]) - np.mean(scoreSnaps[0])
    print(f"Score delta: {scoreDelta}")
    
    print(f"Score final mean: {np.mean(scoreSnaps[-1])}")

    displayAsImage(scoreSnaps, totalScore, ruleSnaps)
    heightmaps(scoreSnaps, totalScore, ruleSnaps, params["iters"])