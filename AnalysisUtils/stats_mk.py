import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def compute_zscore(arr, axis):
    """Compute z-score of array along given axis."""
    mean = np.nanmean(arr, axis=axis,keepdims=True)
    std_dev = np.nanstd(arr, axis=axis,keepdims=True)
    return (arr - mean) / std_dev

def plot_curves(data, clrmap='viridis', fig_size=(8, 8), tlim=[-2,4]):
    cmap = mpl.cm.get_cmap(clrmap)
    fig = plt.figure(figsize=fig_size)
    dataz = compute_zscore(data, axis=1)
    Ncell = data.shape[0]
    ax = None
    if len(data.shape) == 2:
        NN = 1
    else:
        NN = np.shape(data)[2]
        print(NN)
    for n in range(NN):
        ax = plt.subplot(1, NN, n + 1, frameon=False, sharex=ax)
        for i in range(Ncell):
            if len(data.shape) == 2:
                Y = dataz[i, :]
            else:
                Y = dataz[i, :, n]
            X = np.linspace(tlim[0], tlim[1], len(Y))  
            color = cmap(i / Ncell)
            ax.plot(X, Y + 4*i, color=color, linewidth=1, zorder=100 - i)  

            ax.yaxis.set_tick_params(tick1On=False)
            ax.set_xlim([-0.5,1.0])
            ax.set_ylim([-1, Ncell*4+1])
            ax.axvline(0.0, ls="--", lw=0.5, color="blue", zorder=250,alpha=0.5)
            ax.text(
                0.0,
                1.0,
                "Value %d" % (n + 1),
                ha="right",
                va="top",
                transform=ax.transAxes,
            )
        if n == 0:
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_yticks(np.linspace(0, 4*(Ncell-1), Ncell))
            ax.set_yticklabels(["ID: %d" % i for i in range(1, Ncell+1)])
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(10)
                tick.label.set_verticalalignment("bottom")
        else:
            ax.yaxis.set_tick_params(labelleft=False)
    plt.tight_layout()
    plt.show()