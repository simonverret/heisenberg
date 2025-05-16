import numpy as np
import matplotlib.pyplot as plt

def square_lattice(n_rows=8, n_cols=8, ax=None, size=(6,6)):
    X, Y = size

    show=False
    if ax is None:
        fig, ax = plt.subplots(figsize=(X, Y))
        show = True

    positions = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    ax.scatter(x_coords, y_coords, s=600/(n_rows+n_cols), color='lightgray', edgecolors='black', linewidths=1.5, zorder=2)
    for i in range(n_rows-1):
        for j in range(n_cols):
            ax.plot([i, i+1], [j, j], 'gray', linewidth=2.0, alpha=0.7, zorder=1)
    for i in range(n_rows):
        for j in range(n_cols-1):
            ax.plot([i, i], [j, j+1], 'gray', linewidth=2.0, alpha=0.7, zorder=1)

    np.random.seed(42)
    arrows = np.random.randn(n_rows * n_cols, 2)
    arrows = arrows / np.sqrt(np.sum(arrows**2, axis=1))[:, np.newaxis] * 0.5

    for i in range(len(x_coords)):
        ax.arrow(
            x_coords[i]-arrows[i, 0]*0.6, 
            y_coords[i]-arrows[i, 1]*0.6, 
            arrows[i, 0], 
            arrows[i, 1], 
            head_width=0.1, 
            head_length=0.1, 
            fc='black', ec='black', zorder=3)

    ax.set_xlim(-0.5, n_rows - 0.5)
    ax.set_ylim(-0.5, n_cols - 0.5)
    ax.axis('off')
    ax.set_aspect('equal')
    
    if show:
        plt.tight_layout()
        plt.show()


fig, ax = plt.subplots(3,3, figsize=(10,10))

square_lattice(2,2, size=(2,1), ax=ax[0,0])
square_lattice(4,4, size=(2,1), ax=ax[0,1])
square_lattice(6,6, size=(2,1), ax=ax[0,2])

square_lattice(2,2, size=(4,4), ax=ax[1,0])
square_lattice(4,4, size=(4,4), ax=ax[1,1])
square_lattice(6,6, size=(4,4), ax=ax[1,2])

square_lattice(2,2, ax=ax[2,0])
square_lattice(4,4, ax=ax[2,1])
square_lattice(6,6, ax=ax[2,2])

plt.show()