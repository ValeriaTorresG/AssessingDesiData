import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.style.use('./data/plots/desi.mplstyle')


def main(npz:str, h5_dir:str, night:str, tile:str, output:str):
    os.makedirs(output, exist_ok=True)

    data = np.load(npz)
    ids, petals = data['ids'], data['petals']
    outlier_mask = data['outlier_mask']

    fig, ax = plt.subplots(figsize=(5, 5))

    for petal_id in range(10):
        h5_path = os.path.join(h5_dir,
                               f'{night}-{tile}-{petal_id}.h5')
        if not os.path.isfile(h5_path):
            # print(f'Couldnt find {h5_path}')
            continue

        with h5py.File(h5_path, 'r') as f:
            x_all = f['metadata/fiber_x'][:]
            y_all = f['metadata/fiber_y'][:]
            tid_all = f['metadata/target_id'][:]

        ax.scatter(x_all, y_all, s=2, c='lightgrey', zorder=5)

        idx_tile = np.where((petals == petal_id) & outlier_mask)[0]
        if idx_tile.size > 0:
            out_ids = ids[idx_tile]
            local_pos = np.nonzero(np.in1d(tid_all, out_ids))[0]
            ax.scatter(x_all[local_pos], y_all[local_pos],
                       s=5, c='black', edgecolor='black',
                       linewidth=0.1, zorder=10, label='Outliers')

        mean_x, mean_y = x_all.mean(), y_all.mean()
        d = np.sqrt((x_all - mean_x)**2 + (y_all - mean_y)**2)

        edge_x, edge_y = x_all[np.argmax(d)], y_all[np.argmax(d)]
        label_x = mean_x - 0.8 * (edge_x - mean_x)
        label_y = mean_y - 0.8 * (edge_y - mean_y)

        ax.text(label_x, label_y, str(petal_id),
                fontsize=13, color='black', ha='center',
                va='center', zorder=20)

    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.margins(x=0.21, y=0.21)
    plt.grid(linewidth=0.2, zorder=0)
    plt.title(f'{night} - {tile}', y=1.05, fontdict={'fontsize': 13})

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=11)

    plt.xlabel('Fiber X [mm]', fontsize=12)
    plt.ylabel('Fiber Y [mm]', fontsize=12)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{output}/fibers_{night}_{tile}', dpi=200)


if __name__ == '__main__':
    main()