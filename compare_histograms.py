import numpy as np

#from pyemd import emd
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import pdist

def main(args):
    data = np.load(args.distributions_file)
    target_dist = data["target_dist"]
    pred_dist = data["pred_dist"]

    # blur the target distribution, slightly
    #target_dist = gaussian_filter(target_dist, 1., truncate=1.)

    #mask = [target_dist == 0]
    #target_dist[mask] = pred_dist[mask]
    mask = [target_dist != 0]
    target_dist = target_dist[mask]
    pred_dist = pred_dist[mask]

    print "Correlation:", (target_dist * pred_dist).sum()

    print "Hellinger distance:", np.sqrt(
        0.5 * np.square(np.sqrt(target_dist) - np.sqrt(pred_dist)).sum())

    print "Kullback-Leibler divergence:", np.sum(
        target_dist * (np.log(target_dist) - np.log(pred_dist)))

    print "Two-way Kullback-Leibler divergence:", (
        (target_dist * (np.log(target_dist) - np.log(pred_dist))).sum() +
        (pred_dist * (np.log(pred_dist) - np.log(target_dist))).sum())

    # build a distance matrix between all bin centers
    #grid = np.stack(np.meshgrid(
    #        np.arange(target_dist.shape[0], dtype=np.float) * args.bin_size,
    #        np.arange(target_dist.shape[1], dtype=np.float) * args.bin_size,
    #        np.arange(target_dist.shape[2], dtype=np.float) * args.bin_size,
    #        np.arange(target_dist.shape[3], dtype=np.float) * args.bin_size,
    #        indexing="ij"),
    #    axis=-1)
    #grid = grid.reshape(-1, 4)
    #print grid.shape
    #exit()
    #distance_matrix = pdist(grid)

    #print emd(target_dist.ravel(), pred_dist.ravel(), distance_matrix)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("distributions_file", type=str)

    parser.add_argument("--bin_size", type=float, default=0.1,
        help="")

    args = parser.parse_args()

    main(args)

