import matplotlib
matplotlib.use("Agg")

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import erf
from baselines import get_baseline_data, get_difficulty_ids

def main(args):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    data = np.load(args.results_file)
    x, target, pred = data["x"], data["target"], data["pred"]

    # only analyze +1.0s, for now
    target = target[:,:,-1,0] # Bx4

    plt.scatter(target[:,0], target[:,1], s=4, edgecolor="none")
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plt.ylim(*ylim[::-1])
    plt.savefig(os.path.join(args.output_folder, "test_dist_xy.pdf"))

    plt.clf()
    plt.scatter(target[:,2], target[:,3], s=4, edgecolor="none")
    wlim = plt.gca().get_xlim()
    hlim = plt.gca().get_ylim()
    plt.ylim(*ylim[::-1])
    plt.savefig(os.path.join(args.output_folder, "test_dist_wh.pdf"))

    # create a grid with (0,0,0,0) in the middle of a voxel
    b = args.bin_size # for convenience
    xmin = np.floor((-0.5 * b + xlim[0]) / b) * b - 0.5 * b
    xmax = np.floor((0.5 * b + xlim[1]) / b) * b + 0.5 * b
    nx = int(np.round((xmax - xmin) / b))
    ymin = np.floor((-0.5 * b + ylim[0]) / b) * b - 0.5 * b
    ymax = np.floor((0.5 * b + ylim[1]) / b) * b + 0.5 * b
    ny = int(np.round((ymax - ymin) / b))
    wmin = np.floor((-0.5 * b + wlim[0]) / b) * b - 0.5 * b
    wmax = np.floor((0.5 * b + wlim[1]) / b) * b + 0.5 * b
    nw = int(np.round((wmax - wmin) / b))
    hmin = np.floor((-0.5 * b + hlim[0]) / b) * b - 0.5 * b
    hmax = np.floor((0.5 * b + hlim[1]) / b) * b + 0.5 * b
    nh = int(np.round((hmax - hmin) / b))

    # bilinear interpolation for the binning
    alpha = (target - (xmin, ymin, wmin, hmin)) / b
    binned_target = alpha.astype(np.int)
    alpha -= binned_target
    alpha = np.dstack((1. - alpha, alpha))

    target_dist = np.zeros((nx + 1, ny + 1, nw + 1, nh + 1))
    for offset in itertools.product(*([(0,1)]*4)):
        print 'offset:', offset
        np.add.at(target_dist,
                  tuple((binned_target + offset).T),
                  np.prod(alpha[:,range(4),offset], axis=-1))
    target_dist = target_dist[:-1, :-1, :-1, :-1]
    target_dist /= len(binned_target) # normalize

    # plot xy marginal
    xy_marginal = target_dist.sum(axis=(2, 3))
    log_xy_marginal = np.log(xy_marginal)
    plt.figure()
    plt.gca().set_facecolor("k")
    im = plt.imshow(log_xy_marginal.T, interpolation="nearest",
                    vmax=-3, vmin=-18,
                    extent=(xmin, xmax, ymax, ymin))
    plt.title("Marginal Log-probability for GT Transformations (x, y)")
    plt.xlabel("$\\Delta x$")
    plt.ylabel("$\\Delta y$")
    plt.colorbar(
        im,
        cax=make_axes_locatable(plt.gca()).append_axes(
            "right", size="5%", pad=0.05))
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_folder, "log_xy_marginal.pdf"),
        bbox_inches="tight", pad_inches=0)

    # plot wh marginal
    wh_marginal = target_dist.sum(axis=(0, 1))
    log_wh_marginal = np.log(wh_marginal)
    plt.figure()
    plt.gca().set_facecolor("k")
    im = plt.imshow(log_wh_marginal.T, interpolation="nearest",
                    vmax=-3, vmin=-18,
                    extent=(wmin, wmax, hmin, hmax))
    plt.title("Marginal log-probability for GT Transformations (w, h)")
    plt.xlabel("$\\Delta w$")
    plt.ylabel("$\\Delta h$")
    plt.colorbar(
        im,
        cax=make_axes_locatable(plt.gca()).append_axes(
            "right", size="5%", pad=0.05))
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_folder, "log_wh_marginal.pdf"),
        bbox_inches="tight", pad_inches=0)

    # compute the predicted distribution
#    data = np.load(os.path.join(args.output_folder, "distributions.npz"))
#    target_dist = data["target_dist"]
#    pred_dist = data["pred_dist"]
    pred_dist = np.zeros((nx, ny, nw, nh))
    grid = np.stack(np.meshgrid(np.linspace(xmin + 0.5 * b, xmax - 0.5 * b, nx),
                                np.linspace(ymin + 0.5 * b, ymax - 0.5 * b, ny),
                                np.linspace(wmin + 0.5 * b, wmax - 0.5 * b, nw),
                                np.linspace(hmin + 0.5 * b, hmax - 0.5 * b, nh),
                                indexing="ij"),
                    axis=-1)
    inv_grid_volume = 1. / ((xmax - xmin - b) * (ymax - ymin - b) *
                            (wmax - wmin - b) * (hmax - hmin - b))

    print "Grid size:", grid.shape, pred_dist.shape

    M = 1.345
    print x.shape, target.shape
    x_base, y_base = get_baseline_data()
    easy, med, hard = get_difficulty_ids(x_base, y_base)
    w1, w2, w3 = 1, 2, 3
    for i, p in enumerate(pred[:,:,-1,:]):
        print i + 1, "/", len(pred)
        mu, sigma = p.T

        z_score = np.abs(grid - mu) / sigma
        probs = np.exp(np.where(
                (z_score < M),
                -0.5 * np.square(z_score),
                M * (0.5 * M - z_score)).sum(axis=-1))

        if i in easy:
            pred_dist += w1 * (probs / probs.sum())
        elif i in med:
            pred_dist += w2 * (probs / probs.sum())
        elif i in hard:
            pred_dist += w3 * (probs / probs.sum())
        else:
            print('shouldnt occur')

        #pred_dist += w * (probs / probs.sum())
        
    
    #pred_dist *= (1. / len(pred))
    pred_dist *= (1. / (w1*len(easy) + w2*len(med) + w3*len(hard)))

    # plot xy marginal
    xy_marginal = pred_dist.sum(axis=(2, 3))
    log_xy_marginal = np.log(xy_marginal)
    plt.figure()
    plt.gca().set_facecolor("k")
    im = plt.imshow(log_xy_marginal.T, interpolation="nearest",
                    vmax=-3, vmin=-18,
                    extent=(xmin, xmax, ymax, ymin))
    plt.title("Marginal Log-probability for Predicted Transformations (x, y)")
    plt.xlabel("$\\Delta x$")
    plt.ylabel("$\\Delta y$")
    plt.colorbar(
        im,
        cax=make_axes_locatable(plt.gca()).append_axes(
            "right", size="5%", pad=0.05))
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_folder, "pred_log_xy_marginal.pdf"),
        bbox_inches="tight", pad_inches=0)

    # plot wh marginal
    wh_marginal = pred_dist.sum(axis=(0, 1))
    log_wh_marginal = np.log(wh_marginal)
    plt.figure()
    plt.gca().set_facecolor("k")
    im = plt.imshow(log_wh_marginal.T, interpolation="nearest",
                    vmax=-3, vmin=-18,
                    extent=(wmin, wmax, hmin, hmax))
    plt.title("Marginal log-probability for Predicted Transformations (w, h)")
    plt.xlabel("$\\Delta w$")
    plt.ylabel("$\\Delta h$")
    plt.colorbar(
        im,
        cax=make_axes_locatable(plt.gca()).append_axes(
            "right", size="5%", pad=0.05))
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_folder, "pred_log_wh_marginal.pdf"),
        bbox_inches="tight", pad_inches=0)

    print "Absolute error:", np.abs(target_dist - pred_dist)

    # save the distributions
    np.savez(os.path.join(args.output_folder, "distributions.npz"),
             target_dist=target_dist,
             pred_dist=pred_dist)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("results_file", type=str, help="")
    parser.add_argument("--output_folder", type=str, default="test_results",
        help="")

    parser.add_argument("--bin_size", type=float, default=0.1,
        help="")

    args = parser.parse_args()

    main(args)

