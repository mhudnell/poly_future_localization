import numpy as np
from baselines import get_baseline_data, get_difficulty_ids, stats_per_difficulty

OFFSET_T = False

def main(args):
    data = np.load(args.results_file)
    x, target, pred = data["x"], data["target"], data["pred"]

    x_base, y_base = get_baseline_data()
    easy_ids, med_ids, hard_ids = get_difficulty_ids(x_base, y_base)

    ious_e, ious_m, ious_h = stats_per_difficulty(x, target, pred, easy_ids, med_ids, hard_ids,
                                offset_t=OFFSET_T)

    print('ious per difficulty:', np.mean(ious_e[:, 9]), np.mean(ious_m[:, 9]), np.mean(ious_h[:, 9]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("results_file", type=str, help="")

    args = parser.parse_args()
    main(args)

