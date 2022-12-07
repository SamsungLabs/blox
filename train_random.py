import os
import copy
import pickle
import random
import argparse

import blox
import blox.training

import train


def get_points(flatss, number):
    ret = set()
    while len(ret) < number:
        pt = [random.randint(0, d-1) for d in flatss]
        pt = blox.utils.freeze(pt)
        if pt not in ret:
            ret.add(pt)

    return list(ret)


def main():
    ss = blox.search_space.get_search_space()
    flat_ss = blox.utils.flatten(ss)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--stem', type=int, default=32)
    parser.add_argument('--cells', '-c', type=int, default=1)
    parser.add_argument('num_points', type=int)
    args = blox.training.Args.from_stdargs(parser)

    meaningful_args = copy.deepcopy(args)
    del meaningful_args.gpu
    del meaningful_args.load
    del meaningful_args.save

    results = {}
    results_file = blox.utils.get_next_unused_filename(f'{args.wdir}/results_random.pickle')

    os.makedirs(args.wdir, exist_ok=True)

    pts = get_points(flat_ss, args.num_points)
    for pt in pts:
        arch_vec = blox.utils.copy_structure(pt, ss)
        h = blox.search_space.get_model_hash(arch_vec)
        hist = train.train_single(arch_vec, args.seed, args.stem, args.cells, args)
        if hist is None or not hist:
            continue
        results[h] = hist

        with open(results_file, 'ab') as f:
            pickle.dump(meaningful_args, f)
            pickle.dump(results, f)


if __name__ == '__main__':
    main()
