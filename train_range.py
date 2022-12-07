import os
import copy
import pickle
import shutil
import argparse

import blox
import blox.training

import train


def get_points(start, end):
    if os.path.exists('data/unique_archs.pickle'):
        with open('data/unique_archs.pickle', 'rb') as f:
            models = pickle.load(f)
    else:
        models = blox.search_space.get_unique_architectures()
        if not os.path.isdir('data'):
            os.mkdir('data')
        with open('data/unique_archs.pickle', 'wb') as f:
            pickle.dump(models, f)

    return models[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--stem', type=int, default=32)
    parser.add_argument('--cells', '-c', type=int, default=1)
    parser.add_argument('range', type=int, nargs=2)
    args = blox.training.Args.from_stdargs(parser)

    meaningful_args = copy.deepcopy(args)
    del meaningful_args.gpus
    del meaningful_args.load
    del meaningful_args.save

    print(f'Will train models from {args.range[0]} to {args.range[1]}')

    results = {}
    results_file = f'{args.wdir}/results_{args.range[0]}_{args.range[1]}.pickle'
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            loadedargs = pickle.load(f)
            if loadedargs == meaningful_args:
                print('Resuming training...')
                results = pickle.load(f)
            else:
                print(f'Args mismatch! Please delete or move the file {results_file!r} if you wish to overwrite')
                return
    else:
        os.makedirs(args.wdir, exist_ok=True)

    pts = get_points(args.range[0], args.range[1])
    if not pts:
        print('No architectures to train!')
        return

    for arch_vec in pts:
        h = blox.search_space.get_model_hash(arch_vec)
        if h in results:
            print(f'Skipping architecture {h} -- already in the results')
            continue

        if os.path.exists(f'{args.wdir}/{h}'):
            shutil.rmtree(f'{args.wdir}/{h}')

        hist = train.train_single(arch_vec=arch_vec, seed=args.seed, stem=args.stem, cells=args.cells, args=args)
        results[h] = hist

        with open(results_file, 'wb') as f:
            pickle.dump(meaningful_args, f)
            pickle.dump(results, f)


if __name__ == '__main__':
    main()
