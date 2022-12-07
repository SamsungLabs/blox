import time
from pathlib import Path

import blox.io as bio


def main():
    folder = Path(__file__).parents[1].joinpath('release')

    ret = {}

    for ext in ['.blox', '.csv', '.pickle']:
        ret[ext] = {}
        for part in ['blox-info', 'blox-env-0', 'blox-base-0', 'blox-ext-0']:
            f = (folder / part).with_suffix(ext)
            tick = time.time()
            _, data = bio.read_dataset_file(f, fast=True)
            tock = time.time()
            print(f'Loaded file {f} in {tock-tick} seconds...')
            ret[ext][part] = data

        print()

    ref = ret['.pickle']
    custom = ret['.blox']
    csv = ret['.csv']

    print('CSV:', { k: ref[k] == csv[k] for k in ref.keys() })
    print('Custom:', { k: ref[k] == custom[k] for k in ref.keys() })


if __name__ == '__main__':
    main()
