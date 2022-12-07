import re
import pickle
import argparse

import tabulate

import blox


_results_re = re.compile(r'.*results[a-zA-Z_]*_([0-9]+)_([0-9]+).pickle')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='+', type=str)
    parser.add_argument('--output', '-o', type=str, required=True)
    args = parser.parse_args()

    combined = {}
    partials = []
    refargs = None
    models_to_files = {}

    for f in args.file:
        with open(f, 'rb') as ff:
            try:
                loadedargs = pickle.load(ff)
                results = pickle.load(ff)
            except EOFError:
                print(f'File: {f!r} is corrupted, experienced unexpected EOF when unpickling!')
                continue

            del loadedargs.range
            if refargs is None:
                refargs = loadedargs
            else:
                pass
                # assert loadedargs == refargs, f'{loadedargs}\n{refargs}'

            if not results:
                print(f'File {f!r} does not contain any results, skipping')
                continue

            def fix_value(v):
                if v and v['env']['codebase_commit'] == 'f6f188191db6b4458a4fc5bdb6d7bd443e82dff2 (dirty)':
                    v['env']['codebase_commit'] = 'f6f188191db6b4458a4fc5bdb6d7bd443e82dff2'

            it = iter(results.values())
            first = None
            while first is None:
                try:
                    first = next(it)
                except StopIteration:
                    break
            if first is None:
                print(f'File: {f!r} is not empty but does not contain any valid results, skipping')
                continue
            curr = first
            while True:
                fix_value(curr)
                try:
                    curr = next(it)
                except StopIteration:
                    break

            if first['env']['codebase_commit'] != 'f6f188191db6b4458a4fc5bdb6d7bd443e82dff2':
                print(f'Ignoring file: {f!r} since it uses a wrong codebase version! Expected f6f188191db6b4458a4fc5bdb6d7bd443e82dff2 but got {first["env"]["codebase_commit"]}')
                continue

            valid_results = { key: value for key, value in results.items() if value }
            l1, l2 = len(combined), len(valid_results)
            old_combined = combined
            combined = {
                **combined,
                **valid_results
            }

            m = _results_re.fullmatch(f)
            beg = int(m.group(1))
            end = int(m.group(2))
            count = end-beg
            partials.append((f, beg, end, len(results), count, f'{len(results)/count:.2%}'))
            if len(combined) != l1+l2:
                intersection = set(old_combined.keys()).intersection(set(results.keys()))
                assert intersection, 'A duplicated key must exist if len(a+b) != len(a)+len(b)'
                print(f'Found {len(intersection)} duplicated keys!')
                for k in intersection:
                    print(f'Model {k} is present in file {models_to_files[k]!r} and {f!r}') # --> Values:\n\n{pprint.pformat(old_combined[k])}\n\n{pprint.pformat(results[k])}\n\n')
                #sys.exit(1)
            #else:
            for k in results.keys():
                models_to_files[k] = f


    partials = sorted(partials, key=lambda p: p[1])
    # partials = sorted(partials, key=lambda p: p[3]/p[4])
    print(tabulate.tabulate([p for p in partials if p[-1] != '100.00%']))
    print(len(combined))

    # compute gaps
    last_end = 0
    for p in partials:
        b, e = p[1:3]
        if b > last_end:
            print(f'Hole: {last_end} - {b}')
        last_end = max(last_end, e)

    print('Last model:', last_end)

    with open(args.output, 'wb') as f:
        pickle.dump(combined, f)

    best = [(arch, info['test'][200]['top1'][0] if info else 0) for arch, info in combined.items() if info]
    best = sorted(best, key=lambda p: p[1], reverse=True)

    with open('data/hash_to_arch.pickle', 'rb') as f:
        h2a = pickle.load(f)

    print('Worst')
    for h, acc in best[:-21:-1]:
        print(h2a[h], h, acc)

    print('Best')
    for h, acc in best[:20]:
        print(h2a[h], h, acc)

    print('Median')
    h, acc = best[len(best)//2]
    print(h2a[h], h, acc)

    for i, (h, _) in enumerate(best[:20]):
        blox.graph_utils.show_model(h2a[h], aid=f'best{i}', show=False, inc_full=False, out_dir='graphs/best_worst')

    for i, (h, _) in enumerate(best[:-21:-1]):
        blox.graph_utils.show_model(h2a[h], aid=f'worst{i}', show=False, inc_full=False, out_dir='graphs/best_worst')


if __name__ == '__main__':
    main()
