import os
import copy
import pickle
import datetime
from pathlib import Path

import csv
import yaml
import blox.search_space as ss
import blox.io as bio


def make_header(dtype, version=1, extra_columns=None, **extra_fields):
    extra_columns = extra_columns or []
    base = {
        'version': version,
        'search_space': {
            'shape': ss.get_search_space(ss.all_ops, ss.default_nodes, ss.default_stages),
            'nodes': ss.default_nodes,
            'stages': ss.default_stages,
            'ops': ss.all_ops
        },
        'dataset_type': dtype,
        'columns': ['model_hash'] + extra_columns,
        **extra_fields
    }

    if dtype == 'training':
        if extra_fields['part'] == 'base':
            base['columns'].extend(['val_top1', 'test_top1', 'train_time_s'])
        elif extra_fields['part'] == 'ext':
            base['columns'].extend(['train_top1', 'train_loss', 'train_top5', 'val_loss', 'val_top5', 'test_loss', 'test_top5'])
        else:
            raise ValueError(f'Unknown part: {extra_fields["part"]}')

    elif dtype == 'static':
        base['columns'].extend([ 'params', 'flops', 'arch_vec'])
    elif dtype == 'env':
        pass
    elif dtype == 'benchmarking':
        base['columns'].extend(['latency'])
    else:
        raise ValueError(f'Unknown dataset type {dtype}')

    return base


def round_floats(hdr, data):
    def _r_acc_single(v):
        return round(v, 3)
    def _r_acc_list(v):
        return [round(vv, 3) for vv in v]
    def _r_loss_single(v):
        return round(v, 6)
    def _r_loss_list(v):
        return [round(vv, 6) for vv in v]

    data = copy.deepcopy(data)

    for row in range(len(data)):
        for idx, cname in enumerate(hdr['columns']):
            if 'top' in cname:
                if 'test' in cname:
                    fn = _r_acc_single
                else:
                    fn = _r_acc_list
            elif 'loss' in cname:
                if 'test' in cname:
                    fn = _r_loss_single
                else:
                    fn = _r_loss_list
            else:
                fn = None

            if fn is not None:
                data[row][idx] = fn(data[row][idx])

    return data

def _save_pickle(hdr, data, fname):
    hdr = hdr.copy()
    hdr['format'] = 'pickle'
    data = round_floats(hdr, data)
    with open(fname, 'wb') as f:
        f.write(yaml.dump(hdr, explicit_start=True, sort_keys=False).encode('utf-8'))
        f.write('---\n'.encode('utf-8'))
        pickle.dump(data, f)


def _save_csv(hdr, data, fname):
    hdr = hdr.copy()
    hdr['format'] = 'csv'
    data = round_floats(hdr, data)
    with open(fname, 'w', encoding='utf-8', newline='') as f:
        f.write(yaml.dump(hdr, explicit_start=True, sort_keys=False))
        f.write('---\n')
        csv.writer(f).writerows(data)


def _save_custom(hdr, data, fname):
    hdr = hdr.copy()
    hdr['format'] = 'custom'

    data = bio.encode(hdr, data)

    with open(fname, 'wb') as f:
        f.write(yaml.dump(hdr, explicit_start=True, sort_keys=False).encode('utf-8'))
        f.write('---\n'.encode('utf-8'))
        f.write(data)


def save(hdr, data, fname):
    fname = Path(fname)
    os.makedirs(fname.parent, exist_ok=True)

    _save_pickle(hdr, data, fname.with_suffix('.pickle'))
    _save_csv(hdr, data, fname.with_suffix('.csv'))
    _save_custom(hdr, data, fname.with_suffix('.blox'))


def parse_timedelta(tstr):
    if '.' in tstr:
        full, fraction = tstr.split('.')
        fraction = f'0.{fraction}'
    else:
        full, fraction = tstr, '0.0'

    parts = reversed([int(v) for v in full.split(':')])
    bases = ['seconds', 'minutes', 'hours']

    td = datetime.timedelta(**dict(zip(bases, parts)))

    seconds = td.total_seconds()
    fraction = float(fraction)

    ret = int(round(seconds + fraction))
    assert ret > 100
    return ret

def extract_train_info(all_info, ext):
    values = []
    def append_group(name):
        assert name in all_info
        epochs = sorted(list(all_info[name]))
        assert epochs

        if (ext and name == 'train') or (not ext and name != 'train'):
            top1 = [all_info[name][e]['top1'][0] for e in epochs]
            if len(epochs) == 1:
                top1 = top1[0]

            values.extend([top1])

        if ext:
            loss = [all_info[name][e]['loss'][0] for e in epochs]
            top5 = [all_info[name][e]['top5'][0] for e in epochs]
            if len(epochs) == 1:
                loss = loss[0]
                top5 = top5[0]

            values.extend([loss, top5])

    append_group('train')
    append_group('valid')
    append_group('test')
    if not ext:
        values.append(parse_timedelta(all_info['time']))

    return values


def main():
    seed = 0 # TODO: add argument?

    with open(os.path.expanduser('exp/combined.pickle'), 'rb') as f:
        raw = pickle.load(f)

    total = len(raw)
    missing = { key for key, value in raw.items() if not value }
    raw = { key: value for key, value in raw.items() if value }
    meaningful = len(raw)

    with open('data/unique_archs.pickle', 'rb') as f:
        unique_a = pickle.load(f)
    for a in unique_a:
        h = ss.get_model_hash(a)
        if h not in raw:
            missing.add(h)

    print(f'Dataset has {meaningful} meaningful entries out of {total} recorded')

    with open('data/hash_to_arch.pickle', 'rb') as f:
        h2a = pickle.load(f)

    with open('data/unique_archs.pickle', 'rb') as f:
        archs = pickle.load(f)

    missing = [h2a[k] for k in missing]
    positions = [archs.index(k) for k in missing]
    positions = sorted(positions)
    if missing:
        beg = positions[0]
        last = positions[0]
        for p in positions[1:]:
            if p != last+1:
                print('Missing:', beg, last, ss.get_model_hash(archs[beg]))
                beg = p
            last = p

        print('Missing:', beg, last, ss.get_model_hash(archs[beg]))

        with open('data/missing_archs.pickle', 'wb') as f:
            pickle.dump(missing, f)

    valid_archs = [h2a[k] for k in raw.keys()]
    with open('data/valid_archs.pickle', 'wb') as f:
        pickle.dump(valid_archs, f)

    def collapse_gpu(val, key):
        if key != 'gpu':
            return val
        assert len(val) == 1
        return val[0]

    env_keys = sorted(list(next(iter(raw.values()))['env'].keys()))
    env_info = [[key] + [collapse_gpu(values['env'][env_key], env_key) for env_key in env_keys] for key, values in raw.items()]
    env_header = make_header('env', extra_columns=env_keys, seed=seed)
    save(env_header, env_info, f'release/blox-env-{seed}')
    del env_keys, env_header, env_info

    train_info = [[key] + extract_train_info(values, ext=False) for key, values in raw.items()]
    train_header = make_header('training', seed=seed, epochs=len(train_info[0][1]), dataset='cifar100', part='base')
    save(train_header, train_info, f'release/blox-base-{seed}')
    del train_header, train_info

    train_info = [[key] + extract_train_info(values, ext=True) for key, values in raw.items()]
    train_header = make_header('training', seed=seed, epochs=len(train_info[0][1]), dataset='cifar100', part='ext')
    save(train_header, train_info, f'release/blox-ext-{seed}')
    del train_header, train_info

    static_info = [[key] + [values['params'], int(values['flops']), h2a[key]] for key, values in raw.items()]
    static_header = make_header('static')
    save(static_header, static_info, 'release/blox-info')
    del static_header, static_info


if __name__ == '__main__':
    main()
