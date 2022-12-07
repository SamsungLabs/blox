import sys
import pathlib
import functools
import collections
import operator as ops
import re
import torch

class LazyModule():
    def __init__(self, module):
        self.module = module

    def __repr__(self):
        return repr(self.module)

    def __getattr__(self, name):
        return getattr(self.module, name)


def add_module_properties(module_name, properties):
    module = sys.modules[module_name]
    replace = False
    if isinstance(module, LazyModule):
        lazy_type = type(module)
    else:
        lazy_type = type('LazyModule({})'.format(module_name), (LazyModule,), {})
        replace = True

    for name, prop in properties.items():
        setattr(lazy_type, name, prop)

    if replace:
        sys.modules[module_name] = lazy_type(module)


class staticproperty(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        if fget is not None and not isinstance(fget, staticmethod):
            raise ValueError('fget should be a staticmethod')
        if fset is not None and not isinstance(fset, staticmethod):
            raise ValueError('fset should be a staticmethod')
        if fdel is not None and not isinstance(fdel, staticmethod):
            raise ValueError('fdel should be a staticmethod')
        super().__init__(fget, fset, fdel, doc)

    def __get__(self, inst, cls=None):
        if inst is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget.__get__(inst, cls)() # pylint: disable=no-member

    def __set__(self, inst, val):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        return self.fset.__get__(inst)(val) # pylint: disable=no-member

    def __delete__(self, inst):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        return self.fdel.__get__(inst)() # pylint: disable=no-member


# utils to work with nested collections
def recursive_iter(seq):
    ''' Iterate over elements in seq recursively (returns only non-sequences)
    '''
    if isinstance(seq, collections.abc.Sequence) and not isinstance(seq, (str, bytes)):
        for e in seq:
            for v in recursive_iter(e):
                yield v
    else:
        yield seq


def flatten(seq):
    ''' Flatten all nested sequences, returned type is type of ``seq``
    '''
    return list(recursive_iter(seq))


def copy_structure(data, shape):
    ''' Put data from ``data`` into nested containers like in ``shape``.
        This can be seen as "unflatten" operation, i.e.:
            seq == copy_structure(flatten(seq), seq)
    '''
    d_it = recursive_iter(data)

    def copy_level(s):
        if isinstance(s, collections.abc.Sequence):
            return type(s)(copy_level(ss) for ss in s)
        else:
            return next(d_it)
    return copy_level(shape)


def count(seq):
    ''' Count elements in ``seq`` in a streaming manner.
    '''
    ret = 0
    for _ in seq:
        ret += 1
    return ret


def get_first_n(seq, n):
    ''' Get first ``n`` elements of ``seq`` in a streaming manner.
    '''
    c = 0
    i = iter(seq)
    while c < n:
        yield next(i)
        c += 1


def pairwise(seq):
    i = iter(seq)
    while True:
        try:
            c = next(i)
        except StopIteration:
            break
        c2 = next(i)
        yield c, c2


def make_nice_number(num):
    n = str(num)
    parts = (len(n)-1)//3 + 1
    if parts == 1:
        return n
    offset = len(n)%3 or 3
    breaks = [0] + [offset + i*3 for i in range(parts)] + [len(n)]
    return ','.join(n[breaks[i]:breaks[i+1]] for i in range(parts))


def get_next_unused_filename(filename):
    p = pathlib.Path(filename)

    if '.' in p.name:
        basename,_ = p.name.split('.', maxsplit=1)
    else:
        basename = p.name

    curr = 1
    while p.exists():
        p = p.with_name(f'{basename}_{curr}' + ''.join(p.suffixes))
        curr += 1

    return p


def freeze(seq):
    if isinstance(seq, collections.abc.Sequence) and not isinstance(seq, (str, bytes)):
        return tuple(freeze(sub) for sub in seq)
    return seq


def get_total_point(ss):
    return functools.reduce(ops.mul, flatten(ss))


def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True, verbose=True, uptostage=None, 
                            custom_stages=None, blocks_weight_path=None):
    if weights_path is not None:
      state_dict = torch.load(weights_path)
      state_dict = {re.sub("^module.", '', k): v for k, v in state_dict['model'].items()}

    if uptostage is not None or custom_stages is not None:
        model_dict = model.state_dict()
        if weights_path is not None:
          if not load_fc:
              state_dict.pop('classifier.weight')
              state_dict.pop('classifier.bias')
          # filter out unnecessary keys
          state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
          # overwrite entries in the existing state dict
          assert not state_dict=={}, 'state_dict is empty'
          ret = model.load_state_dict(state_dict, strict=False)
          assert not ret.unexpected_keys, 'Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
        # load block weights
        if blocks_weight_path is not None:
            for stage, path in zip(custom_stages, blocks_weight_path):
                try:
                  block_dict = torch.load(path)
                  block_dict = {re.sub("^block", f'_blocks.{stage}', k): v for k, v in block_dict.items() if re.sub("^block", f'_blocks.{stage}', k) in model_dict}
                  assert not block_dict=={}, 'block_dict of stage {} is empty'.format(stage)
                  ret = model.load_state_dict(block_dict, strict=False)
                  assert not ret.unexpected_keys, 'Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
                except Exception as e:
                  print(e)
    elif load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        #assert not ret.missing_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    else:
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        #assert set(ret.missing_keys) == set(
        #    ['classifier.weight', 'classifier.bias']), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    #assert not ret.unexpected_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)

    if verbose:
        print('Loaded pretrained weights for {}'.format(model_name))