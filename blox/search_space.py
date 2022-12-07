import random

from .utils import pairwise, flatten, copy_structure


all_ops = ['conv', 'mbconv', 'bconv']#, 'zero']
ops_no_zero = all_ops[:-1]
default_nodes = 2
default_stages = 3


def get_search_space(ops=None, nodes=None, stages=None):
    ''' Return boundaries of the search space for the given list
        of available operations and number of nodes. 
    '''
    ops = ops if ops is not None else all_ops
    nodes = nodes if nodes is not None else default_nodes
    stages = stages if stages is not None else default_stages
    search_space = [[[len(ops)] + ([2,2,2] if nidx else []) for nidx in range(nodes)]]  * stages
    return search_space


def get_model_hash(arch_vec, ops=None, minimize=True):
    ''' Get hash of the architecture specified by arch_vec.
        Architecture hash can be used to determine if two
        configurations from the search space are in fact the
        same (graph isomorphism).
    '''
    from .graph_utils import get_model_graph, graph_hash, is_empty
    g, _ = get_model_graph(arch_vec, ops=ops, minimize=minimize)
    if is_empty(g):
        return None
    return graph_hash(g)


def get_all_architectures(ops=None, nodes=None, stages=None):
    ''' Yields all architecture configurations in the search space
    '''
    search_space = get_search_space(ops, nodes, stages)
    flat = flatten(search_space)
    cfg = [0 for _ in range(len(flat))]
    end = False
    while not end:
        yield copy_structure(cfg, search_space)
        for dim in range(len(flat)):
            cfg[dim] += 1
            if cfg[dim] != flat[dim]:
                break
            cfg[dim] = 0
            if dim+1 >= len(flat):
                end = True


def get_unique_architectures(ops=None, nodes=None, stages=None):
    ''' Yields unique architecture configurations in the search space
    '''
    ret = []
    memo = set()
    for arch in get_all_architectures(ops=ops, nodes=nodes, stages=stages):
        h = get_model_hash(arch, ops=ops)
        if h is not None and h not in memo:
            memo.add(h)
            ret.append(arch)

    return ret


def get_random_architectures(num, ops=None, nodes=None, stages=None, seed=None):
    ''' Get random architecture configurations from the search space
    '''
    if seed is not None:
        random.seed(seed)
    search_space = get_search_space(ops, nodes, stages)
    flat = flatten(search_space)
    models = []
    while len(models) < num:
        m = [random.randrange(opts) for opts in flat]
        m = copy_structure(m, search_space)
        models.append(m)
    return models


def get_archs_with_zero():
    models_with_zero = {}
    for m in get_all_architectures():
        if 5 in flatten(m):
            h = get_model_hash(m)
            models_with_zero[h] = m
    new_model_archs = [models_with_zero[k] for k in sorted(models_with_zero.keys())]
    return new_model_archs


def arch_vec_to_names(arch_vec, ops=None):
    ''' Translates identifiers of operations in ``arch_vec`` to their names.
        ``ops`` can be provided externally to avoid relying on the current definition
        of available ops. Otherwise canonical ``all_ops`` will be used.
    '''

    if ops is None:
        ops = all_ops

    # current approach is to have an arch vector contain sub-vectors for node in a cell,
    # each subvector has a form of:
    # [op_idx, branch_op_idx...]
    # where op_idx points to an operation from ``all_ops`` and ``branch_op_idx`` is
    # either 0 (no skip connection) or 1 (identity skip connection)
    # since skip connects are already quite self-explanatory we leave them as they are
    # and only change numbers of the main operations to their respective names

    ret = []
    for cell_cfg in arch_vec:
        new = []
        for node_cfg in cell_cfg:
            op, *rest = node_cfg
            new.append([ops[op], *rest])
        ret.append(new)

    return ret


if __name__ == '__main__':
    print(get_search_space())
