import os
import copy
import hashlib
import tempfile
import subprocess

import tqdm
import numpy as np
import networkx as nx

from .utils import flatten, freeze

_use_np = True


def get_model_graph_np(arch_vector, ops=None, minimize=True, keep_dims=False):
    if ops is None:
        from . import search_space as ss
        ops = ss.all_ops

    num_nodes = sum(len(options) for options in arch_vector)
    mat = np.zeros((num_nodes+2+len(arch_vector), num_nodes+2+len(arch_vector)))
    labels = [('input', None)]

    nidx = 1
    outputs = [0]
    all_outputs = {0: [0] }

    for sidx, stage in enumerate(arch_vector):
        stage_outputs = outputs.copy()
        for in_cell_nidx, node in enumerate(stage):
            op, *cons = node
            if in_cell_nidx > 0:
                prev, inp, sc = cons
            else:
                sc = 0 if not cons else cons[0]
                inp = 0
                prev = 1

            labels.append((ops[op], sidx))
            inputs = []
            if prev:
                inputs.extend(outputs)
            if inp:
                inputs.extend(stage_outputs)

            for i in inputs:
                mat[i, nidx] = 1

            outputs = [nidx]
            if sc:
                outputs.extend(all_outputs[nidx-1])

            all_outputs[nidx] = outputs

            nidx += 1

        labels.append(('reduction', sidx))
        for o in outputs:
            mat[o, nidx] = 1
        outputs = [nidx]
        all_outputs[nidx] = outputs
        nidx += 1

    labels.append(('output', None))
    for o in outputs:
        mat[o, nidx] = 1

    orig = None
    if minimize:
        orig = copy.copy(mat), copy.copy(labels)
        for n in range(len(mat)):
            if labels[n][0] == 'zero':
                for n2 in range(len(mat)):
                    if mat[n,n2]:
                        mat[n,n2] = 0
                    if mat[n2,n]:
                        mat[n2,n] = 0
        def bfs(src, mat, backward):
            visited = np.zeros(len(mat))
            q = [src]
            visited[src] = 1
            while q:
                n = q.pop()
                for n2 in range(len(mat)):
                    if visited[n2]:
                        continue
                    if (backward and mat[n2,n]) or (not backward and mat[n,n2]):
                        q.append(n2)
                        visited[n2] = 1
            return visited
        vfw = bfs(0, mat, False)
        vbw = bfs(len(mat)-1, mat, True)
        v = vfw + vbw
        dangling = (v < 2).nonzero()[0]
        if dangling.size:
            if keep_dims:
                mat[dangling, :] = 0
                mat[:, dangling] = 0
                for i in dangling:
                    labels[i] = None
            else:
                mat = np.delete(mat, dangling, axis=0)
                mat = np.delete(mat, dangling, axis=1)
                for i in sorted(dangling, reverse=True):
                    del labels[i]

    return (mat, labels), orig

def get_model_graph_nx(arch_vector, ops=None, minimize=True, keep_dims=False):
    ''' Get :class:`netwworkx.DiGraph` object from an arch vector.
        If ``minimize`` is ``True``, the graph will be minimized by removing
        "zero" operations and consequently any dangling nodes.
    '''
    if ops is None:
        from . import search_space as ss
        ops = ss.all_ops
    num_nodes = sum(len(options) for options in arch_vector)
    g = nx.DiGraph()
    g.add_node(0, label=('input', None))

    nidx = 1
    for sidx, stage in enumerate(arch_vector):
        node_input = nidx-1
        for in_cell_nidx, node in enumerate(stage):
            op, *cons = node
            if in_cell_nidx > 1:
                prev, inp, sc = cons
            else:
                prev = cons
                inp = 0
                sc = 0

            g.add_node(nidx, label=(ops[op], sidx))

            if prev:
                g.add_edge(nidx-1, nidx)
            if inp:
                g.add_edge(node_input, nidx)
            if sc:
                g.add_edge(nidx-1, nidx+1)

            nidx += 1

    g.add_node(nidx, label=('output', None))
    g.add_edge(nidx-1, nidx)

    orig = None
    if minimize:
        orig = copy.deepcopy(g)
        for n in dict(g.nodes):
            if g.nodes[n]['label'][0] == 'zero':
                g.remove_node(n)
        for _i in range(2):
            if 0 in g.nodes:
                from_source = nx.descendants(g, 0)
            else:
                from_source = []
            for n in dict(g.nodes):
                keep = True
                desc = nx.descendants(g, n)
                if n != num_nodes+1:
                    if num_nodes+1 not in desc:
                        keep = False
                if n > 0:
                    if n not in from_source:
                        keep = False
                if not keep:
                    if not _i:
                        if keep_dims:
                            edges = list(g.in_edges(n)) + list(g.out_edges(n))
                            g.remove_edges_from(edges)
                            g.nodes[n]['label'] = None
                        else:
                            g.remove_node(n)
                    else:
                        print(_i, n, desc)
                        show_graph(g)
                        show_graph(orig)
                        assert False
    return g, orig

def get_model_graph(arch_vector, ops=None, minimize=True, keep_dims=False):
    if _use_np:
        return get_model_graph_np(arch_vector, ops, minimize, keep_dims)
    else:
        return get_model_graph_nx(arch_vector, ops, minimize, keep_dims)


def is_empty(g):
    if _use_np:
        return not bool(g[1])
    else:
        return not bool(g)


def optimise_config(cell_config, ops=None):
    if ops is None:
        from . import search_space as ss
        ops = ss.all_ops

    (mat, lab), (_, fulllab) = get_model_graph_np([cell_config], ops=ops, minimize=True, keep_dims=False)
    if not lab:
        return []

    assert len(fulllab) == len(cell_config) + 3
    assert lab[0][0] == 'input', lab
    assert lab[-1][0] == 'output', lab
    assert lab[-2][0] == 'reduction', lab
    ret = []
    for nidx in range(1, len(lab)-2):
        op = ops.index(lab[nidx][0])
        if nidx > 1:
            prev = int(mat[nidx-1, nidx])
            input = int(mat[0, nidx])
            sc = int(mat[nidx-1, nidx+1])
        else:
            prev = 1
            input = 0
            sc = 0
        ret.append([op, prev, input, sc])

    return ret


def graph_hash_np(g):
    from . import search_space as ss
    m, l = g

    def hash_module(matrix, labelling):
        """Computes a graph-invariance MD5 hash of the matrix and label pair.
        Args:
            matrix: np.ndarray square upper-triangular adjacency matrix.
            labelling: list of int labels of length equal to both dimensions of
                matrix.
        Returns:
            MD5 hash of the matrix and labelling.
        """
        vertices = np.shape(matrix)[0]
        in_edges = np.sum(matrix, axis=0).tolist()
        out_edges = np.sum(matrix, axis=1).tolist()
        assert len(in_edges) == len(out_edges) == len(labelling), f'{labelling} {matrix}'
        hashes = list(zip(out_edges, in_edges, labelling))
        hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
        # Computing this up to the diameter is probably sufficient but since the
        # operation is fast, it is okay to repeat more times.
        for _ in range(vertices):
            new_hashes = []
            for v in range(vertices):
                in_neighbours = [hashes[w] for w in range(vertices) if matrix[w, v]]
                out_neighbours = [hashes[w] for w in range(vertices) if matrix[v, w]]
                new_hashes.append(hashlib.md5(
                        (''.join(sorted(in_neighbours)) + '|' +
                        ''.join(sorted(out_neighbours)) + '|' +
                        hashes[v]).encode('utf-8')).hexdigest())
            hashes = new_hashes
        fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()
        return fingerprint

    labels = []
    if l:
        labels.append(-1) # input
        for lab in l[1:-1]:
            op, sidx = lab
            if op == 'reduction':
                op = len(ss.all_ops)
            else:
                op = ss.all_ops.index(op)

            op = op + ((len(ss.all_ops)+1) * sidx)
            labels.append(op)

        labels.append(-2) # output

    return hash_module(m, labels)

def graph_hash_nx(g):
    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(g, node_attr='label')

def graph_hash(g):
    if _use_np:
        return graph_hash_np(g)
    else:
        return graph_hash_nx(g)

def get_adjacency_and_features(matrix, labels):
    if isinstance(matrix, np.ndarray):
        matrix = matrix.astype(int).tolist()
    # Add global node
    for row in matrix:
        row.insert(0, 0)
    global_row = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    matrix.insert(0, global_row)
    # Add diag matrix
    for idx, row in enumerate(matrix):
        row[idx] = 1

    # Create features matrix from labels
    features = [[0 for _ in range(15)] for _ in range(12)]
    features[0][0] = 1 # global
    #features[1][0] = 1 # input
    #features[-1][13] = 1 # output
    for idx, op in enumerate(labels):
        if op is not None:
            op = f'{op[0]}_{op[1]}'
            features[idx+1][_op_to_int[op]] = 1
    return matrix, features

_op_to_int = {
        'input_None': 1,
        'mbconv_0': 2,
        'bconv_0': 3,
        'conv_0': 4,
        'reduction_0': 5,
        'mbconv_1': 6,
        'bconv_1': 7,
        'conv_1': 8,
        'reduction_1': 9,
        'mbconv_2': 10,
        'bconv_2': 11,
        'conv_2': 12,
        'reduction_2': 13,
        'output_None': 14
}

_op_to_node_color = {
    'conv': 'tomato',
    'mbconv': 'cadetblue1',
    'bconv': 'olivedrab2',
    'reduction': 'lightgray'
}

_op_to_label = {
    'conv': 'Conv Block',
    'mbconv': 'MBConv Block',
    'bconv': 'BConv Block',
    'input': 'Input',
    'output': 'Output',
    'zero': 'Zero',
    'reduction': 'Reduction'
}

def _make_nice(agraph, dist_from_source, degrees):
    positions = {}
    agraph.node_attr['shape'] = 'rectangle'
    agraph.node_attr['style'] = 'rounded'
    agraph.edge_attr['arrowsize'] = 0.5
    agraph.graph_attr['splines'] = 'true'
    agraph.graph_attr['esep'] = 0.17
    #agraph.graph_attr['overlap'] = 'false'

    nodes_and_distances = list(dist_from_source.items())
    nodes_and_distances = sorted(nodes_and_distances, key=lambda p: p[1])
    nodes_per_dist = {}
    for _, d in nodes_and_distances:
        nodes_per_dist.setdefault(d, 0)
        nodes_per_dist[d] += 1

    next_x = {}
    xs = {}
    ys = {}
    node_x_sep = 2.0
    for d, num_nodes in nodes_per_dist.items():
        total_width = node_x_sep*(num_nodes-1)
        xmax = total_width/2
        xmin = -xmax
        xs[d] = [xmin+node_x_sep*node_idx for node_idx in range(num_nodes)]
        next_x[d] = 0

    def get_next_x(distance):
        xidx = next_x[distance]
        next_x[distance] += 1
        return xs[distance][xidx]

    y_offset = 0

    def get_next_y(nidx, distance):
        ret = ys.setdefault(nidx, -0.47*(distance+y_offset))
        ys[nidx] = ret - 0.47
        return ret

    last_dist = 0
    next_offset = 0

    for nidx, distance in nodes_and_distances:
        if distance != last_dist:
            y_offset = next_offset

        node = agraph.get_node(nidx)
        op, stage = eval(node.attr['label'])

        node.attr['label'] = _op_to_label.get(op, op) + f' (S: {stage})'
        node.attr['width'] = 1.2
        node.attr['height'] = 0.3
        if op in _op_to_node_color:
            node.attr['fillcolor'] = _op_to_node_color[op]
            node.attr['style'] = 'filled,rounded'

        this_node_x = get_next_x(distance)

        if degrees[nidx] > 1:
            inp_node = f'input_{nidx}'
            agraph.add_node(inp_node, label='+', shape='circle', width=0.3, height=0.3, fixedsize=True, fontsize=16, pos=f'{this_node_x},{get_next_y(nidx,distance)}!')
            next_offset = max(next_offset, y_offset+1)
            in_nodes = []
            for n in agraph.in_neighbors(nidx):
                in_nodes.append(n)
                if not n.isnumeric() or (dist_from_source[int(n)] + 1 != dist_from_source[nidx]):
                    agraph.add_edge(n, inp_node, group='branches', style='dashed')
                else:
                    agraph.add_edge(n, inp_node, group='main')

                agraph.delete_edge(n, nidx)

            agraph.add_edge(inp_node, nidx, group='main')

            for n in agraph.nodes():
                if n != inp_node:
                    has_all_inputs = all(inp in agraph.in_neighbors(n) for inp in in_nodes)
                    if has_all_inputs:
                        for inp in in_nodes:
                            agraph.delete_edge(inp, n)
                        agraph.add_edge(inp_node, n, group='branches', style='dashed')

        node.attr['pos'] = f'{this_node_x},{get_next_y(nidx,distance)}!'


def show_graph(g, aid=None, show=True, out_dir=None):
    ''' Renders graph ``g`` using graphiviz.
        ``aid`` is an optional architecture id, if provided,
        the rendered graph will be stored under "{out_dir}/nb_graph.{aid}.png".
        (If ``out_dir`` is ``None``, it will default to ``graphs``).
        Otherwise, it will be saved in a temporary file.
        If ``show`` is ``True``, the rendered file will be opened with "xdg-open".
    '''
    if _use_np:
        a, l = g
        g = nx.from_numpy_array(a, create_using=nx.DiGraph)
        for idx, label in enumerate(l):
            g.nodes[idx]['label'] = label

    dist = {}
    for n in nx.topological_sort(g):
        m_dist = 0
        for p in g.predecessors(n):
            assert p in dist
            if m_dist < dist[p] + 1:
                m_dist = dist[p] + 1

        dist[n] = m_dist

    degrees = { n: g.in_degree[n] for n in g.nodes() }

    a = nx.nx_agraph.to_agraph(g)
    _make_nice(a, dist, degrees)
    a.layout('dot', '-Kfdp')
    if aid is None:
        fname = tempfile.mktemp('.png', 'nb_graph.')
    else:
        dname = out_dir if out_dir is not None else "graphs"
        os.makedirs(dname, exist_ok=True)
        fname = f'{dname}/nb_graph.{aid}.png'
    a.draw(fname)
    if show:
        subprocess.run(['xdg-open', fname], check=True)


def show_model(arch_vec, aid=None, show=True, inc_full=True, out_dir=None):
    ''' Renders graphs constructed from arch vector (both minimal and full).
        Full graph is only rendered if different from minimal.
        ``aid`` is an architecture id which will be used when saving rendered graphs,
        if not provided it will be derived from ``arch_vec``.
    '''
    g, full = get_model_graph(arch_vec)
    if aid is None:
        aid = '_'.join(map(str, flatten(arch_vec)))
    show_graph(g, aid=aid, show=show, out_dir=out_dir)
    if full is not None and inc_full:
        if graph_hash(g) != graph_hash(full):
            #assert 3 in flatten(arch_vec)
            show_graph(full, aid=f'{aid}_full', show=show, out_dir=out_dir)
        else:
            assert 3 not in flatten(arch_vec)


def compare_nx_and_np():
    import functools
    from .search_space import get_all_architectures, get_search_space, all_ops, default_nodes, default_stages
    global _use_np
    all_count = functools.reduce(lambda a,b: a*b, flatten(get_search_space(all_ops, default_nodes, default_stages)))
    _use_np = False
    all_hashes = set()
    without_zero = set()
    unique_graphs = []
    conflicts = {}
    for m in tqdm.tqdm(get_all_architectures(all_ops, default_nodes, default_stages), total=all_count):
        has_zero = 3 in flatten(m)
        g, _ = get_model_graph(m)
        h = graph_hash(g)
        if h not in all_hashes:
            unique_graphs.append(m)
        else:
            conflicts[h] = m
        all_hashes.add(h)
        if not has_zero:
            without_zero.add(h)
    _use_np = True
    np_hashes = set()
    invalid = []
    for m in tqdm.tqdm(get_all_architectures(all_ops, default_nodes, default_stages), total=all_count):
        has_zero = 3 in flatten(m)
        g, _ = get_model_graph(m)
        h = graph_hash(g)
        if h not in np_hashes:
            if m not in unique_graphs:
                invalid.append(m)
        np_hashes.add(h)
    print('Core:', len(without_zero))
    print('With zeros:', len(all_hashes))
    print('Unique:', len(unique_graphs))
    print('Np unique:', len(np_hashes))
    print('Invalid:', len(invalid))
    _use_np = False
    if invalid:
        inv = invalid[0]
        g, _ = get_model_graph(inv)
        h = graph_hash(g)
        conflicting = conflicts[h]
        show_model(invalid[0])
        show_model(conflicting)


def main():
    import functools
    from .search_space import get_all_architectures, get_search_space, all_ops, default_nodes, default_stages
    all_count = functools.reduce(lambda a,b: a*b, flatten(get_search_space(all_ops, default_nodes, default_stages)))
    
    unique_archs = set()
    hashes_to_archs = {}
    all_hashes = set()

    for m in tqdm.tqdm(get_all_architectures(all_ops, default_nodes, default_stages), total=all_count):
        g, _ = get_model_graph_np(m)
        h = graph_hash(g)
        if h not in all_hashes:
            # show_graph(g, '_'.join([str(d) for d in flatten(m)]), show=False)
            unique_archs.add(freeze(m))
            hashes_to_archs[h] = m

        all_hashes.add(h)

    print('Unique models:', len(all_hashes))

if __name__ == '__main__':
    import blox.search_space as ss
    # for m in ss.get_random_architectures(5, seed=100):
    hashes = set()
    for m in ss.get_all_architectures():
        ok = True
        for cfg in m:
            for opcfg in cfg:
                if opcfg[0] != 0:
                    ok = False
        if not ok:
            continue
        h = ss.get_model_hash(m)
        if h not in hashes:
            hashes.add(h)
            print(m)
            show_model(m, show=False, inc_full=False)
    # show_model([[[0], [0, 0, 1, 1]], [[0], [2, 0, 0, 1]], [[2], [2, 1, 1, 1]]], show=False, inc_full=False)
    # compare_nx_and_np()
    main()
