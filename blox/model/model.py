import collections

import torch
import torch.nn as nn

from . import operations as ops
from .. import graph_utils as gu
from .. import search_space as ss


_all_ops = []
for stage in range(ss.default_stages):
     for op in ss.all_ops:
         _all_ops.append((op, stage))
     _all_ops.append(('reduction', stage))
_all_ops.extend([('input', None), ('output', None), 'global'])

_opname_to_index = { name: idx for idx, name in enumerate(_all_ops) }


def get_adjacency_and_features(matrix, labels):
    # Add global node
    for row in matrix:
        row.insert(0, 0)

    nodes = len(matrix)
    global_row = [1 if i else 0 for i in range(nodes+1)] # zero followed by ones
    matrix.insert(0, global_row)

    labels.insert(0, 'global')

    # Add diag matrix
    for idx, row in enumerate(matrix):
        row[idx] = 1

    possible_ops = len(_opname_to_index)

    # Create features matrix from labels
    features = [[0 for _ in range(possible_ops)] for _ in range(nodes+1)]
    for idx, op in enumerate(labels):
        if op is not None:
            op = int(_opname_to_index[op])
            features[idx][op] = 1

    return matrix, features


class Node(nn.Module):
    available_ops = collections.OrderedDict({
        'conv': ops.ConvBlock,
        'mbconv': ops.MBConvBlock,
        'bconv': ops.BottleneckBlock,
        'zero': ops.Zero,
        'vit': ops.ViTBlock
    })

    optype_to_name = {
        val: key for key, val in available_ops.items()
    }

    def __init__(self, op, prev, input, sc, channels, imgsize):
        super().__init__()

        if not prev and not input and not sc:
            raise ValueError('Invalid model')

        if not prev and not input:
            self.op = None
        else:
            self.op = self.available_ops[op](channels, imgsize)

        self.prev = prev
        self.input = input
        self.sc = sc

    def reset_parameters(self):
        if self.op is not None:
            self.op.reset_parameters()

    def forward(self, prev, input):
        if self.op is not None:
            if self.prev and self.input:
                out = self.op(prev + input)
            elif self.prev:
                out = self.op(prev)
            elif self.input:
                out = self.op(input)
            else:
                assert False

            if self.sc:
                out = out + prev
        elif self.sc:
            out = prev
        else:
            assert False

        return out

class Cell(nn.Module):
    def __init__(self, nodes_cfg, channels, imgsize):
        super().__init__()

        self.nodes = nn.ModuleList()
        for cfg in nodes_cfg:
            if len(cfg) != 4:
                if len(cfg) == 1:
                    cfg.append(1)
                cfg.extend([0] * (4-len(cfg)))

            self.nodes.append(Node(*cfg, channels=channels, imgsize=imgsize))

    def reset_parameters(self):
        for node in self.nodes:
            node.reset_parameters()

    def forward(self, x):
        input = x
        for n in self.nodes:
            x = n(x, input)

        return x

class Reduction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 2, 2), nn.BatchNorm2d(out_channels), nn.ReLU())

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x):
        p = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if x.shape[-1] > p.shape[-1]:
            x = x[:,:,:-1,:-1]
        elif x.shape[-1] < p.shape[-1]:
            p = p[:,:,:-1,:-1]
        return x + p

class Block(nn.Module):
    def __init__(self, cell_cfg, channels, imgsize, cells_per_stage=1, scaling=2):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(cells_per_stage):
            self.layers.append(Cell(cell_cfg, channels, imgsize))
        self.layers.append(Reduction(channels, int(channels*scaling)))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class Model(nn.Module):
    def __init__(self, cfg, num_classes=100, stem=32, stem_kernel=3, stem_stride=1, stem_steps=1, imgsize=32, cells_per_stage=1, scaling=2, blocks=None, custom_stages=None):
        super().__init__()

        channels = 3

        stages = len(cfg)
        '''
        self.stem = []
        for i in range(stem_steps):
            self.stem.append(ops.GenericConv(channels, (i+1)*stem//stem_steps, kernel_size=stem_kernel, stride=stem_stride, padding=stem_kernel//2))
            channels = (i+1) * stem // stem_steps

        self.stem = nn.Sequential(*self.stem)
        '''
        self.stem = ops.GenericConv(channels, stem//stem_steps, kernel_size=stem_kernel, stride=stem_stride, padding=stem_kernel//2)
        channels = stem

        self.blocks = nn.ModuleList([])
        for s in range(stages):
            if custom_stages is None:
                cell_cfg = cfg[s]
                self.blocks.append(Block(cell_cfg, channels, imgsize, cells_per_stage, scaling))
                channels = int(channels * scaling)
                imgsize //= 2
            else:
                if s in custom_stages:
                    self.blocks.append(blocks.pop(0))
                else:
                    cell_cfg = cfg[s]
                    self.blocks.append(Block(cell_cfg, channels, imgsize, cells_per_stage, scaling))
                channels = int(channels * scaling)
                imgsize //= 2

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()

        final_neurons = channels
        self.classifier = nn.Linear(final_neurons, num_classes)

    def reset_parameters(self):
        self.stem.reset_parameters()
        for b in self.blocks:
            b.reset_parameters()
        self.classifier.reset_parameters()

    def set_blocks(self, blocks):
        for idx, b in enumerate(blocks):
            if b is not None:
                self.blocks[idx] = b

    def get_blocks(self):
        return [b for b in self.blocks]

    def get_config(self):
        pt = []
        for b in self.blocks:
            cell = b.layers[0]
            assert cell.nodes
            cfg = []
            for i, n in enumerate(cell.nodes):
                if not i:
                    cfg.append([ss.all_ops.index(Node.optype_to_name[type(n.op)])])
                else:
                    opidx = 0
                    if n.op is not None:
                        opidx = ss.all_ops.index(Node.optype_to_name[type(n.op)])
                    cfg.append([opidx, int(n.prev), int(n.input), int(n.sc)])

            if len(cfg) == 1:
                cfg.append([0, 0, 0, 1]) # skip second op

            pt.append(cfg)

        return pt

    def get_model_hash(self):
        pt = self.get_config()
        return ss.get_model_hash(pt)

    def get_graph(self, graph=None):
        # for a GCN predictor
        graph, _ = gu.get_model_graph(self.get_config(), ops=ss.all_ops, minimize=True, keep_dims=True)
        adj, labels = graph
        adj = adj.tolist()
        return get_adjacency_and_features(adj, labels)

    def _forward_normal(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)

        x = self.pool(x)
        x = self.flat(x)
        out = self.classifier(x)
        return out

    @torch.jit.unused
    def _forward_features(self, x):
        x = self.stem(x)

        fin = []
        fout = []

        for b in self.blocks:
            fin.append(x)
            x = b(x)
            fout.append(x)

        return fin, fout

    def forward(self, x, return_features: bool = False):
        if return_features:
            return self._forward_features(x)
        else:
            return self._forward_normal(x)

    def get_input_size(self):
        return [1,3,32,32]


def get_model(arch_vec, num_classes=100, stem=32, stem_kernel=3, stem_stride=1, stem_steps=1, imgsize=32, cells_per_stage=1, scaling=2, ops=None):
    import blox.search_space as ss
    import blox.graph_utils as gu
    opt = [gu.optimise_config(cell_cfg) for cell_cfg in arch_vec]
    if any(not cfg for cfg in opt):
        raise ValueError('Invalid model')
    opt = ss.arch_vec_to_names(opt, ops=ops)
    return Model(opt, num_classes=num_classes, stem=stem, stem_kernel=stem_kernel, stem_stride=stem_stride, stem_steps=stem_steps, imgsize=imgsize, cells_per_stage=cells_per_stage, scaling=scaling)


if __name__ == '__main__':
    import tqdm
    import copy
    import pickle
    from autocaml.generators.utils import maybe_wrap
    from autocaml.generators.blockwise.generator import DistillationRealization

    with open('data/unique_archs.pickle', 'rb') as f:
        ua = pickle.load(f)

    t = get_model([[[0], [1, 1, 1, 1]], [[1], [1, 1, 0, 0]], [[0], [0, 1, 1, 0]]])
    print(t.get_config(), t.get_model_hash())

    with tqdm.tqdm(ua) as q:
        for a2 in q:
            m2 = get_model(a2)
            m22 = copy.deepcopy(t)
            m22.set_blocks(m2.get_blocks())
            m222 = DistillationRealization(m22, teacher=None)
            print(m222.get_graph())
            assert ss.get_model_hash(a2) == m2.get_model_hash(), f'{a2} {m2.get_config()}'
            assert m22.get_model_hash() == m2.get_model_hash(), f'{m2.get_config()} {m22.get_config()}'

    # import argparse
    # import ptflops
    # import blox.search_space as ss
    # import blox.utils as utils

    # ss = ss.get_search_space()
    # flat_ss = utils.flatten(ss)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', '-m', type=int, nargs=len(flat_ss), required=True)
    # parser.add_argument('--stem', '-c', type=int, default=32)
    # parser.add_argument('--stem_kernel', '-k', type=int, default=3)
    # parser.add_argument('--stem_stride', '-s', type=int, default=1)
    # parser.add_argument('--imgsize', '-i', type=int, default=32)
    # parser.add_argument('--cells', '-r', type=int, default=1)
    # parser.add_argument('--scaling', '-a', type=float, default=2)
    # parser.add_argument('--num_classes', '-n', type=int, default=100)
    # parser.add_argument('--stem_steps', type=int, default=1)
    # args = parser.parse_args()

    # arch_vec = utils.copy_structure(args.model, ss)
    # model = get_model(arch_vec, args.num_classes, args.stem, args.stem_kernel, args.stem_stride, args.stem_steps, args.imgsize, args.cells, args.scaling)

    # print(ptflops.get_model_complexity_info(model, (3, args.imgsize, args.imgsize), print_per_layer_stat=False))
