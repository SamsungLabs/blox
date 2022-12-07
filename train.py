import os
import re
import sys
import random
import argparse
import datetime
import platform
import subprocess
from pprint import pformat

import torch
import torchvision
import numpy as np
import ptflops
import pkg_resources

import blox
import blox.model
import blox.training


_nvidia_driver_capture = re.compile(r'Driver Version: ([0-9]+\.[0-9]+(\.[0-9]+)?)')
_cuda_version_capture = re.compile(r'CUDA Version: ([0-9]+\.[0-9]+)')
_gpu_capture = re.compile('GPU ([0-9]+): ([a-zA-Z0-9 -]*) ')


def get_env_info(gpuidx):
    nvidia_smi_output = subprocess.run('nvidia-smi', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    nvidia_smi_output = nvidia_smi_output.stdout.decode('utf-8')
    driver = _nvidia_driver_capture.search(nvidia_smi_output).group(1)
    cuda = _cuda_version_capture.search(nvidia_smi_output).group(1)

    gpus_str = subprocess.run('nvidia-smi -L', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    gpus_str = gpus_str.stdout.decode('utf-8')
    gpus = { int(m.group(1)): m.group(2) for m in _gpu_capture.finditer(gpus_str) }


    return {
        'python': '.'.join(map(str, sys.version_info)),
        'gpu': [gpus[gidx] for gidx in gpuidx],
        'driver': driver,
        'cuda': cuda,
        'platform': platform.platform(),
        'codebase_version': blox.__version__,
        'codebase_commit': blox.__commit__,
        'torch_version': torch.__version__,
        'torch_commit': torch.version.git_version,
        'torch_cuda_version': torch.version.cuda,
        'torchvision_version': torchvision.__version__,
        'ptflops_version': pkg_resources.get_distribution('ptflops').version
    }


def train_single(arch_vec, seed, stem=32, stem_kernel=3, stem_stride=1, cells=1, scaling=2, args=None):
    if args is None:
        args = blox.training.Args() # defaults

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # see: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    h = blox.search_space.get_model_hash(arch_vec)
    print(arch_vec)
    print(h)

    cfgid = args.get_args_id()

    print(cfgid)

    if cfgid == 'ecb4eb2f2af07c0853e1bbe8041d616d': # default config
        args.save = f'seed_{seed}/{h}'
    else:
        args.save = f'cfg_{cfgid}/seed_{seed}/{h}'

    args.load = args.save

    if os.path.exists(os.path.join(args.wdir, args.save)):
        print('Experiment folder already exists! Will not train model to prevent overwrites!')
        return None

    if args.dataset == 'cifar10':
        classes = 10
    elif args.dataset == 'cifar100':
        classes = 100
    elif args.dataset == 'ImageNet':
        classes = 1000

    try:
        m = blox.model.get_model(arch_vec,
                num_classes=classes,
                stem=stem,
                stem_kernel=stem_kernel,
                stem_stride=stem_stride,
                cells_per_stage=cells,
                scaling=scaling)
    except ValueError:
        return {}

    t = blox.training.Trainer(m, args)

    start = datetime.datetime.today()
    t.train()
    end = datetime.datetime.today()

    test_info = t.test()
    t.history.setdefault('test', {})[t.epoch_idx] = test_info
    flops, params = ptflops.get_model_complexity_info(m, (3, 32, 32), print_per_layer_stat=False, as_strings=False, verbose=False)
    t.history['start'] = str(start)
    t.history['time'] = str(end - start)
    t.history['flops'] = flops
    t.history['params'] = params
    t.history['env'] = get_env_info(args.gpus)

    t.logger.info('Total training time: ' + str(end - start))
    t.logger.info('Final result:\n' + pformat(t.history, width=160))

    return t.history


def main():
    ss = blox.search_space.get_search_space()
    flat_ss = blox.utils.flatten(ss)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--model', '-m', type=int, nargs=len(flat_ss), required=True)
    parser.add_argument('--stem', type=int, default=32)
    parser.add_argument('--stem_kernel', type=int, default=3)
    parser.add_argument('--stem_stride', type=int, default=1)
    parser.add_argument('--cells', '-c', type=int, default=1)
    parser.add_argument('--scaling', type=float, default=2)
    args = blox.training.Args.from_stdargs(parser)

    arch_vec = blox.utils.copy_structure(args.model, ss)
    train_single(arch_vec, args.seed, args.stem, args.stem_kernel, args.stem_stride, args.cells, args.scaling, args)


if __name__ == '__main__':
    main()
