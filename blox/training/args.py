import pickle
import hashlib


class arg_types():
    @staticmethod
    def uint(val, inc_zero=True):
        val = int(val)
        if val <= 0 and (not inc_zero or val < 0):
            raise ValueError('Expected unsigned integer ({} zero) but got value {}'.format('including' if inc_zero else 'excluding', val))
        return val

    @staticmethod
    def positive_uint(val):
        return arg_types.uint(val, inc_zero=False)

    @staticmethod
    def positive_float(val):
        val = float(val)
        if val <= 0:
            raise ValueError('Expected positive float value, got: {}'.format(val))
        return val

    @staticmethod
    def probability(val):
        val = float(val)
        if val < 0.0 or val > 1.0:
            raise ValueError('Expected float between 0 and 1, got: {}'.format(val))
        return val

    @staticmethod
    def simple_list(val, elem_type):
        return list(map(lambda x: elem_type(x.strip()), val.split(',')))

    @staticmethod
    def uint_list(val):
        return arg_types.simple_list(val, arg_types.uint)

    @staticmethod
    def positive_uint_list(val):
        return arg_types.simple_list(val, arg_types.positive_uint)

    @staticmethod
    def positive_float_list(val):
        return arg_types.simple_list(val, arg_types.positive_float)

    @staticmethod
    def alternative(*types):
        def _try(val):
            for t in types:
                try:
                    return t(val)
                except ValueError:
                    pass
            raise ValueError('Non of the expected data types ({}) can be constructed from the passed argument value: {}'.format(types, val))

        return _try


class Args():
    _known_args = {
        # optimizer
        'opt': ('SGD', 'Optimizer type'),
        'lr': (0.01, 'Learning rate', { 'type': float }),
        'lr_min': (0, 'Minimum learning rate (passed to lr scheduler)', { 'type': float }),
        'lr_schedule': ('cosine', 'Learning rate scheduler type', { 'type': arg_types.alternative(str, arg_types.positive_uint_list) }),
        'lr_decay': (None, 'Learning rate decay (passed to lr scheduler, if applicable)', { 'type': float }),
        'betas': ((0.9, 0.999), 'Adam betas', { 'type': tuple }),
        'eps': (1e-8, 'Adam episolon', { 'type': float }),
        'decay': (5e-4, 'Weight decay', { 'type': float }),
        'momentum': (0.9, 'Momentum for SGD optim', { 'type': float }),
        'nesterov': (False, 'Whether to use Nesterov momentum', { 'action': 'store_true' }),
        'loss': ('xentropy', 'Loss function to use'),
        'alpha': (0.9, 'For KD: alpha*KL + (1-alpha)*CE', { 'type': arg_types.probability }),
        'T': (1.0, 'Softmax temperature for KD', { 'type': float }),
        'topk': (25, 'TopK classes for KD2', { 'type': int }),
        'grad_clip': (False, 'Gradient clipping', { 'type': float }),
        'auxiliary': (False, 'Auxiliary tower a la GoogLeNet', { 'action': 'store_true' }),
        'auxiliary_weight': (0.4, 'Weight for auxiliary tower', { 'type': float }),
        'force_data_parallel': (False, 'Force using nn.DataParallel even if only one GPU is used', { 'action': 'store_true' }),

        # datasets and data agumentation
        'dataset': ('cifar100', 'Dataset to use for training'),
        'train_batch': (256, 'Training Batch size', { 'type': arg_types.positive_uint }),
        'test_batch': (100, 'Test Batch size', { 'type': arg_types.positive_uint }),
        'shuffle': (True, 'Shuffle dataset', { 'action': 'store_true' }),
        'workers': (6, 'Number of workers to use when preparing data', { 'type': arg_types.uint }),
        'normalize': (True, 'Whether to normalize dataset images', { 'action': 'store_true' }),
        'rotate': (False, 'Random rotation to apply to train images', { 'type': float }),
        'hflip': (False, 'Probability of randomly flipping a training image horizontally', { 'type': arg_types.probability }),
        'vflip': (False, 'Probability of randomly flipping a training image vertically', { 'type': arg_types.probability }),
        'pad': (False, 'Fixed padding to apply to a training image', { 'type': arg_types.uint }),
        'random_crop_pad': (0, 'Padding applied to a training image after random crop', { 'type': arg_types.uint }),
        'crop': (False, 'Crop random part of a training image and resize to the desired size', { 'action': 'store_true' }),
        'cutout': (False, 'Apply cutout', { 'action': 'store_true' }),
        'cutout_holes': (1, 'Number of cutout holes', { 'type': arg_types.uint }),
        'cutout_length': (16, 'Length of cutout', { 'type': arg_types.uint }),
        'cutout_prob': (1.0, 'Probability of doing cutout', { 'type': arg_types.probability }),
        'rand_aug': (0, 'Magnitude of RangAugment', { 'type': arg_types.uint }),
        'rand_aug_num': (2, 'Number of transformations in RangAugment', { 'type': arg_types.uint }),

        'datadir': ('./data', 'Directory to hold datasets'),
        'pin_memory': (True, 'Whether to pin the memory used to store training/validation images', { 'action': 'store_true' }),
        'hdf5': (False, 'Use data in hdf5 format', { 'action': 'store_true' }),

        'val_split': (None, 'Indices to extract from the training set as validation set', { 'type': arg_types.uint_list }),
        'split_test': (False, 'Whether to split test set rather than training set to make a validation set', { 'action': 'store_true' }),
        'train_portion': (0.9, 'Portion of training data to use for train vs validation', { 'type': arg_types.probability }), # not really probability but the same range
        'split_seed': (0, 'Random seed to use when extracting validation set by randomly splitting training/testing set. Only used if --train_portion is used, None means '
                             'that orderly split (N last elements) should be used instead of random, if set to a number this number will be used as a seed for random '
                             'permutation (np.random.permute) which is performed before taking the last N elements as a validation set.', { 'type': arg_types.uint }),
        'apply_train_transforms_to_validation': (False, 'Apply train transformations on validation set (instead of applying plain test-set-like transforms)', { 'action': 'store_true' }),
        'reseed_workers': (0, 'Reseed workers to preserve reproducibility', { 'type': arg_types.uint }),

        # misc
        'epochs': (200, 'Number of epochs to run', { 'type': arg_types.uint }),
        'gpus': (None, 'GPU to use. If empty/None use CPU.', { 'type': arg_types.uint_list }),
        'blockwise': (False, 'Run training in a blockwise mode', { 'action': 'store_true' }),
        'eval_only': (False, 'Do not train, only evaluate model', { 'action': 'store_true' }),
        'warmup_bn': (False, 'Warmup batch normalization before the first epoch', { 'action': 'store_true' }),

        # filesystem
        'wdir': ('exp', 'Folder with experiments'),
        'save': ('', 'Name of the experiment used to save stats. If ommited, a generic "tmp" folder inside "wdir" will be used. The tmp experiment cannot be loaded.'),
        'load': ('', 'Name of the experiment to load. If ommited (i.e. using the generic "tmp" folder) no loading is done'),
        'print_freq': (64, 'Number of batches to process between consecutive status prints', { 'type': arg_types.uint }),
        'save_freq': (None, 'Number of epochs to process between consecutive checkpoint saves', { 'type': arg_types.uint }),
        'val_freq': (1, 'Number of epochs to process between val', { 'type': arg_types.uint }),
        'dump_tensorboard': (False, 'Dump model to tensorboard (default_dir=tb_run, change with --tb_logdir)', { 'action': 'store_true' }),
        'tb_logdir': ('tb_run', 'Tensorboard directory'),
        'dont_save': (False, 'Completely disable model checkpointing', { 'action': 'store_true' })
    }

    def __init__(self, args=None, inc_default=True):
        assert inc_default or args
        if inc_default:
            for name, value in Args._known_args.items():
                if not args or name not in args:
                    setattr(self, name, value[0])
        if args:
            for name, value in args.items():
                setattr(self, name, value)

    @staticmethod
    def from_stdargs(parser=None):
        import sys
        return Args.from_list(sys.argv[1:], parser)

    @staticmethod
    def from_list(args, parser=None):
        if parser is None:
            import argparse
            parser = argparse.ArgumentParser()

        for name, value in Args._known_args.items():
            default = value[0]
            helpstr = value[1]
            kwargs = {}
            if len(value) > 2:
                kwargs = value[2]

            parser.add_argument('--' + name, default=default, help=helpstr, **kwargs)

        args = parser.parse_args(args=args)
        return Args(vars(args), inc_default=False)

    def asdict(self):
        return self.__dict__.copy()

    def __eq__(self, other):
        if not isinstance(other, Args):
            return False

        k1 = set(self.__dict__.keys())
        k2 = set(other.__dict__.keys())
        if k1 != k2:
            return False

        for ka in self.__dict__.keys():
            if getattr(self, ka, None) != getattr(other, ka, None):
                return False

        return True

    def __str__(self):
        return str(self.asdict())

    def __repr__(self):
        return repr(self.asdict())

    def get_meaningful_args(self):
        return {
            'opt': self.opt,
            'lr': self.lr,
            'lr_min': self.lr_min,
            'lr_schedule': self.lr_schedule,
            'betas': self.betas if self.opt == 'Adam' else None,
            'eps': self.eps if self.opt == 'Adam' else None,
            'decay': self.decay,
            'momentum': self.momentum if self.opt == 'SGD' else None,
            'nesterov': self.nesterov if self.opt == 'SGD' else None,
            'loss': self.loss,
            'alpha': self.alpha if self.loss in ['kd', 'kd2'] else None,
            'T': self.T if self.loss in ['kd', 'kd2'] else None,
            'topk': self.topk if self.loss == 'kd2' else None,
            'grad_clip': self.grad_clip,
            'auxiliary': self.auxiliary,
            'auxiliary_weight': self.auxiliary_weight if self.auxiliary else None,
            'force_data_parallel': self.force_data_parallel,
            'dataset': self.dataset,
            'train_batch': self.train_batch,
            'shuffle': self.shuffle,
            'normalize': self.normalize,
            'rotate': self.rotate,
            'hflip': self.hflip,
            'vflip': self.vflip,
            'pad': self.pad,
            'random_crop_pad': self.random_crop_pad,
            'crop': self.crop,
            'cutout': self.cutout,
            'cutout_holes': self.cutout_holes if self.cutout else None,
            'cutout_length': self.cutout_length if self.cutout else None,
            'cutout_prob': self.cutout_prob if self.cutout else None,
            'rand_aug': self.rand_aug,
            'rand_aug_num': self.rand_aug_num if self.rand_aug else None,
            'val_split': self.val_split,
            'split_test': self.split_test if (self.val_split or self.train_portion) else None,
            'train_portion': self.train_portion,
            'split_seed': self.split_seed,
            'apply_train_transforms_to_validation': self.apply_train_transforms_to_validation,
            'reseed_workers': self.reseed_workers,
            'epochs': self.epochs,
            'blockwise': self.blockwise,
            'eval_only': self.eval_only,
            'warmup_bn': self.warmup_bn,
        }

    def get_args_id(self):
        args = list(self.get_meaningful_args().items())
        args = sorted(args, key=lambda p: p[0]) # sort by key
        args = str(args).encode('utf8')
        return hashlib.md5(args).hexdigest()
