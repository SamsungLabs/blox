import os
import sys
import copy
import time
import random
import shutil
import logging
import contextlib
import collections
import collections.abc as cabc
from pprint import pformat

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from .utils import Cutout, AverageMeter, MultiAverageMeter, ProgressMeter, accuracy, KDLoss, NSR, KDLoss2, Subset, RandAugment


def make_nice_number(num):
    n = str(num)
    parts = (len(n)-1)//3 + 1
    if parts == 1:
        return n
    offset = len(n)%3 or 3
    breaks = [0] + [offset + i*3 for i in range(parts)] + [len(n)]
    return ','.join(n[breaks[i]:breaks[i+1]] for i in range(parts))


def print_model_summary(model, logger):
    logger.info(str(model))
    logger.info('======================')
    def _print(m, level=0):
        for n, child in m.named_children():
            logger.info('  '*level + type(child).__name__ + ' ' + n + ' ' + str(sum(p.numel() for p in child.parameters())))
            _print(child, level+1)
    _print(model)
    logger.info('======================')
    logger.info('Trainable parameters: ' + str(make_nice_number(sum(p.numel() for p in model.parameters()))))


def clone_weights(m):
    state_dict = m.state_dict()
    return type(state_dict)((k, t.clone()) for k, t in state_dict.items())


class ConvergenceError(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)


class Trainer():
    logger = None

    class LRPrinter():
        def __init__(self, trainer):
            self.t = trainer

        def __str__(self):
            return 'LR: {}'.format(round(self.t.lr,8))

    @staticmethod
    def get_logger(args, verbose=True):
        logger = logging.getLogger()
        logger.handlers.clear()
        stdhnd = logging.StreamHandler(sys.stdout)
        stdhnd.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(stdhnd)

        if args.wdir and args.save:
            logfile = os.path.join(args.wdir, args.save, 'log.txt')
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            fhnd = logging.FileHandler(logfile)
            fhnd.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
            logger.addHandler(fhnd)

        if verbose:
            logger.setLevel(logging.INFO)

        return logger

    def __init__(self, model, args, teacher_model=None, loaders=None, verbose=True):
        self.args = copy.deepcopy(args)
        self.logger = self.get_logger(args, verbose=verbose)

        if verbose:
            self.logger.info('Using arguments:')
            self.logger.info(pformat(self.args.asdict()))

            # self.logger.info('Training a model:')
            # print_model_summary(model, self.logger)

        self.model = model
        if self.args.blockwise or self.args.loss.startswith('kd'):
            self.teacher_model = teacher_model
        else:
            self.teacher_model = None

        self.cuda_device = None
        self.data_parallel = False

        if self.args.gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, self.args.gpus))
            if len(self.args.gpus) > 1 or self.args.force_data_parallel:
                self.cuda_device = 0
                self.data_parallel = True
                self.model = torch.nn.DataParallel(self.model).cuda(self.cuda_device)
                if self.teacher_model is not None:
                    self.teacher_model = torch.nn.DataParallel(self.teacher_model).cuda(self.cuda_device)
                self.logger.info('Using nn.DataParallel with the following GPUs: {}'.format(self.args.gpus))
            else:
                torch.cuda.set_device(0)
                self.cuda_device = 0
                self.model.cuda(self.cuda_device)
                if self.teacher_model is not None:
                    self.teacher_model = self.teacher_model.cuda(self.cuda_device)
                self.logger.info('Using single GPU: {}'.format(self.args.gpus[0]))

        self.logger.info(f'GPU:{self.args.gpus} cuda_device:{self.cuda_device} NUM_DEVICES_AVAILABLE={torch.cuda.device_count()}')

        self.epoch_idx = 0
        self.optimizer = Trainer.create_optimizer(self.model.parameters(), self.args)
        self.lr = self.args.lr

        self.lr_scheduler = Trainer.make_lr_scheduler(self.optimizer, self.args)

        if loaders is None:
            self.trainloader, self.validloader, self.testloader = Trainer.create_dataloader(self.args)
        else:
            self.trainloader, self.validloader, self.testloader = loaders

        self.train_batches = len(self.trainloader)
        self.logger.info('Training dataset has {} batches which gives total of {} examples'.format(self.train_batches, self.args.train_batch*self.train_batches))
        self.logger.info('    Valid dataset: {} and {}'.format(len(self.validloader), self.args.test_batch*len(self.validloader)))
        self.logger.info('    Test dataset:  {} and {}'.format(len(self.testloader), self.args.test_batch*len(self.testloader)))

        self.loss = Trainer.create_loss(self.args).cuda(self.cuda_device)
        self.total_steps = self.train_batches * args.epochs
        self.step_idx = 0

        self.best = None
        self.history = collections.OrderedDict()

        self.dump_tensorboard = self.args.dump_tensorboard
        self.graph_dumped = False
        self.tb_name = os.path.join(self.args.wdir, self.args.save or 'tmp', self.args.tb_logdir)
        if self.dump_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.tb_name)

        self._best_weights = None

        if self.teacher_model is not None:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

    def _is_new_best(self, info):
        if not self.best:
            return [True] * len(info['loss'])

        ret = []
        for idx in range(len(info['loss'])):
            if info['top1'] and info['top1'][idx] > self.best[0]['top1'][idx]:
                ret.append(True)
            elif (not info['top1'] or info['top1'][idx] == self.best[0]['top1'][idx]) and info['loss'][idx] <= self.best[0]['loss'][idx]:
                ret.append(True)
            else:
                ret.append(False)
        return ret


    def train(self, ignore_convergence_errors=True):
        if self.load() and self.epoch_idx != 0:
            best_scores = ' '.join(f'{key}: {value}' for key, value in self.best[0].items())
            self.logger.info('Continuing training from epoch {self.epoch_idx}, best model so far is: {best_scores}'.format(self=self, best_scores=best_scores))

        train_info = None
        try:
            while self.epoch_idx < self.args.epochs:
                if self.args.eval_only:
                    self.logger.info('Skipping training due to --eval_only')
                    break

                if not self.epoch_idx and self.args.warmup_bn:
                    for m in self.model.modules():
                        if hasattr(m, 'reset_running_stats'):
                            m.reset_running_stats()

                    self._generic_epoch(self.trainloader, 'warmup')

                train_info = self.epoch()
                self.history.setdefault('train', {})[self.epoch_idx] = train_info
                train_scores = ' '.join([f'{key}: {value}' for key, value in train_info.items()])

                self.logger.info('[Epoch {self.epoch_idx}]: {train_scores}'.format(self=self, train_scores=train_scores))

                val_info = {}

                if (self.args.val_freq and self.epoch_idx % self.args.val_freq == 0) or self.epoch_idx == self.args.epochs:
                    val_info = self.validate()

                    self.history.setdefault('valid', {})[self.epoch_idx] = val_info

                    aliases = []
                    new_bests = self._is_new_best(val_info)
                    if not self.best:
                        self.best = (val_info.copy(), [self.epoch_idx] * len(val_info['loss']))
                        if len(val_info['loss']) > 1:
                            self._best_weights = [clone_weights(self.model[idx]) for idx in range(len(val_info['loss']))]
                        else:
                            self._best_weights = clone_weights(self.model)
                    else:
                        for idx in range(len(new_bests)):
                            if not new_bests[idx]:
                                continue

                            self.best[1][idx] = self.epoch_idx
                            for k, v in val_info.items():
                                if self.best[0][k]:
                                    self.best[0][k][idx] = v[idx]

                            if len(new_bests) == 1:
                                aliases.append('best')
                                self._best_weights = clone_weights(self.model)
                            else:
                                self._best_weights[idx] = clone_weights(self.model[idx])

                        if self.args.save_freq and self.epoch_idx % self.args.save_freq == 0:
                            aliases.append(str(self.epoch_idx))

                    epoch_scores = ' '.join([f'{key}: {value}' for key, value in val_info.items()])
                    best_scores = ' '.join(f'{key}: {value}' for key, value in self.best[0].items())
                    self.logger.info('[Validate {self.epoch_idx}]: {epoch_scores} (Best: {best_scores} @{self.best[1]})'.format(self=self, epoch_scores=epoch_scores, best_scores=best_scores))

                    self.save(aliases=aliases)

                #update learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    self.lr = self.lr_scheduler.get_last_lr()[0]

            if train_info is None and not self.best:
                val_info = self.validate()
                self.best = (val_info.copy(), [self.epoch_idx] * len(val_info['loss']))
                if not self.args.eval_only:
                    if len(val_info['loss']) > 1:
                        self._best_weights = [clone_weights(self.model[idx]) for idx in range(len(val_info['loss']))]
                    else:
                        self._best_weights = clone_weights(self.model)
                self.history.setdefault('valid', {})[self.epoch_idx] = val_info

            return train_info, self.best[0]

        except ConvergenceError as e:
            if ignore_convergence_errors:
                self.logger.error('A convergence error occurred while training a model at epoch {}: {}'.format(self.epoch_idx, e))
                if self.best is None:
                    self.best = ({ 'loss': float('inf'), 'top1': [0.0], 'top5': [0.0] }, -1)
                return { 'loss': float('inf'), 'top1': [0.0], 'top5': [0.0]}, self.best[0]['top1']
            else:
                raise

    def _generic_epoch(self, loader, stage):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = MultiAverageMeter('Loss', ':.4e')
        top1 = MultiAverageMeter('Acc@1', ':6.2f')
        top5 = MultiAverageMeter('Acc@5', ':6.2f')

        is_training = (stage == 'train')

        if is_training:
            self.epoch_idx += 1
            this_epoch = self.epoch_idx

        to_print = [batch_time, data_time, losses, top1, top5]
        if is_training:
            to_print.append(Trainer.LRPrinter(self))

        prefix = '{}: '.format(stage.capitalize())
        if is_training:
            prefix = 'Epoch: [{}]'.format(self.epoch_idx)

        progress = ProgressMeter(
            len(loader),
            to_print,
            prefix=prefix,
            logger=self.logger)

        with contextlib.ExitStack() as stack:
            if is_training or stage == 'warmup':
                self.model.train()
            else:
                self.model.eval()
                stack.enter_context(torch.no_grad())

            end = time.time()
            for i, (images, target) in enumerate(loader):
                # measure data loading time
                data_time.update(time.time() - end)

                if self.cuda_device is not None:
                    images = images.cuda(self.cuda_device, non_blocking=True)
                    target = target.cuda(self.cuda_device, non_blocking=True)

                if is_training:
                    self.epoch_idx = this_epoch + i*(1/self.train_batches)
                    assert int(self.epoch_idx) == this_epoch

                if self.teacher_model is not None:
                    with torch.no_grad():
                        if self.args.blockwise:
                            tin, tout = self.teacher_model(images, return_features=True)
                        else:
                            toutput = self.teacher_model(images)
                else:
                    toutput = None

                # compute output
                if not self.args.auxiliary:
                    if self.args.blockwise:
                        output = self.model(*tin)
                    else:
                        output = self.model(images)
                else:
                    assert not self.args.blockwise
                    output, output_aux = self.model(images)

                if self.args.blockwise:
                    loss = None
                    block_losses = []
                    for bo, bt in zip(output, tout):
                        if bo is not None:
                            _loss = self.loss(bo, bt)
                            block_losses.append(_loss.detach().item())
                            if loss is None:
                                loss = _loss
                            else:
                                loss += _loss
                        else:
                            block_losses.append(0)

                    assert loss is not None, 'No blocks?'
                elif self.teacher_model is not None:
                    loss = self.loss(output, target, toutput)
                else:
                    loss = self.loss(output, target)

                if self.args.auxiliary:
                    loss_aux = self.loss(output_aux, target)
                    loss += self.args.auxiliary_weight*loss_aux

                if is_training:
                    if torch.isnan(loss):
                        raise ConvergenceError('Loss is NaN')

                # measure accuracy and record loss
                if self.args.blockwise:
                    losses.update(block_losses)
                else:
                    losses.update(loss, images.size(0))
                if not self.args.blockwise:
                    acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
                    top1.update(acc1, images.size(0))
                    top5.update(acc5, images.size(0))

                if stage != 'test':
                    if self.dump_tensorboard and i % self.args.print_freq == 0:
                        if not self.graph_dumped:
                            self.tb_writer.add_graph(self.model, images)
                            self.graph_dumped = True
                        x_val = int(self.epoch_idx-1)*len(loader)+i
                        self.tb_writer.add_scalar('loss/{}'.format(stage), loss, x_val)
                        if not self.args.blockwise:
                            self.tb_writer.add_scalar('top1/{}'.format(stage), acc1, x_val)
                            self.tb_writer.add_scalar('top5/{}'.format(stage), acc5, x_val)

                if is_training:
                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.args.grad_clip != False:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if self.args.print_freq and i % self.args.print_freq == 0:
                    progress.display(i)

                self.step_idx += 1

        if is_training:
            self.epoch_idx = this_epoch

        return {
            'loss': losses.avg,
            'top1': top1.avg,
            'top5': top5.avg
        }


    def epoch(self):
        return self._generic_epoch(self.trainloader, 'train')

    def validate(self):
        return self._generic_epoch(self.validloader, 'validate')

    def test(self, load_best=True):
        if load_best:
            if self._best_weights is not None:
                self.logger.info('Recalling best weights from memory')
                if len(self.best[1]) > 1:
                    for idx in range(len(self.best[1])):
                        self.model[idx].load_state_dict(self._best_weights[idx])
                else:
                    self.model.load_state_dict(self._best_weights)
            else:
                self.logger.info('Trying to load a checkpoint for the best model')
                self._load_weights('best')
        return self._generic_epoch(self.testloader, 'test')

    def save(self, epoch=None, aliases=None):
        if self.args.dont_save:
            self.logger.info('Not saving a model due to --dont_save')
            return

        aliases = aliases or []
        base_dir = os.path.join(self.args.wdir, self.args.save or 'tmp')
        model = os.path.join(base_dir, 'model.{}.pt'.format(epoch or 'latest'))
        self.logger.info('Saving a checkpoint to {} with the following aliases: {}'.format(repr(model), aliases))

        checkpoint = {
            'best': self.best,
            'epoch': self.epoch_idx,
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'history': self.history
        }

        os.makedirs(os.path.dirname(model), exist_ok=True)
        torch.save(checkpoint, model)
        for alias in aliases:
            shutil.copyfile(model, os.path.join(base_dir, 'model.{}.pt'.format(alias)))

    def load(self):
        if not self.args.load:
            self.logger.info('Not loading a model')
            return False

        model = os.path.join(self.args.wdir, self.args.load)
        if os.path.isdir(model):
            model = os.path.join(model, 'model.latest.pt')

        if not os.path.exists(model):
            self.logger.info('No checkpoint to load {!r}'.format(model))
            return False

        self.logger.info('Loading a checkpoint: {}'.format(repr(model)))
        checkpoint = torch.load(model)
        self.best = checkpoint['best']
        self.epoch_idx = checkpoint['epoch']
        self.step_idx = self.epoch_idx * self.train_batches
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['opt'])
        self.history = checkpoint['history']
        self.lr_scheduler = Trainer.make_lr_scheduler(self.optimizer, self.args, self.epoch_idx)
        if self.lr_scheduler is not None:
            self.lr = self.lr_scheduler.get_last_lr()[0]
        else:
            self.lr = self.args.lr

        return True

    def _load_weights(self, epoch=None):
        found = False
        for a in [self.args.save, self.args.load, 'tmp']:
            if not a:
                continue
            base_dir = os.path.join(self.args.wdir, a)
            model = os.path.join(base_dir, 'model.{}.pt'.format(epoch or 'latest'))
            if not os.path.exists(model):
                self.logger.info('No checkpoint to load {!r}'.format(model))
                continue

            found = True
            break

        if not found:
            return False

        self.logger.info('Loading weights only from a checkpoint: {}'.format(repr(model)))
        checkpoint = torch.load(model)
        self.model.load_state_dict(checkpoint['model'])


    @staticmethod
    def create_optimizer(model_params, args):
        if args.opt == 'Adam':
            return torch.optim.Adam(model_params, args.lr, betas=args.betas, eps=args.eps, weight_decay=args.decay)
        elif args.opt == 'SGD':
            return torch.optim.SGD(model_params, lr=args.lr, momentum=args.momentum, weight_decay=args.decay, nesterov=args.nesterov)
        elif args.opt == 'RMSprop':
            return torch.optim.RMSprop(model_params, lr=args.lr, weight_decay=args.decay)
        else:
            raise ValueError('Unknown optimizer type: {}'.format(args.opt))

    @staticmethod
    def make_lr_scheduler(optim, args, last_epoch=-1):
        if args.lr_schedule == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optim, float(args.epochs), eta_min=args.lr_min, last_epoch=last_epoch)
        elif args.lr_schedule == 'step':
            return torch.optim.lr_scheduler.StepLR(optim, max(1, args.epochs // 4), gamma=args.lr_decay, last_epoch=last_epoch)
        elif args.lr_schedule == 'decay':
            def sch(epoch):
                return args.decay ** (epoch // args.every)
            return torch.optim.lr_scheduler.LambdaLR(optim, sch)
        elif args.lr_schedule == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=10, threshold=0.01, verbose=True)
        elif isinstance(args.lr_schedule, cabc.Sequence):
            return torch.optim.lr_scheduler.MultiStepLR(optim, list(sorted(args.lr_schedule)), args.lr_decay, last_epoch=last_epoch)
        elif args.lr_schedule == 'constant':
            return None
        else:
            raise ValueError('Unknown LR schedule type: {}'.format(args.lr_schedule))

    @staticmethod
    def create_dataloader(args):
        datadir = os.path.join(args.datadir)
        dataset_ctor = None
        exp_input_size = None
        norm_args = None
        req_resize = None
        if args.dataset == 'cifar10':
            dataset_ctor = torchvision.datasets.CIFAR10
            exp_input_size = (32, 32)
            norm_args = (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)
        elif args.dataset == 'cifar100':
            dataset_ctor = torchvision.datasets.CIFAR100
            exp_input_size = (32, 32)
            norm_args = (0.5070588235294118, 0.48666666666666664, 0.4407843137254902), (0.26745098039215687, 0.2564705882352941, 0.27607843137254906)
        elif args.dataset == 'mnist':
            dataset_ctor = torchvision.datasets.MNIST
            exp_input_size = (28, 28)
            norm_args = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        elif args.dataset == 'imagenet':
            dataset_ctor = torchvision.datasets.ImageFolder
            exp_input_size = (224, 224)
            norm_args = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            req_resize = (256, 256)
        else:
            raise ValueError('Unknown dataset: {}'.format(args.dataset))

        traintransforms = []
        if args.rand_aug:
            traintransforms.append(RandAugment(args.rand_aug_num, args.rand_aug))
        if args.hflip:
            traintransforms.append(transforms.RandomHorizontalFlip(args.hflip))
        if args.vflip:
            traintransforms.append(transforms.RandomVerticalFlip(args.vflip))
        if args.crop:
            traintransforms.append(transforms.RandomCrop(exp_input_size, padding=args.random_crop_pad))
        elif req_resize:
            traintransforms.append(transforms.Resize(req_resize))
            traintransforms.append(transforms.CenterCrop(exp_input_size))
        if args.rotate:
            traintransforms.append(transforms.RandomRotation(args.rotate))

        validtransforms = []
        if req_resize:
            validtransforms.append(transforms.Resize(req_resize))
            validtransforms.append(transforms.CenterCrop(exp_input_size))

        stdtransforms = [
            transforms.ToTensor()
        ]

        if args.normalize:
            stdtransforms.append(transforms.Normalize(*norm_args))

        traintransforms = transforms.Compose(traintransforms + stdtransforms)
        validtransforms  = transforms.Compose(validtransforms + stdtransforms)

        #add cutout at the end if enabled
        if args.pad:
            traintransforms.transforms.append(transforms.Pad(args.pad))
        if args.cutout:
            traintransforms.transforms.append(Cutout(n_holes=args.cutout_holes, length=args.cutout_length, prob=args.cutout_prob))

        if args.dataset != 'imagenet':
            if args.hdf5:
                raise ValueError('HDF5 currently only supported for imagenet.')
            trainset = dataset_ctor(root=datadir, train=True,  download=True)
            testset = dataset_ctor(root=datadir, train=False, download=True)
        else:
            if args.hdf5:
                from h5py_dataset import H5Dataset
                trainset = H5Dataset(os.path.join(datadir, 'imagenet-train-256.h5'))
                testset = H5Dataset(os.path.join(datadir, 'imagenet-val-256.h5'))
            else:
                trainset = dataset_ctor(root=os.path.join(datadir, 'train'))
                testset = dataset_ctor(root=os.path.join(datadir, 'val'))

        # come up with indices used for training and validation
        # if validation indices (or train split) are set,
        # validation set is constructed by taking a subset of the training set
        # (training set is shrunk), otherwise test test is used to do both
        # testing and validation
        train_indices = list(range(len(trainset)))
        test_indices = list(range(len(testset)))
        valid_indices = args.val_split
        if valid_indices is None:
            if args.train_portion is not None and args.train_portion != 1.0:
                def _split_set(indices):
                    split_pt = int(len(indices) * args.train_portion)
                    if args.split_seed is not None:
                        s = np.random.get_state()
                        np.random.seed(args.split_seed)
                        indices_to_split = np.random.permutation(indices).tolist()
                        np.random.set_state(s)
                    else:
                        indices_to_split = indices
                    indices, valid_indices = indices_to_split[:split_pt], indices_to_split[split_pt:]
                    return indices, valid_indices

                if not args.split_test:
                    train_indices, valid_indices = _split_set(train_indices)
                else:
                    test_indices, valid_indices = _split_set(test_indices)
        else:
            if args.train_portion is not None:
                raise ValueError('--train_portion and --val_split are mutually exclusive - you should only use one')

            orig_set = train_indices
            if args.split_test:
                orig_set = test_indices

            orig_set = set(orig_set)
            valid_indices = set(args.val_split)
            missing = valid_indices.difference(orig_set)
            if missing:
                raise ValueError(missing)

            orig_set = list(orig_set.difference(valid_indices))
            valid_indices = list(valid_indices)
            if args.split_test:
                test_indices = orig_set
            else:
                train_indices = orig_set

        validset = testset
        if valid_indices:
            if args.split_test:
                assert len(valid_indices) + len(test_indices) == len(testset)
                assert len(train_indices) == len(trainset)
                validset = Subset(testset, valid_indices)
                testset = Subset(testset, test_indices)
            else:
                assert len(valid_indices) + len(train_indices) == len(trainset)
                assert len(test_indices) == len(testset)
                validset = Subset(trainset, valid_indices)
                trainset = Subset(trainset, train_indices)
        else:
            assert len(train_indices) == len(trainset)
            assert len(test_indices) == len(testset)

        trainset.transform = traintransforms
        if args.apply_train_transforms_to_validation:
            validset.transform = traintransforms
            testset.transform = traintransforms
        else:
            validset.transform = validtransforms
            testset.transform = validtransforms

        g = None
        init_fn = None
        if args.reseed_workers is not None:
            g = torch.Generator()
            g.manual_seed(args.reseed_workers)
            init_fn = Trainer.reseed_worker

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=args.shuffle, num_workers=args.workers, pin_memory=args.pin_memory, worker_init_fn=init_fn, generator=g)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory, worker_init_fn=init_fn, generator=g)
        if validset is testset:
            validloader = testloader
        elif validset is trainset:
            validloader = trainloader
        else:
            validloader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch,  shuffle=False, num_workers=args.workers, pin_memory=args.pin_memory, worker_init_fn=init_fn, generator=g)

        return trainloader, validloader, testloader

    @staticmethod
    def create_loss(args):
        if args.loss == 'xentropy':
            return torch.nn.CrossEntropyLoss()
        elif args.loss == 'mse':
            return torch.nn.MSELoss()
        elif args.loss == 'nsr':
            return NSR()
        elif args.loss == 'kd':
            return KDLoss(args.alpha, args.T)
        elif args.loss == 'kd2':
            return KDLoss2(args.alpha, args.T, args.topk)

    @staticmethod
    def reseed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
