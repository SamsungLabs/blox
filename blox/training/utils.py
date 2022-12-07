import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as t
import torchvision.transforms.functional as TVF


#taked from https://github.com/uoguelph-mlrg/Cutout
class Cutout():
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, prob):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        if np.random.random() > self.prob:
            return img

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        try:
            val = val.detach().item()
        except:
            pass

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MultiAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self._scalar = None
        self.reset()

    def reset(self):
        self.val = []
        self.avg = []
        self.sum = []
        self.count = 0

    def update(self, val, n=1):
        try:
            val = val.detach().cpu().numpy().tolist()
        except:
            pass

        if not isinstance(val, list):
            val = [val]
            if self._scalar is None:
                self._scalar = True
        elif self._scalar is None:
            self._scalar = False

        if not self.count:
            self.val = [0] * len(val)
            self.avg = [0] * len(val)
            self.sum = [0] * len(val)

        self.count += n
        for idx, v in enumerate(val):
            self.val[idx] = v
            self.sum[idx] += v * n
            self.avg[idx] = self.sum[idx] / self.count

    def __str__(self):
        if self._scalar:
            fmtstr = '{name} {val[0]' + self.fmt + '} ({avg[0]' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)

        val_str = '[' + ', '.join([("{" + self.fmt + "}").format(v) for v in self.val]) + ']'
        avg_str = '[' + ', '.join([("{" + self.fmt + "}").format(v) for v in self.avg]) + ']'
        return f'{self.name} {val_str} ({avg_str})'


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger:
            self.logger.info('\t'.join(entries))
        else:
            print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t() # K, B
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class KDLoss(nn.Module):
    def __init__(self, alpha, T=5):
        super().__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, logits, label, teacher):
        hard = (1 - self.alpha) * F.cross_entropy(logits, label)
        soft = self.alpha * F.kl_div(F.log_softmax(logits/self.T, dim=1), F.log_softmax(teacher/self.T, dim=1), reduction='batchmean', log_target=True)
        return hard + soft


def kdtopk(probs, topk, classes=None):
    with torch.no_grad():
        if classes is None:
            classes = probs.shape[1]

        missing_classes = classes - topk

        topk_probs = torch.topk(probs, topk)
        missing_prob = 1 - topk_probs.values.sum(dim=1)
        probs[:,:] = missing_prob.unsqueeze(1) / missing_classes
        probs.scatter_(1, topk_probs.indices, topk_probs.values)

    return probs


class KDLoss2(nn.Module):
    def __init__(self, alpha, T=5, topk=25):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.topk = topk

    def forward(self, logits, label, teacher):
        hard = (1 - self.alpha) * F.cross_entropy(logits, label)
        with torch.no_grad():
            classes = teacher.shape[1]
            tprob = F.softmax(logits/self.T, dim=1)
            tprob = kdtopk(tprob, self.topk, classes)

        soft = self.alpha * F.kl_div(F.log_softmax(logits/self.T, dim=1), tprob, reduction='batchmean', log_target=False)
        return hard + soft


class NSR(nn.Module):
    def forward(self, stu_features, tea_features):

        stu_features_split = torch.chunk(stu_features, stu_features.shape[1], dim=1)
        tea_features_split = torch.chunk(tea_features, tea_features.shape[1], dim=1)
        loss = 0
        for s, t in zip(stu_features_split, tea_features_split):
            loss += F.mse_loss(s, t)/t.var()

        return loss / stu_features.shape[1]


class Subset(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        if self.transform:
            im = self.transform(im)
        return im, labels

    def __len__(self):
        return len(self.indices)


def _apply_op(img: torch.Tensor, op_name: str, magnitude: float,
              interpolation: t.InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = TVF.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = TVF.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = TVF.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = TVF.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = TVF.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = TVF.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = TVF.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = TVF.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = TVF.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = TVF.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = TVF.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = TVF.autocontrast(img)
    elif op_name == "Equalize":
        img = TVF.equalize(img)
    elif op_name == "Invert":
        img = TVF.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31,
                 interpolation: t.InterpolationMode = t.InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(self.num_magnitude_bins, TVF._get_image_size(img))
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_ops={num_ops}'
        s += ', magnitude={magnitude}'
        s += ', num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)
