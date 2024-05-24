"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler

AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


def build_lr_scheduler(optimizer, cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        cfg (CfgNode): optimization config.
    """
    lr_scheduler = cfg.models.optim_lr_scheduler
    stepsize = list(cfg.models.optim_stepsize)
    gamma = cfg.models.optim_gamma
    max_epoch = cfg.models.optim_max_epoch

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            "Unsupported scheduler: {}. Must be one of {}".format(
                lr_scheduler, AVAI_SCHEDS
            )
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = int(stepsize[-1])

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                "be an integer, but got {}".format(type(stepsize))
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                "be a list, but got {}".format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )

    if cfg.models.optim_warmup_epoch > 0:
        if not cfg.models.optim_warmup_recount:
            scheduler.last_epoch = cfg.models.optim_warmup_epoch

        if cfg.models.optim_warmup_type == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, cfg.models.optim_warmup_epoch,
                cfg.models.optim_warmup_cons_lr
            )

        elif cfg.models.optim_warmup_type == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, cfg.models.optim_warmup_epoch,
                cfg.models.optim_warmup_min_lr
            )

        else:
            raise ValueError

    return scheduler


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]
