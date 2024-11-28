import torch
from torch.optim.lr_scheduler import LRScheduler

class LossBasedLRScheduler(LRScheduler):
    """
    Custom learning rate scheduler that adjusts learning rate based on the loss value.
    """
    def __init__(self, optimizer, thresholds, lr_values):
        """
        Args:
            optimizer (torch.optim.Optimizer): Optimizer to adjust the learning rate.
            thresholds (list of float): List of loss thresholds for defining ranges.
            lr_values (list of float): List of learning rate ranges corresponding to thresholds.
                                       Should have len(thresholds) + 1 elements.
        """
        self.optimizer = optimizer
        self.thresholds = thresholds
        self.lr_values = lr_values

        if len(lr_values) != len(thresholds) + 1:
            raise ValueError("lr_values must have one more element than thresholds.")

    def step(self, current_loss):
        """
        Update the learning rate based on the current loss.
        Args:
            current_loss (float): Current loss value.
        """
        for i, threshold in enumerate(self.thresholds):
            if current_loss > threshold:
                lr = self._interpolate(current_loss, threshold, self.thresholds[i-1] if i > 0 else None,
                                       self.lr_values[i], self.lr_values[i-1] if i > 0 else None)
                break
        else:
            lr = self._interpolate(current_loss, self.thresholds[-1], None, self.lr_values[-1], self.lr_values[-2])

        self._set_lr(lr)

    def _interpolate(self, loss, high, low, lr_high, lr_low):
        """Linearly interpolate the learning rate."""
        if low is None or lr_low is None:  # Loss is higher than the highest threshold
            return lr_high
        scale = (loss - low) / (high - low)
        return lr_low + scale * (lr_high - lr_low)

    def _set_lr(self, lr):
        """Set learning rate in the optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
