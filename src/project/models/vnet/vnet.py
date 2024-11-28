import torch
import torch.nn as nn

from project.config import PROJECT_ROOT
from project.data.luna_dataset import Luna16Dataset
from .netblocks import VNet_input_block, VNet_down_block, VNet_up_block, VNet_output_block


class VNet(nn.Module):
    def __init__(self, num_classes=2) -> None:
        super().__init__()

        self.input_block = VNet_input_block(1, 16)

        self.down_block1 = VNet_down_block(16, 32, 2)
        self.down_block2 = VNet_down_block(32, 64, 3)
        self.down_block3 = VNet_down_block(64, 128, 3)
        self.down_block4 = VNet_down_block(128, 256, 3)
        self.up_block1 = VNet_up_block(256, 256, 3)
        self.up_block2 = VNet_up_block(256, 128, 3)
        self.up_block3 = VNet_up_block(128, 64, 2)
        self.up_block4 = VNet_up_block(64, 32, 1)

        self.output_block = VNet_output_block(32, num_classes)

    def forward(self, x):
        out16 = self.input_block(x)
        out32 = self.down_block1(out16)
        out64 = self.down_block2(out32)
        out128 = self.down_block3(out64)
        out256 = self.down_block4(out128)

        out = self.up_block1(out256, out128)
        out = self.up_block2(out, out64)
        out = self.up_block3(out, out32)
        out = self.up_block4(out, out16)
        out = self.output_block(out)
        return out


# https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
class BinaryDiceLoss(nn.Module):
    """Dice loss for binary classification tasks."""
    def __init__(self, smooth=1.0, p=2, reduction='mean', from_logits=True):
        """
        Args:
            smooth (float): A smoothing constant to avoid division by zero.
            p (int): Power to apply in the denominator summation. Default is 2.
            reduction (str): Specifies the reduction type: 'mean', 'sum', or 'none'.
            from_logits (bool): If True, applies a sigmoid to `predict` to ensure it's a probability.
        """
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.from_logits = from_logits
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        """
        Args:
            predict (torch.Tensor): Predicted tensor of shape [N, *].
            target (torch.Tensor): Ground truth tensor of shape [N, *].

        Returns:
            torch.Tensor: Calculated Dice loss.
        """
        # Validate inputs
        if predict.shape != target.shape:
            raise ValueError("Shape mismatch: predict and target must have the same shape.")

        # Apply sigmoid if `from_logits` is True
        if self.from_logits:
            predict = torch.sigmoid(predict)

        print = lambda *args: None  # Comment this line to enable print statements
        print("---")
        # Flatten the tensors
        predict = predict.contiguous().view(predict.shape[0], -1)
        print(predict.min(), predict.max())
        target = target.contiguous().view(target.shape[0], -1)
        if target.max() > 1:
            target = target / 255  # Normalize to [0, 1]
        print(target.min(), target.max())

        # Compute Dice loss
        intersection = torch.sum(torch.mul(predict, target), dim=1)
        print(intersection)
        union = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)
        print(union)
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        print(dice_loss)

        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:  # 'none'
            return dice_loss


class VNetBinaryDiceLoss(nn.Module):
    """VNet outputs a tensor of shape [N, 2, H, W, D], so we need to apply the loss to the first channel."""

    def __init__(self, smooth=1.0, p=2, reduction='mean', from_logits=True):
        super().__init__()
        self.dice_loss = BinaryDiceLoss(smooth=smooth, p=p, reduction=reduction, from_logits=from_logits)

    def forward(self, predict, target):
        return self.dice_loss(predict[:, 0], target)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VNet(num_classes=1).to(device)
    luna16 = Luna16Dataset(root=PROJECT_ROOT / "data/luna16", transforms=None, train=True)


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = VNet(num_classes=2).to(device)

    # inputs = torch.randn(1, 1, 80, 128, 128) # BCDHW
    # inputs = inputs.to(device)
    # out = model(inputs)
    # print(out.shape) # torch.Size([1, 2, 64, 128, 128])
    # slices = out[0, 0, 32, :, :].detach().cpu().numpy()
    # print(slices)
    # main()
    
    loss = BinaryDiceLoss()
    input, target = torch.zeros(5, 20, 20, 20), torch.zeros(5, 20, 20, 20)
    input[2:4, 6:8, 6:8, 6:8] = 1
    target[3, 7:9, 3:7, 5:7] = 1
    print(loss(input, target))
