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
    """Dice loss of binary class"""
    def __init__(self, smooth=1, p=2, reduction='mean'):
        """
        Args:
            smooth: A float number to smooth loss, and avoid NaN error, default: 1
            p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
            reduction: Reduction method to apply, return mean over batch if 'mean',
                return sum if 'sum', return a tensor of shape [N,] if 'none'
        """
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        """
        Args:
            predict: A tensor of shape [N, *]
            target: A tensor of shape same with predict

        Returns:
            Loss tensor according to arg reduction

        Raise:
            Exception if unexpected reduction
        """
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VNet(num_classes=1).to(device)
    luna16 = Luna16Dataset(root=PROJECT_ROOT / "data/luna16", transforms=None, train=True)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VNet(num_classes=2).to(device)

    inputs = torch.randn(1, 1, 80, 128, 128) # BCDHW
    inputs = inputs.to(device)
    out = model(inputs)
    print(out.shape) # torch.Size([1, 2, 64, 128, 128])
    slices = out[0, 0, 32, :, :].detach().cpu().numpy()
    print(slices)
    # main()
