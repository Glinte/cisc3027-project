import importlib

import torch
import torch.nn as nn
from project.config import PROJECT_ROOT
from project.data.luna_dataset import Luna16Dataset
from project.models.vnet.vnet import BinaryDiceLoss
from torch import optim
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

from project.models.attention_unet.BuildingBlocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=16, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=6)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        pool_fea = self.avg_pool(encoders_features[0]).squeeze(0).squeeze(1).squeeze(1).squeeze(1)
        encoders_features = encoders_features[1:]
        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x, pool_fea


if __name__ == '__main__':
    class PadDepthTransform(nn.Module):
        """Pad the depth dimension of the input tensor to be divisible by 16."""

        def forward(self, img, mask) -> tuple[torch.Tensor, torch.Tensor]:
            # Check if the number of depth dimensions is odd
            if img.shape[1] % 16 != 0:
                # Create a zero-filled padding with the same height and width
                padding = torch.zeros(1, 16 - img.shape[1] % 16, *img.shape[2:], device=img.device, dtype=img.dtype)
                # Concatenate the padding to the tensor
                img = torch.cat([img, padding], dim=1)
                mask = torch.cat([mask, padding], dim=1)
            return tv_tensors.Image(img), tv_tensors.Mask(mask)


    class Normalize3D(nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std)

        def forward(self, img, mask) -> tuple[torch.Tensor, torch.Tensor]:
            img = (img - self.mean) / self.std
            return img, mask


    class Resize3d(nn.Module):
        def __init__(self, size: tuple[int, int, int]):
            super().__init__()
            self.size = size

        def forward(self, img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            img = nn.functional.interpolate(img.unsqueeze(0).float(), size=self.size, mode="trilinear",
                                            align_corners=False).squeeze(0).long()
            mask = nn.functional.interpolate(mask.unsqueeze(0).float(), size=self.size, mode="nearest").squeeze(
                0).long()
            return img, mask


    transforms = v2.Compose([
        Resize3d(size=(128, 128, 128)),
        v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}),
        Normalize3D(mean=-790.1, std=889.6),
    ])

    luna16 = Luna16Dataset(root=PROJECT_ROOT / "data/luna16", transforms=transforms, train=True)
    luna_loader = DataLoader(luna16, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=True, f_maps=16, layer_order='crg', num_groups=8).to(device)
    criterion = BinaryDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    print(sum(p.numel() for p in model.parameters()))

    out = model(luna16[0][0].to(device).unsqueeze(0))
    print("Model output shape:", out[0].shape, out[1].shape)

    run = None
    try:
        for epoch in range(10):
            for i, (data, target) in enumerate(luna_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                target = target / 255  # BCELoss expects float targets
                print(output[:, 0:1, :, :, :].shape)
                loss = criterion(output[:, 0:1, :, :, :], target)
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
                if run:
                    run.log({"epoch": epoch, "batch": i, "train/loss": loss.item()})
            scheduler.step()
    except KeyboardInterrupt as e:
        pass

    if run:
        run.finish()

    torch.save(model.state_dict(), "unet3d_attention.pth")