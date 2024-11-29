import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import Decoder, Encoder, InOutDecoder, ResNet


class GTN(nn.Module):
    def __init__(
        self,
        scene_layers=[3, 4, 6, 3, 2],
        depth_layers=[3, 4, 6, 3, 2],
        face_layers=[3, 4, 6, 3, 2],
        scene_inplanes=64,
        depth_inplanes=64,
        face_inplanes=64,
    ):
        super().__init__()

        # Common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.scene_backbone = ResNet(in_channels=4, layers=scene_layers, inplanes=scene_inplanes)
        self.depth_backbone = ResNet(in_channels=4, layers=depth_layers, inplanes=depth_inplanes)
        self.face_backbone = ResNet(in_channels=3, layers=face_layers, inplanes=face_inplanes)

        # Attention
        self.attn = nn.Linear(1808, 1 * 7 * 7)

        # Encoding for scene saliency
        self.scene_encoder = Encoder()

        # Encoding for depth saliency
        self.depth_encoder = Encoder()

        # Decoder for In/Out
        self.inout_decoder = InOutDecoder()

        # Decoding
        self.decoder = Decoder()

    def forward(self, images, depths, heads, heads_masks):
        heads_masks = heads_masks.unsqueeze(1)

        # Scene feature map
        scene_feat = self.scene_backbone(torch.cat((images, heads_masks), dim=1))

        # Depth feature map
        depth_feat = self.depth_backbone(torch.cat((depths, heads_masks), dim=1))

        # Face feature map
        face_feat = self.face_backbone(heads)
        # Reduce feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)

        # Reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(heads_masks))).view(-1, 784)

        # Attention layer
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        # Scene feature map * attention
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat)

        # Depth feature map * attention
        attn_applied_depth_feat = torch.mul(attn_weights, depth_feat)

        # Scene encode
        scene_encoding = self.scene_encoder(torch.cat((attn_applied_scene_feat, face_feat), 1))

        # Depth encoding
        depth_encoding = self.depth_encoder(torch.cat((attn_applied_depth_feat, face_feat), 1))

        # Fused encoding
        fused_encoding = scene_encoding + depth_encoding

        # Decoder for heatmap
        x_heatmap = self.decoder(fused_encoding)

        # Decoder for in/out
        x_inout = self.inout_decoder(
            torch.cat((attn_applied_scene_feat + attn_applied_depth_feat, face_feat), 1)
        )

        return x_heatmap, x_inout, attn_weights
