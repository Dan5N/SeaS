from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P

from diffusers.models.transformer_2d import Transformer2DModel

class bn(nn.Module):
  """
  This class is used to create a batchnorm layer.
  """
  def __init__(
        self, 
        output_size,  
        eps: float = 1e-5,
        momentum: float = 0.1,
        cross_replica: bool = False, 
        mybn: bool = False
    ):
    super(bn, self).__init__()
    self.output_size= output_size
    self.gain = P(torch.ones(output_size), requires_grad=True)
    self.bias = P(torch.zeros(output_size), requires_grad=True)
    self.eps = eps
    self.momentum = momentum
    self.cross_replica = cross_replica
    self.mybn = mybn
    
    self.register_buffer('stored_mean', torch.zeros(output_size))
    self.register_buffer('stored_var',  torch.ones(output_size))
    
  def forward(self, x, y=None):
    if self.cross_replica or self.mybn:
      gain = self.gain.view(1,-1,1,1)
      bias = self.bias.view(1,-1,1,1)
      return self.bn(x, gain=gain, bias=bias)
    else:
      return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                          self.bias, self.training, self.momentum, self.eps)

class UNetMidBlock2DCrossAttn(nn.Module):
    """
    The UNetMidBlock2DCrossAttn is copied from the official diffusers implementation.
    We only use the Transformer2DModel for the Coarse Mask Prediction.
    """
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        do_self_attn=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        attentions = []

        for _ in range(1):
            if do_self_attn:
              attentions.append(
                  Transformer2DModel(
                      num_attention_heads,
                      in_channels // num_attention_heads,
                      in_channels=in_channels,
                      num_layers=num_layers,
                      cross_attention_dim=None,
                      norm_num_groups=resnet_groups,
                      use_linear_projection=use_linear_projection,
                      upcast_attention=upcast_attention,
                  )
              )

        self.attentions = nn.ModuleList(attentions)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        do_self_attn=False,
    ) -> torch.FloatTensor:
        
        if do_self_attn:
          for attn in self.attentions:
              hidden_states = attn(
                  hidden_states,
                  encoder_hidden_states=encoder_hidden_states,
                  cross_attention_kwargs=cross_attention_kwargs,
                  attention_mask=attention_mask,
                  encoder_attention_mask=encoder_attention_mask,
                  return_dict=False,
              )[0]

        else:
            hidden_states = self.resnets[0](hidden_states, temb)
            for resnet in self.resnets:
              hidden_states = resnet(hidden_states, temb)
  
        return hidden_states

class MRM(nn.Module):
    """
    This is the Mask Refinemen Module (MRM) used for accurate mask prediction.
    """
    def __init__(
          self, 
          in_c: int = 512,
          out_c: int = 192,
        ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            bn(out_c),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            bn(out_c),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, bias=False, padding=1),
            bn(out_c),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            bn(out_c),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            bn(out_c),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, bias=False, padding=1),
            bn(out_c),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            bn(out_c),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, bias=False, padding=1),
            bn(out_c),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            bn(out_c),
            nn.ReLU(),
        )
    def forward(self, coarse_feature, fine_feature):
        """
        This function performs Mask Refinemen Module (MRM) which combines the features from Unet (coarse_feature) and 
        features from VAE decoder (fine_feature), 
        and returns the combined features.
        """
        coarse_feature = F.interpolate(coarse_feature, scale_factor=2, mode="bilinear", align_corners=True)
        coarse_feature_skip_1 = self.conv5(coarse_feature)
        coarse_feature_skip_2 = coarse_feature
        coarse_feature = self.conv2(coarse_feature)
        coarse_feature = self.conv3(coarse_feature)
        coarse_feature = self.conv4(coarse_feature)

        coarse_feature_skip_3 = coarse_feature
        coarse_feature_skip_3 = self.conv7(coarse_feature_skip_3)
        coarse_feature_skip_3 = self.conv8(coarse_feature_skip_3)
        coarse_feature_skip_3 = self.conv9(coarse_feature_skip_3)

        coarse_feature = coarse_feature + coarse_feature_skip_1 + coarse_feature_skip_3

        fine_feature = self.conv1(fine_feature)
        fine_feature = fine_feature * coarse_feature

        out = fine_feature + coarse_feature_skip_2

        return out

class RMP(nn.Module):
    """
    The Refined Mask Prediction (RMP) use Normal Image Alignment (NIA) during the training, 
    while during the inference, we use the NIA-free model.
    """
    def __init__(
          self, 
          resolution : int = 512,
          out_dim : int = 1, 
          emb_dim : int = 1280
        ):
        super(RMP, self).__init__()
        self.resolution = resolution
        self.emb_dim = emb_dim
        self.low_feature_size = 32
        self.mid_feature_size = 128

        low_feature_channel = 128
        mid_feature_channel = 64

        self.low_feature_conv = nn.Sequential(
            nn.Conv2d(1280, low_feature_channel, kernel_size=1, bias=False),
        )

        self.mid_feature_conv = nn.Sequential(
            nn.Conv2d(640, mid_feature_channel, kernel_size=1, bias=False),
        )

        self.out_layer = nn.Sequential(
                                bn(low_feature_channel+mid_feature_channel),
                                nn.ReLU(),
                                nn.Conv2d(low_feature_channel+mid_feature_channel,
                                    2, kernel_size=3, padding=1)
                            )
        
        self.mid_attn_block = UNetMidBlock2DCrossAttn(
                num_layers=4,
                transformer_layers_per_block=1,
                in_channels=low_feature_channel+mid_feature_channel, 
                temb_channels=1280,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                cross_attention_dim=None,
                num_attention_heads=8,
                resnet_groups=32,
                dual_cross_attention=False,
                use_linear_projection=False,
                upcast_attention=False,
                do_self_attn=True,
            )
        self.MRM1 = MRM(512, 192)
        self.MRM2 = MRM(512, 192)
        self.MRM3 = MRM(256, 192)
        
    def forward(self, feature, emb, vae_features, forward_type = "train"):
        """
        This function performs RMP (Refined Mask Prediction) which combines the features from Unet (feature) and 
        features from VAE decoder (vae_features),
        and returns the predicted mask.
        """
        # The Coarse Mask Prediction
        low_feat = self.low_feature_conv(feature[1]) # 32*32 resolution
        low_feat = F.interpolate(low_feat, size=64, mode='bilinear', align_corners=False)
        mid_feat = self.mid_feature_conv(feature[2]) # 64*64 resolution
        high_feat = torch.cat([low_feat, mid_feat], dim=1)
        # The coarse features.
        # self.mid_attn_block contains four transfomer blocks.
        high_feat = self.mid_attn_block(hidden_states = high_feat, temb = emb, do_self_attn=True) # 64*64 resolution
        feat_512 = F.interpolate(high_feat, 512, mode='bilinear', align_corners=False) # 512*512 resolution
        # The VAE decoder features
        vae_low = vae_features[0]
        vae_mid = vae_features[1]
        vae_high = vae_features[2]

        # The Mask Refinemen Module (MRM)
        # for NIA during training
        if forward_type == "train":
            _ , out_normal = torch.chunk(feat_512, 2, dim=0)
            out_anomaly , _ = torch.chunk(high_feat, 2, dim=0)
                
            high_feat_128 = self.MRM1(out_anomaly,vae_low)
            high_feat_256 = self.MRM2(high_feat_128,vae_mid)
            high_feat_512 = self.MRM3(high_feat_256,vae_high)
            
            out = torch.cat((high_feat_512, out_normal), dim=0)
            out = self.out_layer(out)
            out = torch.softmax(out, dim=1)

            out_64 = high_feat
            out_64 = self.out_layer(out_64)
            out_64 = torch.softmax(out_64, dim=1)

            return out,out_64

        # without NIA during inference
        elif forward_type == "inference":
            out_anomaly = high_feat
        
            high_feat_128 = self.MRM1(out_anomaly,vae_low)
            high_feat_256 = self.MRM2(high_feat_128,vae_mid)
            high_feat_512 = self.MRM3(high_feat_256,vae_high)
            
            out = high_feat_512
            out = self.out_layer(out)
            out = torch.softmax(out, dim=1)

            return out
            