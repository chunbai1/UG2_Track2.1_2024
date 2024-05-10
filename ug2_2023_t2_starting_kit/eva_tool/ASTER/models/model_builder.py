from PIL import Image
import numpy as np
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from . import create
from .attention_recognition_head import AttentionRecognitionHead
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead

tps_inputsize = [32,64]
with_lstm = True
n_group = 1
tps_outputsize = [32,100]
num_control_points = 20
tps_margins = [0.05, 0.05]
stn_activation = 'none'
beam_width = 5



class ModelBuilder(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes, sDim, attDim, max_len_labels, eos, STN_ON=False):
    super(ModelBuilder, self).__init__()

    self.arch = arch
    self.rec_num_classes = rec_num_classes
    self.sDim = sDim
    self.attDim = attDim
    self.max_len_labels = max_len_labels
    self.eos = eos
    self.STN_ON = STN_ON
    self.tps_inputsize = tps_inputsize

    self.encoder = create(self.arch,
                      with_lstm=with_lstm,
                      n_group=n_group)
    encoder_out_planes = self.encoder.out_planes

    self.decoder = AttentionRecognitionHead(
                      num_classes=rec_num_classes,
                      in_planes=encoder_out_planes,
                      sDim=sDim,
                      attDim=attDim,
                      max_len_labels=max_len_labels)

    if self.STN_ON:
      self.tps = TPSSpatialTransformer(
        output_image_size=tuple(tps_outputsize),
        num_control_points=num_control_points,
        margins=tuple(tps_margins))
      self.stn_head = STNHead(
        in_planes=3,
        num_ctrlpoints=num_control_points,
        activation=stn_activation)

  def forward(self, input_dict):
    return_dict = {}
    return_dict['losses'] = {}
    return_dict['output'] = {}

    x, rec_targets, rec_lengths = input_dict['images'], \
                                  input_dict['rec_targets'], \
                                  input_dict['rec_lengths']

    # rectification
    if self.STN_ON:
      # input images are downsampled before being fed into stn_head.
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_img_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
      if not self.training:
        # save for visualization
        return_dict['output']['ctrl_points'] = ctrl_points
        return_dict['output']['rectified_images'] = x

    encoder_feats = self.encoder(x)
    encoder_feats = encoder_feats.contiguous()

    rec_pred, rec_pred_scores = self.decoder.beam_search(encoder_feats, beam_width, self.eos)
    rec_pred_ = self.decoder([encoder_feats, rec_targets, rec_lengths])
    return_dict['losses']['loss_rec'] = None
    return_dict['output']['pred_rec'] = rec_pred
    return_dict['output']['pred_rec_score'] = rec_pred_scores

    return return_dict
