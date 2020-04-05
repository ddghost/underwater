import torch
import torch.nn as nn
from ..registry import BACKBONES

@BACKBONES.register_module
class WslResNeXt(nn.Module):
	def __init__(self, **kwargs):
		self.model = torch.hub.load( 'facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
	
	def forward(self,x):
		x = self.model.conv1(x)
		x = self.model.norm1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)
		outs = []

		x = self.model.layer1(x)
		outs.append(x)
		x = self.model.layer2(x)
		outs.append(x)
		x = self.model.layer3(x)
		outs.append(x)
		x = self.model.layer4(x)
		outs.append(x)

		return tuple(outs)
