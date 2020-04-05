import torch
import torch.nn as nn
from ..registry import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm
@BACKBONES.register_module
class WslResNeXt(nn.Module):
	def __init__(self, **kwargs):
		super(WslResNeXt, self).__init__()
		self.model = torch.hub.load( 'facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
		self.norm_eval = True
		self.frozen_stages = -1
		
		
	def forward(self,x):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
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
	def init_weights(self, pretrained=None):
		pass
	
	def _freeze_stages(self):
		if self.frozen_stages >= 0:
			self.model.bn1.eval()
			for m in [self.model.conv1, self.model.bn1]:
				for param in m.parameters():
					param.requires_grad = False

		for i in range(1, self.frozen_stages + 1):
			m = getattr(self.model, 'layer{}'.format(i))
			m.eval()
			for param in m.parameters():
				param.requires_grad = False

	def train(self, mode=True):
		super(WslResNeXt, self).train(mode)
		self._freeze_stages()
		if mode and self.norm_eval:
			for m in self.model.modules():
				# trick: eval have effect on BatchNorm only
				if isinstance(m, _BatchNorm):
					m.eval()
	