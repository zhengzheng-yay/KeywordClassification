import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable

class DSDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                    stride=1, bias=True):
        super(DSDilatedConv1d, self).__init__()
        self.receptive_fields = dilation*(kernel_size-1)
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                    padding=0, dilation=dilation, stride=stride,
                    groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                   padding=0, dilation=1, bias=bias)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.pointwise(outputs)
        return outputs

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                     nn.Linear(channel, channel // reduction, bias=False),
                     nn.ReLU(inplace=True),
                     nn.Linear(channel//reduction, channel, bias=False),
                     nn.Sigmoid()
                    )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, res_channels, kernel_size, dilation, dropout):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.receptive_fields = dilation*(kernel_size-1)
        self.half_receptive_fields = self.receptive_fields//2
        self.conv1 = DSDilatedConv1d(in_channels = in_channels,
                                     out_channels = res_channels,
                                     kernel_size = kernel_size,
                                     dilation = dilation)
        self.bn1 = nn.BatchNorm1d(res_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels = res_channels,
                               out_channels = res_channels,
                               kernel_size = 1)
        self.bn2 = nn.BatchNorm1d(res_channels)
        self.relu2 = nn.ReLU()

    def forward(self, inputs):
        outputs = self.relu1(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        inputs = inputs[:, :, self.receptive_fields:]
        if self.in_channels == self.res_channels:
            res_out = self.relu2(outputs + inputs)
        else:
            res_out = self.relu2(outputs)
        return res_out

class ResidualStack(nn.Module):
    def __init__(self, in_channels, layer_size, stack_size, 
                 res_channels, kernel_size, dropout):
        super(ResidualStack, self).__init__()
        self.in_channels = in_channels
        self.layer_size = layer_size
        self.stack_size = stack_size
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.res_blocks = self.stack_res_blocks()
        self.receptive_fields = self.calculate_receptive_fields()
        print("Receptive Fields: %d"%self.receptive_fields)

    def calculate_receptive_fields(self):
        receptive_fields = 0
        for block in self.res_blocks:
            receptive_fields += block.receptive_fields
        return receptive_fields

    def build_dilations(self):
        dilations = []
        for s in range(0, self.stack_size):
            for l in range(0, self.layer_size):
                dilations.append(2**l)
        return dilations

    def stack_res_blocks(self):
        dilations = self.build_dilations()
        res_blocks = nn.ModuleList()
        res_blocks.append(ResidualBlock(self.in_channels, self.res_channels, 
                            self.kernel_size, dilations[0], self.dropout))
        for dilation in dilations[1:]:
            res_blocks.append(ResidualBlock(self.res_channels, self.res_channels, 
                                self.kernel_size, dilation, self.dropout))
        return res_blocks
        

    def forward(self, inputs):
        outputs = inputs
        for i in range(len(self.res_blocks)):
            outputs = self.res_blocks[i](outputs)
        return outputs

class MultiScaleCausalTCN(nn.Module):
    def __init__(self, layer_size, stack_size, in_channels, res_channels, 
                        out_channels, kernel_size, dropout):
        super(MultiScaleCausalTCN, self).__init__()
        self.kernel_size = kernel_size
        self.preprocessor = DSDilatedConv1d(in_channels, res_channels, kernel_size,
                                          dilation=1, stride=1)
        self.relu = nn.ReLU(res_channels)
        self.blocks = nn.ModuleList()
        self.receptive_fields = self.preprocessor.receptive_fields
        for i in range(stack_size):
            self.blocks.append(ResidualStack(res_channels, layer_size, 1, 
                                    res_channels, kernel_size, dropout))
            self.receptive_fields += self.blocks[-1].receptive_fields
        self.half_receptive_fields = self.receptive_fields // 2 
        #self.clss = nn.ModuleList()
        #for i in range(stack_size):
        #    self.clss.append(nn.Sequential(nn.Linear(res_channels, out_channels)))

    def normalize_length(self, skip_connections):
        output_size = skip_connections[-1].shape[-1]
        normalized_outputs = []
        for x in skip_connections:
            remove_length = (x.shape[-1]-output_size)
            if remove_length != 0:
                normalized_outputs.append(x[:, :, remove_length:])
            else:
                normalized_outputs.append(x)
        return normalized_outputs

    def forward(self, x, length):
        outputs = F.pad(x, (0, 0, self.receptive_fields, 0, 0, 0), 'constant')
        outputs = outputs.transpose(1, 2)
        outputs_list = []
        outputs = self.relu(self.preprocessor(outputs))
        #outputs_list.append(outputs)
        for i in range(len(self.blocks)):
            outputs = self.blocks[i](outputs)
            outputs_list.append(outputs)
        outputs_list = self.normalize_length(outputs_list)
        #outputs = []
        #for i in range(len(outputs_list)):
        #    outputs.append(self.clss[i](torch.mean(outputs_list[i], dim=2)))

        outputs = sum(outputs_list)
        outputs = outputs.transpose(1, 2)
        return outputs


