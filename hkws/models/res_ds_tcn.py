import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attnW = nn.Linear(input_dim, input_dim, bias=True)
        nn.init.xavier_uniform_(self.attnW.weight)
        nn.init.constant_(self.attnW.bias, 0.0)
        self.attnV = nn.Linear(input_dim, 1, bias=False)
        nn.init.constant_(self.attnV.weight, 0.0)
        self.tanh = nn.Tanh()

    def forward(self, key, value):
        Wh = self.attnW(key)
        Wh = self.tanh(Wh)
        atten_e = self.attnV(Wh)
        atten_w = F.softmax(atten_e, dim=1)
        atten_w = atten_w.transpose(1, 2).contiguous() # batch*1*99
        output = torch.bmm(atten_w, value) # value: batch*99*64 output: batch*1*64  
        return output

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

class DSDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                       stride=1, bias=True):
        super(DSDilatedConv1d, self).__init__()
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

class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, kernel_size, dilation, dropout):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.receptive_fields = dilation*(kernel_size-1)
        self.conv1 = DSDilatedConv1d(in_channels=res_channels,
                                         out_channels=res_channels,
                                         kernel_size=kernel_size,
                                         dilation=dilation)
        self.bn1 = nn.BatchNorm1d(res_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=res_channels,
                               out_channels=res_channels,               
                               kernel_size=1)
        self.bn2 = nn.BatchNorm1d(res_channels)
        self.relu2 = nn.ReLU()  
        self.se = SELayer(res_channels, reduction=4)
        #self.conv3 = nn.Conv1d(in_channels=res_channels,
        #                      out_channels=res_channels,
        #                      kernel_size=1)
        #self.bn3 = nn.BatchNorm1d(res_channels)
        #self.relu3 = nn.ReLU()

    def forward(self, inputs):
        outputs = self.relu1(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        half_receptive_fields = self.receptive_fields//2
        #short_cut = self.relu3(self.bn3(self.conv3(inputs)))
        inputs = inputs[:, :, half_receptive_fields:-half_receptive_fields]
        output = self.se(outputs)
        res_out = self.relu2(outputs + inputs)
        return res_out

class ResidualStack(nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels,
                 kernel_size, dropout):
        super(ResidualStack, self).__init__()
        self.layer_size = layer_size
        self.stack_size = stack_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.res_blocks = self.stack_res_block(res_channels, skip_channels, 
                                    kernel_size, dropout) 

    def _residual_block(self, res_channels, skip_channels, kernel_size, 
                        dilation, dropout):
        return ResidualBlock(res_channels, skip_channels, kernel_size, 
                             dilation, dropout)

    def build_dilations(self):
        dilations = []
        for s in range(0, self.stack_size):
            for l in range(0, self.layer_size):
                dilations.append(2 ** l)
        return dilations

    def stack_res_block(self, res_channels, skip_channels, kernel_size, dropout):
        res_blocks = nn.ModuleList()
        dilations = self.build_dilations()
        for dilation in dilations:
            block = self._residual_block(res_channels, skip_channels, 
                                         kernel_size, dilation, dropout)
            res_blocks.append(block)

        return res_blocks

    def forward(self, x):
        output = x
        for i in range(len(self.res_blocks)):
            output = self.res_blocks[i](output)
        return output

class ResDSTCN(nn.Module):
    def __init__(self, layer_size, stack_size, in_channels,
                 res_channels, kernel_size, dropout):
        super(ResDSTCN, self).__init__()
        self.kernel_size = kernel_size
        self.preprocess = DSDilatedConv1d(in_channels, res_channels, kernel_size,            
                                          dilation=1, stride=1)
        self.relu = nn.ReLU(res_channels)

        self.receptive_fields = kernel_size - 1 + self.calc_receptive_fields(layer_size, stack_size, kernel_size)
        #self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size, kernel_size)
        print("Receptive Fields: %d"%self.receptive_fields)

        self.res_stacks = ResidualStack(layer_size, stack_size, res_channels,
                                        res_channels, kernel_size=kernel_size,
                                        dropout=dropout)
        #self.att = Attention(res_channels)

    def calc_receptive_fields(self, layer_size, stack_size, kernel_size):
        layers = [ 2**i for i in range(0, layer_size) ] * stack_size * (kernel_size-1)
        return int(sum(layers))

    def check_input_size(self, x, output_size):
        if output_size < 1:
            print("Input size error: input size: %d, receptive_fields: %d, output: %d"%(x.size(2), self.receptive_fields, output_size))
    
    #def normalize_length(self, skip_connections):
    #    output_size = skip_connections[-1].shape[-1]
    #    new_skip_connections = []
    #    for x in skip_connections:
    #        new_skip_connections.append(x[:,:,-output_size:])
    #    return new_skip_connections

    def forward(self, x , length):
        half_receptive_fields = self.receptive_fields//2
        output = F.pad(x, (0, 0, half_receptive_fields, half_receptive_fields, 0, 0), 'constant')
        output = output.transpose(1, 2)
        output = self.relu(self.preprocess(output))
        output = self.res_stacks(output)
        output = output.transpose(1, 2).contiguous()
        #output = self.att(output, output)
        return output
