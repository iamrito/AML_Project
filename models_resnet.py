from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import os
import torch.nn as nn
from torch.autograd import Variable
import torch

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm3d(num_out_layers)
	self.num_out_layers = num_out_layers

    def forward(self, x):
	p = int(np.floor((self.kernel_size-1)/2))
	p2d = (p, p, p, p)
	size = x.size()
	y = Variable(torch.zeros(size[0], size[1], self.num_out_layers, size[3], size[4])).cuda()
	for i in range(x.size()[1]):
		y[:,i,:,:,:] = self.conv_base(F.pad(x[:,i,:,:,:], p2d))
	y2 = y.permute(0, 2, 1, 3, 4)
        y = self.normalize(y2)       #####################################
	y = y.permute(0, 2, 1, 3, 4)
        return F.elu(y, inplace=True)

 
class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)

def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)

'''
class upconv_lstm(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
	x = nn.functional.interpolate(x, scale_factor = self.scale, mode = 'bilinear', align_corners = True)
	return self.conv1(x)
'''
class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
	size = x.size()
	y = Variable(torch.zeros(size[0], size[1], size[2], self.scale*size[3], self.scale*size[4])).cuda()
	for i in range(x.size()[1]):
        	y[:,i,:,:,:] = nn.functional.interpolate(x[:,i,:,:,:], scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(y)

class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size = 3, stride = 1)
        self.normalize = nn.BatchNorm3d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
	size = x.size()
	y = Variable(torch.zeros(size[0], size[1], 2, size[3], size[4])).cuda()
	for i in range(x.size()[1]):
		y[:,i,:,:,:] = self.conv1(F.pad(x[:,i,:,:,:], p2d))
	y2 = y.permute(0, 2, 1, 3, 4)
        y = self.normalize(y2)       #####################################
	y = y.permute(0, 2, 1, 3, 4)
        return 0.3 * self.sigmoid(y)

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, stride, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels = self.input_dim,
                              out_channels = 4 * self.hidden_dim,
                              kernel_size = self.kernel_size,
			      stride = self.stride,
                              padding = self.padding,
                              bias = self.bias)
                              
	self.hconv = nn.Conv2d(in_channels = self.hidden_dim, 
			       out_channels = 4*self.hidden_dim,
			       kernel_size = self.kernel_size,
			       stride = 1,
                               padding = self.padding,
                               bias = self.bias)

	self.cconv = nn.Conv2d(in_channels = self.hidden_dim, 
			       out_channels = 3*self.hidden_dim,
			       kernel_size = (1,1),
			       stride = 1,
                               padding = 0,
                               bias = self.bias)

    def forward(self, input_tensor, cur_state):
        #print(input_tensor.size())
	input_new = self.conv(input_tensor)
	h_cur, c_cur = cur_state
	h_new = self.hconv(h_cur)
	c_new = self.cconv(c_cur)

	ip_i, ip_f, ip_o, ip_g = torch.split(input_new, self.hidden_dim, dim = 1)
	hh_i, hh_f, hh_o, hh_g = torch.split(h_new, self.hidden_dim, dim = 1)
        cc_i, cc_f, cc_o = torch.split(c_new, self.hidden_dim, dim = 1) 
        i = torch.sigmoid(ip_i + hh_i + cc_i)
        f = torch.sigmoid(ip_f + hh_f + cc_f)
        o = torch.sigmoid(ip_o + hh_o + cc_o)
        g = torch.tanh(ip_g + hh_g)

	c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
	return h_next, c_next

    def init_hidden(self, batch_size, height, width): 
        return (Variable(torch.zeros( batch_size, self.hidden_dim, height, width)).cuda(),
                Variable(torch.zeros( batch_size, self.hidden_dim, height, width)).cuda())    
    
class ConvLSTM(nn.Module):

    def __init__(self, input_channels):
        super(ConvLSTM, self).__init__()

        self.height = 128
        self.width = 256
        self.input_dim  = input_channels
        #self.hidden_dim = [32,64,64,128,128,256,256,512,512]
	self.hidden_dim = [32,64,128,128,256,256]
        self.kernel_size = [(7,7),(5,5),(3,3),(3,3),(3,3),(3,3)]
        self.stride = [2,1,2,1,2,2]
        self.num_layers = 6
        self.batch_first = False
        self.bias = True
	
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size = (self.height, self.width),
                                          input_dim = cur_input_dim,
                                          hidden_dim = self.hidden_dim[i],
                                          kernel_size = self.kernel_size[i],
                                          stride = self.stride[i],
                                          bias = self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        # decoder
        #self.upconv5 = upconv(256, 256, 3, 2)
        #self.iconv5 = conv(256+256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(128+256, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64+128+2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(32+32+2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+2, 16, 3, 1)
        self.disp1_layer = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        '''
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        '''
	#print(input_tensor.size())

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
	

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
	
        for layer_idx in range(self.num_layers):
	    #print(cur_layer_input.size())
            h, c = hidden_state[layer_idx]
            output_inner = []
	    output_out = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:,t, :, :, :],
                                                 cur_state=[h, c])
		output_inner.append(h)
		
				

            layer_output = torch.stack(output_inner, dim=1)
	    cur_layer_input = layer_output
	    
            layer_output_list.append(layer_output)
            
	'''
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
	'''
	        
        # skips
	
	skip1 = layer_output_list[0]
        skip2 = layer_output_list[2]
        skip3 = layer_output_list[4]

	x5 = layer_output_list[5]

	
        upconv4 = self.upconv4(x5)
	concat4 = torch.cat((upconv4, skip3), 2)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
	size = self.disp4.size()
	self.udisp4 = Variable(torch.zeros(size[0], size[1], 2, 2*size[3], 2*size[4])).cuda()
	for i in range(self.disp4.size()[1]):
        	self.udisp4[:,i,:,:,:] = nn.functional.interpolate(self.disp4[:,i,:,:,:], scale_factor=2, mode='bilinear', align_corners=True)

	
        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 2)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
	size = self.disp3.size()
	self.udisp3 = Variable(torch.zeros(size[0], size[1], 2, 2*size[3], 2*size[4])).cuda()
	for i in range(self.disp3.size()[1]):
        	self.udisp3[:,i,:,:,:] = nn.functional.interpolate(self.disp3[:,i,:,:,:], scale_factor=2, mode='bilinear', align_corners=True)
        
        upconv2 = self.upconv2(iconv3)
	#print('Uconv2 ka size', upconv2.size())
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 2)
	#print('Concat2 ka size', concat2.size())
        iconv2 = self.iconv2(concat2)
	#print('Iconv2 ka size', iconv2.size())
        self.disp2 = self.disp2_layer(iconv2)
	#print('Disp2 ka size', self.disp2.size())
	size = self.disp2.size()
	self.udisp2 = Variable(torch.zeros(size[0], size[1], 2, 2*size[3], 2*size[4])).cuda()
	for i in range(self.disp2.size()[1]):
        	self.udisp2[:,i,:,:,:] = nn.functional.interpolate(self.disp2[:,i,:,:,:], scale_factor=2, mode='bilinear', align_corners=True)
        
        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 2)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
        
        return self.disp1, self.disp2, self.disp3, self.disp4

    def _init_hidden(self, batch_size):
	next_height = self.height
	next_width = self.width
        init_states = []
        for i in range(self.num_layers):
	    next_height = next_height // self.stride[i]
            next_width = next_width // self.stride[i]
	    init_states.append(self.cell_list[i].init_hidden(batch_size, next_height, next_width))
        return init_states


class Resnet_18(nn.Module):
    def __init__(self, num_in_layers):
        super(Resnet_18, self).__init__()
        # encoder
        self.count = 0
        self.conv1 = conv(num_in_layers, 64, 7, 2)  # H/2  -   64D
        self.pool1 = maxpool(3)  # H/4  -   64D
        self.conv2 = resblock_basic(64, 64, 2, 2)  # H/8  -  64D
        self.conv3 = resblock_basic(64, 128, 2, 2)  # H/16 -  128D
        self.conv4 = resblock_basic(128, 256, 2, 2)  # H/32 - 256D
        self.conv5 = resblock_basic(256, 512, 2, 2)  # H/64 - 512D

        # decoder
        self.upconv6 = upconv(512, 512, 3, 2)
        self.iconv6 = conv(256+512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(128+256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(64+128, 128, 3, 1)
        self.disp4_layer = get_disp(128)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64+64 + 2, 64, 3, 1)
        self.disp3_layer = get_disp(64)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(64+32 + 2, 32, 3, 1)
        self.disp2_layer = get_disp(32)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16+2, 16, 3, 1)
        self.disp1_layer = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x_pool1 = self.pool1(x1)
        x2 = self.conv2(x_pool1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        # skips
        skip1 = x1
        skip2 = x_pool1
        skip3 = x2
        skip4 = x3
        skip5 = x4

        # decoder
        upconv6 = self.upconv6(x5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)
            
        return self.disp1, self.disp2, self.disp3, self.disp4


def class_for_name(module_name, class_name):
    m = importlib.import_module(module_name)
    return getattr(m, class_name)


