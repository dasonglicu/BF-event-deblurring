import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import spconv.pytorch as spconv
from torchvision.ops import deform_conv2d
from torch.nn.modules.utils import _pair, _single
import numpy as np
# from mmcv.cnn import constant_init
# from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError
    m = spconv.SparseSequential(
        conv,
        # norm_fn(out_channels),
        nn.PReLU(),
    )
    return m

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        # self.bn1 = norm_fn(planes)
        self.relu = nn.PReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        # self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        # out.features = out.features
        # out.features = self.relu(out.features)
        out = out.replace_feature(self.relu(out.features))
        out = self.conv2(out)
        # out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity)
        # out = out.replace_feature(self.relu(out.features))

        return out
class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_in = nn.Conv2d(3*dim, 2*dim, kernel_size=1, bias=bias)
        # self.conv = nn.Conv2d(2*dim, 2*dim, kernel_size=3, bias=bias, groups=2*dim, padding=1)
        # self.conv = nn.Conv2d(2*dim, 2*dim, kernel_size=1, bias=bias, groups=2*dim, padding=1)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sg = SimpleGate()
        # self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        in_features = dim * 3
        hidden_features = dim * 4
        out_features = dim
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        # self.drop = nn.Dropout(drop)
        self.channel_interaction1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            # nn.BatchNorm2d(dim // 8),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.channel_interaction2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            # nn.BatchNorm2d(dim // 8),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )

    def forward(self, input_):
        xx = self.project_in(input_)
        xx = self.sg(xx)
        xx = xx * self.sca(xx)
        xx = self.project_out(xx)
        
        x = self.fc1(input_)
        x = self.act(x)
        x = self.fc2(x)
        channel_map1 = self.channel_interaction1(xx)
        channel_map2 = self.channel_interaction2(x)
        xx = xx * torch.sigmoid(channel_map2)
        x = x * torch.sigmoid(channel_map1)
        # out = xx # * self.beta
        
        out = self.project_out2(torch.cat((x, xx), dim=1))
        return out
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class EventImage_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EventImage_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image = LayerNorm2d(3*dim)
        # self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = LayerNorm2d(dim) # nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        # assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        fused = image + self.attn(self.norm1_image(torch.cat((image, event), dim=1))) # b, c, h, w
        # mlp
        # fused = to_3d(fused) # b, h*w, c
        # fused = fused + self.ffn(self.norm2(fused))
        # fused = to_4d(fused, h, w)
        return fused
class ResBlock(nn.Module):
    def __init__(self, in_size, out_size): # cat
        super(ResBlock, self).__init__()
        relu_slope = 0.2
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
    def forward(self, x):
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)
        return out
    
class UNetConvBlock0(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None): # cat
        super(UNetConvBlock0, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, in_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads
        self.deform = DeformableAlignment(in_size*2, in_size, 3, padding=1, deform_groups=16, other_channels = in_size)
        self.in_channels = in_size
        self.resblock1 = ResBlock(in_size, in_size)
        # self.resblock2 = ResBlock(in_size, in_size)
        self.resblock3 = ResBlock(in_size*2, in_size)
        #self.attn = NeighborhoodAttention(dim, kernel_size=7, dilation=dilation,num_heads=num_heads,
        #    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,)
        self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(in_size, in_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(in_size, in_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(in_size, in_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(in_size, in_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(in_size, out_size, bias=False)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(in_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
            self.image_event_transformer2 = EventImage_ChannelAttentionTransformerBlock(in_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        

    def forward(self, x, flow, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.resblock1(x)
        # out1 = self.resblock2(out)
        out1 = self.image_event_transformer(out, event_filter)
        # out1 = self.resblock2(out1)
        # out1 = self.image_event_transformer2(out1, cond)
        out2 = self.deform(out, flow, out1)
        #out2 = self.resblock2(out2)
        # out2 = self.image_event_transformer2(out2, cond)
        # out2 = self.deform(out, flow)
        out = self.resblock3(torch.cat((out1, out2), dim=1))
        # out = self.deform(out, event_flow)
        
        if enc is not None and dec is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask(enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(dec)
            out = out + out_enc + out_dec        
        #if event_filter is not None and merge_before_downsample:
        #    # b, c, h, w = out.shape
        #    # print(out.shape, event_filter.shape)
        #    out = self.image_event_transformer(out, event_filter)
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_down = self.image_event_transformer(out_down, event_filter) 

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)
def add_noise_to_voxel(voxel, noise_std=0.2, noise_fraction=0.1):
    noise = noise_std * torch.randn_like(voxel)
    # 10 / 255.0 = 0.25
    #if noise_fraction < 1.0:
    #    mask = torch.rand_like(voxel) >= noise_fraction
    #    noise.masked_fill_(mask, 0)
    return torch.round(voxel + noise)
class BFNet(nn.Module):
    def __init__(self, in_chn=3, ev_chn=6, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super(BFNet, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.deform_path = nn.ModuleList()
        self.conv_01 = nn.Sequential(nn.Conv2d(in_chn, wf, 3, 1, 1), nn.LeakyReLU(relu_slope), ResBlock(wf, wf))
        # self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        # event
        self.down_path_ev = nn.ModuleList()
        self.down_path_ev2 = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(16, wf, 3, 1, 1)
        self.conv_ev2 = nn.Sequential(nn.Conv2d(16+3, wf, 3, 2, 1), nn.LeakyReLU(relu_slope), nn.Conv2d(wf, wf, 3, 2, 1), nn.LeakyReLU(relu_slope))
        self.conv_ev_out = nn.Conv2d(wf, 16*2, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False 

            self.down_path_1.append(UNetConvBlock0(wf*(2**(i)), wf*(2**(i))*2, downsample, relu_slope, num_heads=self.num_heads[i]))
            # self.down_path_2.append(UNetConvBlock0(wf*(2**(i)), wf*(2**(i))*2, downsample, relu_slope, use_emgc=downsample))
            # ev encoder
            if i < self.depth:
                self.down_path_ev.append(UNetEVConvBlock(prev_channels, (2**i) * wf, downsample , relu_slope))
                self.down_path_ev2.append(UNetEVConvBlock(prev_channels, (2**i) * wf, downsample , relu_slope))
            # self.deform_path.append(DeformableAlignment(wf*(2**i), wf*(2**i), 3, padding=1, deform_groups=16, other_channels = (2**i) * wf))
            if i == self.depth - 1:
                self.down_path_1d = UNetConvBlock(wf*(2**(i)), wf*(2**(i))*2, downsample, relu_slope, use_emgc=downsample)
                # self.down_path_2d = UNetConvBlock(wf*(2**(i)), wf*(2**(i))*2, downsample, relu_slope, use_emgc=downsample)

            prev_channels = (2**i) * wf
        self.down = nn.AvgPool2d(2)
        # self.up = nn.Upsample(scale_factor=2)
            
        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.up_path_ev = nn.ModuleList()
        self.up_path_ev1 = nn.ModuleList()
        self.up_path_conv = UNetConvBlock(prev_channels, prev_channels, downsample, relu_slope, use_emgc=False)
        self.up_path1_conv = UNetConvBlock(prev_channels, prev_channels, downsample, relu_slope, use_emgc=False)
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_ev = nn.ModuleList()
        self.skip_conv_ev1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            # self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            # self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_1.append(ResBlock((2**i)*wf, (2**i)*wf))
            # self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_ev.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_ev1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.up_path_ev.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_ev1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)
        self.down = nn.AvgPool2d(2)
        self.flow_up = nn.Upsample(scale_factor=4)

        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)
    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = x.replace_feature(torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = x.replace_feature(x_m.features + x.features)
        x = conv_inv(x)
        return x
    def reblurring(self, sharp, optical_flow):
        b, c, h, w = optical_flow.shape
        # print(sharp.shape, optical_flow.shape)
        # sharp_resize = sharp.view()
        # exit(0)
        sharp = sharp.unsqueeze(1)
        optical_resize = optical_flow.view(b*(c//2), 2, h, w).permute(0,2,3,1)
        sharp_resize = sharp.repeat(1,c//2,1,1,1).view(b*c//2,3,h,w)
        # print(sharp_resize.shape, optical_resize.shape)
        # exit(0)
        cond = flow_warp(sharp_resize, optical_resize)
        # print(cond.shape)
        # exit(0)
        cond = cond.view(b,c//2,3,h,w)
        # print(cond.shape)
        cond = torch.cat((cond, sharp),dim=1)
        # print(cond.shape)
        # exit(0)
        cond = torch.mean(cond, dim=1)
        # print(cond.shape)
        # exit(0)
        return cond
        
        
    def forward(self, x, sharp_input, voxel_features, voxel_coordinates, batch_size, spatial_shapes, mask=None):
        # print(voxel_features.shape, voxel_coordinates.shape)
        # sharp_input =x 
        # batch_size = 2 # batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coordinates.int(),
            spatial_shape=spatial_shapes, # [45809409, 256, 256]
            batch_size=batch_size
        )
        out = input_sp_tensor.dense(channels_first=True)
        # print(out.shape, out.max(), out.min())
        # exit(0)
        b,c1,c2,h,w = out.shape
        x_up1_dense = out.view(b,c1*c2,h,w).permute(0,1,3,2)
        x_up1_dense = add_noise_to_voxel(x_up1_dense)
        image = x

        # ev = [x_up1_dense, x_up3_dense, x_up4_dense]
        #EVencoder
        # ev.append()
        ev = []
        ev_decs = []
        e2 = self.conv_ev2(torch.cat((x_up1_dense.detach(), image), dim=1))
        for i, down in enumerate(self.down_path_ev2):
            if i < self.depth-1:
                e2, e2_up = down(e2, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev_decs.append(e2_up)
                else:
                    ev_decs.append(e2)
            else:
                e2 = down(e2, self.fuse_before_downsample)
                ev_decs.append(e2)
                e2 = self.up_path_conv(e2)
        for i, up in enumerate(self.up_path_ev):
        #    # print(e1.shape, ev[-i-1].shape, ev[-i-2].shape)
            e2 = up(e2, self.skip_conv_ev[i](ev_decs[-i-2]))
        event_flow = self.conv_ev_out(e2)
        event_flow = self.flow_up(event_flow)
        event_flow_d = - event_flow.detach()
        event_flow_list = [event_flow_d, 0.5 * self.down(event_flow_d), 0.25 * self.down(self.down(event_flow_d))]
        reblurring_out = self.reblurring(sharp_input, event_flow)
        #EVencoder
        
        # ev = []
        ev1_encs = []
        ev1_decs = []
        e1 = self.conv_ev1(x_up1_dense.detach()) 
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth-1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev1_encs.append(e1_up)
                else:
                    ev1_encs.append(e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev1_encs.append(e1)
                e1 = self.up_path1_conv(e1)
                ev1_decs.append(e1)
        for i, up in enumerate(self.up_path_ev1):
        #    # print(e1.shape, ev[-i-1].shape, ev[-i-2].shape)
            e1 = up(e1, self.skip_conv_ev1[i](ev1_encs[-i-2]))
            ev1_decs.append(e1)
            # print(i, e1.shape)
        # print(ev1_decs[0].shape, ev1_decs[1].shape, ev1_decs[2].shape)
        # exit(0)

        x1 = self.conv_01(image)
        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            # x1 = self.deform_path[i](x1, event_flow_list[i])
            # print(i, x1.shape, ev1_decs[2-i].shape)
            if (i+1) < self.depth:
                x1, x1_up = down(x1, event_flow_list[i], event_filter=torch.cat((ev1_decs[2-i], ev1_encs[i]), dim=1), merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))
            
            else:
                x1 = down(x1, event_flow_list[i], event_filter=torch.cat((ev1_decs[2-i], ev1_encs[i]), dim=1), merge_before_downsample=self.fuse_before_downsample)
                x1 = self.down_path_1d(x1)                                       


        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            # x1 = self.deform_path[-i-2](x1, ev_decs[i+1])
            decs.append(x1)
        out_1 = self.last(x1) + image
        return [out_1], reblurring_out

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2
        Returns:
        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = x.replace_feature(features.view(n, out_channels, -1).sum(dim=2))
        return x

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None): # cat
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, in_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads
        self.resblock1 = ResBlock(in_size, in_size)
        # self.resblock2 = ResBlock(in_size, in_size)
        # self.conv_1 = nn.Conv2d(in_size, in_size, kernel_size=3, padding=1, bias=True)
        # self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        # self.conv_2 = nn.Conv2d(in_size, in_size, kernel_size=3, padding=1, bias=True)
        # self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False) 
        #self.attn = NeighborhoodAttention(dim, kernel_size=7, dilation=dilation,num_heads=num_heads,
        #    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,)

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(in_size, in_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(in_size, in_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(in_size, in_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(in_size, in_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(in_size, out_size, bias=False)

        if self.num_heads is not None:
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(in_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        # print(x.shape, event_filter.shape, "U_Net")
        out = self.resblock1(x)
        # out = self.resblock2(out)

        if enc is not None and dec is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask(enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(dec)
            out = out + out_enc + out_dec        
            
        if event_filter is not None and merge_before_downsample:
            # b, c, h, w = out.shape
            # print(out.shape, event_filter.shape)
            out = self.image_event_transformer(out, event_filter) 
             
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_down = self.image_event_transformer(out_down, event_filter) 

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)
def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
class ModulatedDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0,
                 dilation=1,groups=1,deform_groups=1, bias=True):
        super(ModulatedDeformConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
class DeformableAlignment(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = 5 # kwargs.pop('max_residue_magnitude', 10)
        self.other_channels = kwargs.pop('other_channels', 10)
        # print(self.other_channels)
        super(DeformableAlignment, self).__init__(*args, **kwargs)
        relu_scope = 0.2
        self.conv_offset = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Conv2d((self.in_channels // self.deform_groups +2) * self.deform_groups, self.out_channels//2, 3, 1, 1),
            nn.LeakyReLU(relu_scope, inplace=True),
            # nn.AvgPool2d(2),
            nn.Conv2d(self.out_channels//2, self.out_channels//2, 3, 1, 1),
            nn.LeakyReLU(relu_scope, inplace=True),
            nn.Conv2d(self.out_channels//2, self.out_channels//2, 3, 1, 1),
            nn.LeakyReLU(relu_scope, inplace=True),
            nn.Conv2d(self.out_channels//2, 27 * self.deform_groups, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        self.init_offset()
    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)
    def forward(self, x, flow, add_feat):
        b, c, h, w = flow.shape
        flow_resize = flow.view(b*(c//2), 2, h, w)
        flow_resize3 = flow_resize.flip(1).view(b, c//2, 2, h,w)
        flow_resize3 = flow_resize3.repeat(1,1,9,1,1).view(b,-1,h,w)
        extra_feat = torch.cat([x, flow_resize.view(b,-1,h,w), add_feat], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset = offset + flow_resize3
        mask = torch.sigmoid(mask)
        # print(offset.shape, add_feat.shape, mask.shape)
        # exit()
        # print(x.shape, add_feat.shape)
        # exit(0)
        x_resize = x.view(b, self.deform_groups, self.in_channels//2//self.deform_groups, h, w)
        add_resize = add_feat.view(b, self.deform_groups, self.in_channels//2//self.deform_groups, h, w)
        x_input = torch.cat((x_resize, add_resize), dim=2).view(b, self.in_channels,h,w)
        deform_out = deform_conv2d(x_input, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask=mask)
        # print(deform_out.shape)
        return deform_out
        # return deform_conv2d(x_input, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask=mask)

class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0) 
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=False):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)
             
        if self.downsample:

            out_down = self.downsample(out)
            
            if not merge_before_downsample: 
            
                out_down = self.conv_before_merge(out_down)
            else : 
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(out_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        # print(x.shape)
        up = self.up(x)
        # print("up", up.shape, "bridge", bridge.shape)
        # out = torch.cat([up, bridge], 1)
        out = up + bridge
        out = self.conv_block(out)
        return out

def hash_3d(ts, ys, xs):
    return xs + 256*ys + 256*256* ts
# a * 39 + b * 255 + c
# a * 100 + b * 10 + c
# 111 =  100 
# 11 
def dehash_3d(input_s):
    ts, temp = input_s // (256*256), input_s % (256*256)
    ys, xs = (temp) // 256, temp % 256
    return ts, ys, xs
if __name__ == "__main__":
    img = torch.FloatTensor(np.load("/workspace/1208/EFNet/image.npy")).cuda().unsqueeze(0)
    # print(img.shape)
    xs = torch.FloatTensor(np.load("/workspace/1208/EFNet/xs.npy")).cuda()
    ys = torch.FloatTensor(np.load("/workspace/1208/EFNet/ys.npy")).cuda()
    ts = torch.FloatTensor(np.load("/workspace/1208/EFNet/ts.npy")).cuda()
    ps = torch.FloatTensor(np.load("/workspace/1208/EFNet/ps.npy")).cuda()
    # print(xs.shape, ys.shape, ts.shape, ps.shape)
    # print("xs:", xs.max(), xs.min())
    # print("ys:", ys.max(), ys.min())
    ts = ts - ts.min()
    dt = ts.max() - ts.min()
    ts = ts / dt * 16
    # print("ts:", ts.max(), ts.min(), ts.max() - ts.min())
    # ts = ts - ts.min()
    # exit(0)
    xs = xs.int()
    ys = ys.int()
    ts = ts.int().clamp(None, 15)
    # print(ts.max(), ts.min(), ys.max(), ys.min(), xs.max(), xs.min())
    print(xs.shape, ys.shape, ts.shape)
    pos_xs = xs[ps > 0]
    pos_ys = ys[ps > 0]
    pos_ts = ts[ps > 0]
    neg_xs = xs[ps < 0]
    neg_ys = ys[ps < 0]
    neg_ts = ts[ps < 0]
    #print(pos_xs.shape)
    #exit(0)
    
    pos_hashed_index = hash_3d(pos_ts, pos_ys, pos_xs)
    print(pos_hashed_index.shape)
    idx_sort = torch.argsort(pos_hashed_index)
    sorted_records_array = pos_hashed_index[idx_sort]
    pos_vals, pos_count = torch.unique(sorted_records_array, return_counts=True)
    print(pos_vals.shape, pos_count.shape)
    # print(count.max(), count.min())
    
    neg_hashed_index = hash_3d(neg_ts, neg_ys, neg_xs)
    idx_sort = torch.argsort(neg_hashed_index)
    sorted_records_array = neg_hashed_index[idx_sort]
    neg_vals, neg_count = torch.unique(sorted_records_array, return_counts=True)
    neg_count = -neg_count
    print(neg_vals.shape, neg_count.shape, pos_count.max(), neg_count.max())
    
    pos_vals_np = pos_vals.cpu().data.numpy()
    neg_vals_np = neg_vals.cpu().data.numpy()
    
    xy, x_ind, y_ind = np.intersect1d(pos_vals_np, neg_vals_np, return_indices=True)
    # print(len(x_ind), len(y_ind))
    if len(x_ind) > 0:
        print(pos_count[x_ind])
        print(neg_count[x_ind])
        # exit(0)
        pos_count[x_ind] += neg_count[y_ind]
        # exit(0)
        pos_count_np = pos_count.cpu().data.numpy()
        neg_count_np = neg_count.cpu().data.numpy()
        vals_np = np.concatenate((pos_vals_np, np.delete(neg_vals_np, y_ind)), axis=0)
        count_np = np.concatenate((pos_count_np, np.delete(neg_count_np, y_ind)), axis=0)
        #print(vals_np.shape, count_np.shape)
        # exit(0)
        vals = torch.IntTensor(vals_np).cuda()
        count = torch.FloatTensor(count_np).cuda()
        print(vals.shape, count.shape)
    else:
        vals = torch.cat((pos_vals, neg_vals), dim=0)
        count = torch.cat((pos_count, neg_count), dim=0)
    # exit(0)
    re_ts, re_ys, re_xs = dehash_3d(vals.unsqueeze(0))
    
    # exit(0)
    
    ps = count.float().unsqueeze(0)
    
    zeros = torch.zeros_like(re_ts)
    ones = torch.ones_like(re_ts)
    index_s = torch.cat((zeros, re_ts, re_ys, re_xs), dim=0).permute(1,0).cuda()
    index_s1 = torch.cat((ones, re_ts, re_ys, re_xs), dim=0).permute(1,0).cuda()
    #index_s2 = torch.cat((zeros, re_ts, re_ys, re_xs), dim=0).permute(1,0).cuda()
    #index_s2[:,0] = 1
    #print(torch.mean(torch.abs(index_s2.float() - index_s1.float())))
    #exit(0)
    ps = ps.permute(1,0).cuda()
    print("PS max min", index_s.shape, ps.shape, ps.max(), ps.min(), re_ts.max(), re_ts.min())
    # exit(0)
    
    network = EFNet5148d().cuda()
    # out = network(img, ps.clone(), index_s.clone())
    ps_batch = torch.cat((ps,ps[0:-1]),dim=0)
    index_s_batch = torch.cat((index_s, index_s1[0:-1]), dim=0)
    img_batch = torch.cat((img, img), dim=0)
    out, _ = network(img_batch, img_batch, ps_batch, index_s_batch, 2, [16,256,256])
    print(out[0].shape)
    exit(0)
    
