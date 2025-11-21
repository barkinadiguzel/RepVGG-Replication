import torch
import torch.nn as nn
import torch.nn.functional as F

def fuse_conv_bn(conv, bn):
    w = conv.weight
    if conv.bias is None:
        b = torch.zeros(w.size(0), device=w.device)
    else:
        b = conv.bias

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    w_fused = w * (gamma / torch.sqrt(var + eps)).reshape([-1,1,1,1])
    b_fused = beta + (b - mean) * gamma / torch.sqrt(var + eps)
    return w_fused, b_fused

def pad_1x1_to_3x3(w_1x1):
    C_out, C_in, _, _ = w_1x1.shape
    w_3x3 = torch.zeros(C_out, C_in, 3, 3, device=w_1x1.device)
    w_3x3[:,:,1,1] = w_1x1[:,:,0,0]
    return w_3x3

def fuse_residual_block(block):
    w3, b3 = fuse_conv_bn(block.conv3x3, block.bn3x3)
    
    w_total = w3
    b_total = b3

    if block.conv1x1 is not None:
        w1, b1 = fuse_conv_bn(block.conv1x1, block.bn1x1)
        w1_pad = pad_1x1_to_3x3(w1)
        w_total += w1_pad
        b_total += b1

    if block.use_identity:
        C = w_total.shape[0]
        w_id = torch.zeros_like(w_total)
        for i in range(C):
            w_id[i,i,1,1] = 1.0
        w_total += w_id

    fused_conv = nn.Conv2d(block.conv3x3.in_channels,
                           block.conv3x3.out_channels,
                           kernel_size=3, stride=block.conv3x3.stride,
                           padding=1, bias=True)
    fused_conv.weight.data = w_total
    fused_conv.bias.data = b_total
    return fused_conv
