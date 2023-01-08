import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
'''
self.s = torch.nn.Parameter(torch.ones(1))  #V2
激活值量化参数s初始化使用了常数1
'''

# ********************* quantizers（量化器，量化） *********************
# 取整(ste)
class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class FunLSQ(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel=False):
        #根据论文里LEARNED STEP SIZE QUANTIZATION第2节的公式
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        #根据论文里LEARNED STEP SIZE QUANTIZATION第2.1节
        #分为三部分：位于量化区间的、小于下界的、大于上界的
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float() #bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float() #bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller -bigger #得到位于量化区间的index
        if per_channel:
            grad_alpha = ((smaller * Qn + bigger * Qp + 
                between * Round.apply(q_w) - between * q_w)*grad_weight * g)
            grad_alpha = grad_alpha.contiguous().view(grad_alpha.size()[0], -1).sum(dim=1)
        else:
            grad_alpha = ((smaller * Qn + bigger * Qp + 
                between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0) #?
        #在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight  
        return grad_weight, grad_alpha, None, None, None, None

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

# A(特征)量化
class LSQActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False, batch_init = 20):
        #activations 没有per-channel这个选项的
        super(LSQActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)  #V2
        # self.register_parameter('Ascale', self.s)
        self.init_state = 0

    # 量化/反量化
    def forward(self, activation):
        if self.a_bits == 32:
            output = activation
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            if self.init_state==0:
                self.g = 1.0/math.sqrt(activation.numel() * self.Qp)
                self.init_state += 1
            # print(self.s, self.g)
            q_a = FunLSQ.apply(activation, self.s, self.g, self.Qn, self.Qp)

            # alpha = grad_scale(self.s, g)
            # q_a = Round.apply((activation/alpha).clamp(Qn, Qp)) * alpha
        return q_a

# W(权重)量化
class LSQWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, per_channel=False, batch_init = 20):
        super(LSQWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** w_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (w_bits - 1)
            self.Qp = 2 ** (w_bits - 1) - 1
        self.per_channel = per_channel
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.register_parameter('Wscale', self.s)
        self.init_state = 0

    # 量化/反量化
    def forward(self, weight):
        if self.init_state==0:
            self.g = 1.0/math.sqrt(weight.numel() * self.Qp)
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                self.s.data = torch.mean(torch.abs(weight_tmp), dim=1)*2/(math.sqrt(self.Qp))
            else:
                self.s.data = torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.Qp))
            self.init_state += 1
        elif self.init_state<self.batch_init:
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                self.s.data = 0.9*self.s.data + 0.1*torch.mean(torch.abs(weight_tmp), dim=1)*2/(math.sqrt(self.Qp))
            else:
                self.s.data = 0.9*self.s.data + 0.1*torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.Qp))
            self.init_state += 1
        elif self.init_state==self.batch_init:
            # self.s = torch.nn.Parameter(self.s)
            self.init_state += 1
        if self.w_bits == 32:
            output = weight
        elif self.w_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.w_bits != 1
        else:
            # print(self.s, self.g)
            w_q = FunLSQ.apply(weight, self.s, self.g, self.Qn, self.Qp, self.per_channel)

            # alpha = grad_scale(self.s, g)
            # w_q = Round.apply((weight/alpha).clamp(Qn, Qp)) * alpha
        return w_q

class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False, 
                 all_positive=False, 
                 per_channel=False, 
                 batch_init = 20):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive,batch_init = batch_init)
        self.weight_quantizer = LSQWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel,batch_init = batch_init)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        # print('input:',input.size(),self.quant_inference)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)
        return output


class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False, 
                 all_positive=False, 
                 per_channel=False, 
                 batch_init = 20):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                                   dilation, groups, bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive,batch_init = batch_init)
        self.weight_quantizer = LSQWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel,batch_init = batch_init)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return output


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False, 
                 all_positive=False, 
                 per_channel=False, 
                 batch_init = 20):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive,batch_init = batch_init)
        self.weight_quantizer = LSQWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel,batch_init = batch_init)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8,
                 quant_inference=False, all_positive=False, per_channel=False, 
                 batch_init = 20):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.ConvTranspose2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                                child.out_channels,
                                                                child.kernel_size,
                                                                stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation,
                                                                groups=child.groups,
                                                                bias=True,
                                                                padding_mode=child.padding_mode,
                                                                a_bits=a_bits,
                                                                w_bits=w_bits,
                                                                quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
                    quant_conv_transpose.bias.data = child.bias
                else:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                                child.out_channels,
                                                                child.kernel_size,
                                                                stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation,
                                                                groups=child.groups, bias=False,
                                                                padding_mode=child.padding_mode,
                                                                a_bits=a_bits,
                                                                w_bits=w_bits,
                                                                quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=True, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
                    quant_linear.bias.data = child.bias
                else:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=False, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
                quant_linear.weight.data = child.weight
                module._modules[name] = quant_linear
        else:
            add_quant_op(child, layer_counter, a_bits=a_bits, w_bits=w_bits,
                         quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)


def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False,
            all_positive=False, per_channel=False, batch_init = 20):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits,
                 quant_inference=quant_inference, all_positive=all_positive, 
                 per_channel=per_channel, batch_init = batch_init)
    return model
