import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import CondConv2d, get_condconv_initializer, create_conv2d, DropPath, get_norm_act_layer


def num_groups(group_size, channels):
    if not group_size:
        return 1
    else:
        assert channels % group_size == 0
        return channels // group_size


def _init_weight_goog(m, n='', fix_group_fanout=True):
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(
            lambda w: nn.init.normal_(w, 0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        nn.init.uniform_(m.weight, -init_range, init_range)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1)
        self.pos = nn.Conv2d(int(dim * mlp_ratio), int(dim * mlp_ratio),
                             3, padding=1, groups=int(dim * mlp_ratio))
        self.fc2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x


class Conv2Former(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()

        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + \
            self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + \
            self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class Bottleneck(nn.Module):
    def __init__(self, dim):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                      padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
        )

    def forward(self, input):
        return self.bottleneck(input)


class ASPP(nn.Module):
    def __init__(self, in_channels, hidden_channel, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channel, 1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, hidden_channel, rate1))
        modules.append(ASPPConv(in_channels, hidden_channel, rate2))
        modules.append(ASPPConv(in_channels, hidden_channel, rate3))
        modules.append(ASPPPooling(in_channels, hidden_channel))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * hidden_channel, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Cloud_Detection_Module(nn.Module):
    def __init__(self, dim=48):
        super(Cloud_Detection_Module, self).__init__()
        self.cloud_det = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(True),
            nn.Conv2d(2*dim, 4*dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(True),
            nn.Conv2d(4*dim, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.cloud_det(input)


class Feature_Extractor(nn.Module):
    def __init__(self, dim=32):
        super(Feature_Extractor, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(4, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
        )
        self.ReLU = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.cloud_detection = Cloud_Detection_Module(dim)
        self.bottle1 = Bottleneck(dim)
        self.bottle2 = Bottleneck(dim)
        self.bottle3 = Bottleneck(dim)
        self.bottle4 = Bottleneck(dim)
        self.bottle5 = Bottleneck(dim)
        self.bottle6 = Bottleneck(dim)
        self.ASPP = ASPP(in_channels=dim, hidden_channel=128,
                         atrous_rates=[12, 24, 36])
        self.aux_conv = nn.Conv2d(dim, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        out = self.conv_in(x)
        out = self.ReLU(self.bottle1(out) + out)

        cloud_mask1 = self.cloud_detection(out)
        out = self.ReLU(self.bottle2(out) * cloud_mask1 + out)
        out = self.ReLU(self.bottle3(out) * cloud_mask1 + out)

        cloud_mask2 = self.cloud_detection(out)
        out = self.ReLU(self.bottle4(out) * cloud_mask2 + out)
        out = self.ReLU(self.bottle5(out) * cloud_mask2 + out)

        out = self.ReLU(self.bottle6(out) + out)
        out = self.ASPP(out)
        pred = self.tanh(self.aux_conv(out))

        return cloud_mask2, out, pred


class DepthwiseSeparableConv(nn.Module):
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            se_layer=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, groups=groups)
        self.bn1 = norm_act_layer(in_chs, inplace=True)

        self.se = se_layer(
            in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d(
            in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(
            out_chs, inplace=True, apply_act=self.has_pw_act)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:
            return dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),

            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class FirstInjectionMultiSum(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1, norm=nn.BatchNorm2d) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = nn.Sequential(
            nn.Conv2d(inp*3, oup*3, kernel, groups=groups*3,
                      padding=int((kernel - 1) / 2), bias=False),
            norm(oup*3)
        )
        self.global_embedding = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, groups=groups,
                      padding=int((kernel - 1) / 2), bias=False),
            norm(oup)
        )
        self.global_act = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, groups=groups,
                      padding=int((kernel - 1) / 2), bias=False),
            norm(oup)
        )
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        B, N, H, W = x_l.shape

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W))

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W))

        out = local_feat.view(B, 3, -1, H, W) * sig_act.view(B,
                                                             1, -1, H, W) + global_feat.view(B, 1, -1, H, W)
        return out.view(B, 3, -1, H, W).sum(1)


class InjectionMultiSum(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1, norm=nn.BatchNorm2d) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, groups=groups,
                      padding=int((kernel - 1) / 2), bias=False),
            norm(oup)
        )
        self.global_embedding = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, groups=groups,
                      padding=int((kernel - 1) / 2), bias=False),
            norm(oup)
        )
        self.global_act = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, groups=groups,
                      padding=int((kernel - 1) / 2), bias=False),
            norm(oup)
        )
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        B, N, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W))

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W))

        out = local_feat * sig_act + global_feat
        return out


class OtherUConvBlock(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=128, out_channels=64, upsampling_depth=4, norm=nn.BatchNorm2d, act=nn.ReLU, is_first=False):
        super().__init__()
        self.proj_1x1 = DepthwiseSeparableConv(
            in_channels, hidden_channels, dw_kernel_size=1, norm_layer=norm, act_layer=act)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DepthwiseSeparableConv(
                hidden_channels, hidden_channels, dw_kernel_size=3, stride=1, group_size=hidden_channels, pad_type="same"
            )
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                DepthwiseSeparableConv(
                    hidden_channels, hidden_channels, dw_kernel_size=3, stride=2, group_size=hidden_channels
                )
            )

        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(InjectionMultiSum(
                hidden_channels, hidden_channels))

        self.res_conv = nn.Conv2d(hidden_channels, out_channels, 1)

        self.globalatt = Conv2Former(hidden_channels, drop_path=0.1)

        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(InjectionMultiSum(
                hidden_channels, hidden_channels, 5))

        self.hidden = hidden_channels
        self.is_sum = in_channels == out_channels*3
        self.out_channels = out_channels
        self.is_first = is_first

        self._init_weights()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.is_sum:
            residual = x.view(B, -1, self.out_channels, H, W).sum(1).clone()
        else:
            residual = x.clone()

        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        global_f = torch.zeros(
            output[-1].shape, requires_grad=True, device=output1.device
        )
        for fea in output:
            global_f = global_f + F.adaptive_avg_pool2d(
                fea, output_size=output[-1].shape[-2:]
            )

        global_f = self.globalatt(global_f)

        x_fused = []

        for idx in range(self.depth):
            local = output[idx]
            x_fused.append(self.loc_glo_fus[idx](local, global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)

        if self.is_first:
            return self.res_conv(expanded)
        return self.res_conv(expanded) + residual

    def _init_weights(self):
        init_fn = _init_weight_goog
        for n, m in self.named_modules():
            init_fn(m, n)


class PMAA(nn.Module):
    def __init__(self, hidden_channels=64, out_channels=3, norm=nn.BatchNorm2d, act=nn.ReLU) -> None:
        super(PMAA, self).__init__()
        self.feature_extractor = Feature_Extractor(dim=hidden_channels)
        self.concat_unet = OtherUConvBlock(
            hidden_channels*3, hidden_channels*4, hidden_channels, upsampling_depth=5, norm=norm, act=act)
        self.other_unet = nn.Sequential(*[OtherUConvBlock(hidden_channels, hidden_channels*4,
                                        hidden_channels, upsampling_depth=5, norm=norm, act=act) for i in range(2)])
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels,
                      kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        multi_output = []
        multi_cloud_mask = []
        multi_pred = []
        for i in range(x.shape[1]):
            tmp_cloud, tmp_output, tmp_pred = self.feature_extractor(
                x[:, i, :, :, :])
            multi_output.append(tmp_output)
            multi_cloud_mask.append(tmp_cloud)
            multi_pred.append(tmp_pred)

        other_output = self.concat_unet(torch.cat(multi_output, dim=1))
        other_output = self.other_unet(other_output)
        return self.output_layer(other_output), multi_cloud_mask, multi_pred


if __name__ == '__main__':
    inp = torch.randn(2, 3, 4, 256, 256)
    net = PMAA(32, 4)
    print(net(inp)[1][0].shape)
