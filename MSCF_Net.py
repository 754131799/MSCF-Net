import torch
import torch.nn as nn
import torch.nn.functional as F


class _CP_ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim // reduction, 1, bias=True)
        self.fc2 = nn.Conv2d(dim // reduction, dim, 1, bias=True)

    def forward(self, x):
        avg = self.fc2(torch.relu(self.fc1(F.adaptive_avg_pool2d(x, (1, 1)))))
        mx = self.fc2(torch.relu(self.fc1(F.adaptive_max_pool2d(x, (1, 1)))))
        return torch.sigmoid(avg + mx)


class MTAB(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, reduction=4):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.ca = _CP_ChannelAttention(out_ch, reduction)
        self.dconv5 = nn.Conv2d(out_ch, out_ch, 5, padding=2, groups=out_ch)
        self.dconv7h = nn.Conv2d(out_ch, out_ch, (1, 7), padding=(0, 3), groups=out_ch)
        self.dconv7v = nn.Conv2d(out_ch, out_ch, (7, 1), padding=(3, 0), groups=out_ch)
        self.dconv11h = nn.Conv2d(out_ch, out_ch, (1, 11), padding=(0, 5), groups=out_ch)
        self.dconv11v = nn.Conv2d(out_ch, out_ch, (11, 1), padding=(5, 0), groups=out_ch)
        self.dconv21h = nn.Conv2d(out_ch, out_ch, (1, 21), padding=(0, 10), groups=out_ch)
        self.dconv21v = nn.Conv2d(out_ch, out_ch, (21, 1), padding=(10, 0), groups=out_ch)
        self.conv1x1 = nn.Conv2d(out_ch, out_ch, 1)
        if in_ch == out_ch and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.conv1x1(out)
        out = self.act(out)
        out = out * self.ca(out)
        x1 = self.dconv5(out)
        x2 = self.dconv7v(self.dconv7h(x1))
        x3 = self.dconv11v(self.dconv11h(x1))
        x4 = self.dconv21v(self.dconv21h(x1))
        out = self.conv1x1(x1 + x2 + x3 + x4)
        return self.act(out + identity)


class _ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        mx = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return torch.sigmoid(avg + mx)


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, dim, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = _ChannelAttention(dim, ratio)
        self.sa = _SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(x)


class WrapFusion(nn.Module):
    def __init__(self, fusion_cls, dim):
        super().__init__()
        self.fusion = fusion_cls(dim)

    def forward(self, f_spect, f_ct):
        fused = self.fusion(f_spect, f_ct)
        return f_spect + fused, f_ct + fused


class _MSCA(nn.Module):
    def __init__(self, dim, nh=8):
        super().__init__()
        self.dim = dim
        self.nh = nh
        self.hd = dim // nh
        self.scale = self.hd ** -0.5
        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.a1 = nn.Parameter(torch.tensor(0.5))
        self.a2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, y):
        B, C, H, W = x.shape
        target_N = 64
        q_h = max(8, min(H, int((target_N * W / H) ** 0.5)))
        q_w = max(8, min(W, int((target_N * H / W) ** 0.5)))
        N = q_h * q_w
        q = F.adaptive_avg_pool2d(self.q(x), (q_h, q_w))
        kv = F.adaptive_avg_pool2d(self.kv(y), (q_h, q_w))
        k, v = kv[:, :C], kv[:, C:]
        q = q.reshape(B, self.nh, N, self.hd)
        k = k.reshape(B, self.nh, N, self.hd)
        v = v.reshape(B, self.nh, N, self.hd)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        o1 = attn @ v
        o2 = attn @ v
        out = (o1 * self.a1 + o2 * self.a2).transpose(-2, -1).reshape(B, C, q_h, q_w)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return self.proj(out)


class DualEncoder2D(nn.Module):
    def __init__(self, in_ch=1, dims=[48, 96, 192, 384], depths=[2, 2, 2, 2]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0] // 2, 7, 2, 3, bias=False),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
        )
        self.conv_blocks = nn.ModuleList()
        for i, (d, depth) in enumerate(zip(dims, depths)):
            in_d = dims[0] // 2 if i == 0 else dims[i - 1]
            stride = 2 if i > 0 else 1
            blocks = [MTAB(in_d, d, stride)]
            blocks += [MTAB(d, d) for _ in range(depth - 1)]
            self.conv_blocks.append(nn.Sequential(*blocks))
        self.cbam_blocks = nn.ModuleList([CBAM(d) for d in dims])

    def forward(self, x):
        x = self.stem(x)
        conv_feats = []
        for block in self.conv_blocks:
            x = block(x)
            conv_feats.append(x)
        cbam_feats = [self.cbam_blocks[i](conv_feats[i]) for i in range(len(conv_feats))]
        return conv_feats, cbam_feats


class UpBlock2D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, fusion_blk=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, 2)
        self.skip_fusion = fusion_blk
        self.fuse = MTAB(in_ch + skip_ch, out_ch)

    def forward(self, x, skip_spect, skip_ct):
        up_f = self.up(x)
        if self.skip_fusion is not None:
            _, skip_f = self.skip_fusion(skip_spect, skip_ct)
            fused_skip = skip_spect + skip_ct + skip_f
        else:
            fused_skip = skip_spect + skip_ct
        return self.fuse(torch.cat([up_f, fused_skip], dim=1))


class MSCF_Net(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, dims=[48, 96, 192, 384],
                 depths=[2, 2, 2, 2], dropout=0.0):
        super().__init__()
        self.enc_spect = DualEncoder2D(1, dims, depths)
        self.enc_ct = DualEncoder2D(1, dims, depths)
        self.fusion = nn.ModuleList([WrapFusion(_MSCA, d) for d in dims])
        skip_dims = [dims[2], dims[1], dims[0]]
        self.skip_fusions = nn.ModuleList([WrapFusion(_MSCA, d) for d in skip_dims])
        self.decoder4 = UpBlock2D(dims[3], dims[2], dims[2], fusion_blk=self.skip_fusions[0])
        self.decoder3 = UpBlock2D(dims[2], dims[1], dims[1], fusion_blk=self.skip_fusions[1])
        self.decoder2 = UpBlock2D(dims[1], dims[0], dims[0], fusion_blk=self.skip_fusions[2])
        self.decoder1 = MTAB(dims[0], dims[0])
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Dropout2d(dropout),
            nn.Conv2d(dims[0], out_chans, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        spect = x[:, 0:1]
        ct = x[:, 1:2]
        conv_s, cbam_s = self.enc_spect(spect)
        conv_c, cbam_c = self.enc_ct(ct)
        enc_fused = []
        for i in range(4):
            fs = conv_s[i] + cbam_s[i]
            fc = conv_c[i] + cbam_c[i]
            fs_f, fc_f = self.fusion[i](fs, fc)
            enc_fused.append(fs_f + fc_f)
        x = self.decoder4(enc_fused[3], enc_fused[2], enc_fused[2])
        x = self.decoder3(x, enc_fused[1], enc_fused[1])
        x = self.decoder2(x, enc_fused[0], enc_fused[0])
        x = self.decoder1(x)
        return self.head(x)


if __name__ == "__main__":
    model = MSCF_Net(in_chans=1, out_chans=1)
    x = torch.randn(1, 2, 256, 256)
    y = model(x)
    print(f"Output shape: {y.shape}")
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {total:.2f}M")
