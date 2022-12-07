import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericConv(nn.Module):
    def __init__(self, in_channels, out_channels, *args, norm=nn.BatchNorm2d, act=nn.ReLU, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, *args, **kwargs)

        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        if act is not None:
            self.act = act()
        else:
            self.act = None

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 1)
        self.c2 = nn.Conv2d(channels, channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        self.c1.reset_parameters()
        self.c2.reset_parameters()

    def forward(self, x):
        y = self.pool(x)
        y = self.c1(y)
        y = self.relu(y)
        y = self.c2(y)
        return self.sigmoid(y) * x




class ConvBlock(nn.Module):
    def __init__(self, channels, imgsize):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(GenericConv(channels, channels, kernel_size=3, padding=1))

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class MBConvBlock(nn.Module):
    expansion_ratios = [2, 2]

    def __init__(self, channels, imgsize):
        super().__init__()
        expansions = [channels*er for er in self.expansion_ratios]

        self.layers = nn.ModuleList()
        for i in range(2): # inverted bottleneck 
            self.layers.append(GenericConv(channels, expansions[i], kernel_size=3, padding=1))
            self.layers.append(SE(expansions[i]))
            self.layers.append(GenericConv(expansions[i], channels, kernel_size=1))

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def forward(self, x):
        y = x
        for i, l in enumerate(self.layers):
            y = l(y)
            if (i+1) % 3 == 0:
                y = y + x
                x = y

        return y


class BottleneckBlock(nn.Module):
    bottleneck_ratios = [2, 1, 0.5]

    def __init__(self, channels, imgsize):
        super().__init__()
        bottlenecks = [int(channels//br) for br in self.bottleneck_ratios]

        self.layers = nn.ModuleList()
        for i in range(6): # depthwise-sep convs
            self.layers.append(GenericConv(channels, bottlenecks[i%3], kernel_size=1))
            self.layers.append(GenericConv(bottlenecks[i%3], bottlenecks[i%3], kernel_size=5, padding=2, groups=bottlenecks[i%3]))
            self.layers.append(GenericConv(bottlenecks[i%3], channels, kernel_size=1))

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def forward(self, x):
        y = x
        for i, l in enumerate(self.layers):
            y = l(y)
            if (i+1) % 3 == 0:
                y = y + x
                x = y

        return y


class Zero(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return torch.zeros_like(x)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = [img_size, img_size]
        patch_size = [patch_size, patch_size]
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, channels, img_size, patch_size=4, embedding_dim=96, num_heads=4, mlp_ratio=4, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=channels, embed_dim=embedding_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dim))

        self.norm1 = norm_layer(embedding_dim)
        self.attn = Attention(embedding_dim, num_heads=num_heads, qkv_bias=False, attn_drop=0, proj_drop=0)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(embedding_dim)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embedding_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=0)

        self.rev_proj = nn.ConvTranspose2d(embedding_dim, channels, kernel_size=patch_size, stride=patch_size)

        self.embedding_dim = embedding_dim
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        # remove class token
        x = x[:,:-1,:]

        x.transpose(1, 2) # BNC -> BCN
        x = x.reshape(B, self.embedding_dim, H//self.patch_size, W//self.patch_size)
        x = self.rev_proj(x)
        x = x.reshape(B, C, H, W)
        return x


if __name__ == '__main__':
    import ptflops
    for sizes in [(32,32), (64,16), (128,8)]:
        c, img = sizes
        print(c, img)
        for bt in [ConvBlock, MBConvBlock, BottleneckBlock, ViTBlock]:
            b = bt(c, img)
            print(bt.__name__, ptflops.get_model_complexity_info(b, (c,img,img), print_per_layer_stat=False, as_strings=True))

        print()
