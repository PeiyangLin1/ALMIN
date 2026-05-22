import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import DropPath
from einops import rearrange
import torch
from torch import Tensor, nn
from abc import abstractmethod
def silu(x):
    return x * F.sigmoid(x)
class RMSNorm(nn.Module):

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, z):
        x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2(nn.Module):
    def __init__(self, d_model: int,
                 n_layer: int = 24,
                 d_state: int = 128,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,
                 chunk_size: int = 64,
                 ):
        super().__init__()
        self.n_layer = n_layer
        self.d_state = d_state
        self.headdim = headdim
        self.chunk_size = chunk_size

        self.d_inner = expand * d_model
        assert self.d_inner % self.headdim == 0, "self.d_inner 必须能被 self.headdim 整除"
        self.nheads = self.d_inner // self.headdim

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, ))
        self.A_log = nn.Parameter(torch.empty(self.nheads, ))
        self.D = nn.Parameter(torch.empty(self.nheads, ))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: Tensor):
        A = -torch.exp(self.A_log)

        zxbcdt = self.in_proj(u)

        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias)

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )

        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        _b, _l, _hp = x.shape
        _h = _hp // self.headdim
        _p = self.headdim
        x = x.reshape(_b, _l, _h, _p)

        y = self.ssd(x * dt.unsqueeze(-1),
                     A * dt,
                     B.unsqueeze(2),
                     C.unsqueeze(2))

        y = y + x * self.D.unsqueeze(-1)

        _b, _l, _h, _p = y.shape
        y = y.reshape(_b, _l, _h * _p)

        y = self.norm(y, z)

        y = self.out_proj(y)

        return y

    def segsum(self, x: Tensor) -> Tensor:
        T = x.size(-1)
        device = x.device
        x = x[..., None].repeat(1, 1, 1, 1, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, x, A, B, C):
        chunk_size = self.chunk_size
        x = x.reshape(x.shape[0], x.shape[1] // chunk_size, chunk_size, x.shape[2], x.shape[3])
        B = B.reshape(B.shape[0], B.shape[1] // chunk_size, chunk_size, B.shape[2], B.shape[3])
        C = C.reshape(C.shape[0], C.shape[1] // chunk_size, chunk_size, C.shape[2], C.shape[3])
        A = A.reshape(A.shape[0], A.shape[1] // chunk_size, chunk_size, A.shape[2])
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        Y = Y_diag + Y_off
        Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4])

        return Y


class _BiMamba2(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 d_model: int,
                 n_layer: int = 24,
                 d_state: int = 128,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,
                 chunk_size: int = 64,
                 ):
        super().__init__()
        self.fc_in = nn.Linear(cin, d_model, bias=False)
        self.mamba2_for = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size)
        self.mamba2_back = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size)
        self.fc_out = nn.Linear(d_model, cout, bias=False)
        self.chunk_size = chunk_size

    @abstractmethod
    def forward(self, x):
        pass


class BiMamba2_2D(_BiMamba2):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__(cin, cout, d_model, **mamba2_args)
        self.fc_in = torch.nn.Linear(cin, d_model)
        self.fc_out = torch.nn.Linear(d_model, cout)

    def forward(self, x, return_intermediate=False):
        h, w = x.shape[2:]
        x = F.pad(x, (0, (8 - x.shape[3] % 8) % 8,
                      0, (8 - x.shape[2] % 8) % 8))
        _b, _c, _h, _w = x.shape

        x = x.permute(0, 2, 3, 1).reshape(_b, _h * _w, _c)

        x = self.fc_in(x)
        x1 = self.mamba2_for(x)
        x2 = self.mamba2_back(x.flip(1)).flip(1)

        x = x1 + x2
        x = self.fc_out(x)

        x = x.reshape(_b, _h, _w, -1).permute(0, 3, 1, 2)
        x = x[:, :, :h, :w]

        if return_intermediate:
            x1 = x1.reshape(_b, _h, _w, -1).permute(0, 3, 1, 2)
            x1 = x1[:, :, :h, :w]
            x2 = x2.reshape(_b, _h, _w, -1).permute(0, 3, 1, 2)
            x2 = x2[:, :, :h, :w]
            return x, x1, x2
        return x
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class Dense(nn.Module):
    def __init__(self, in_channels):
        super(Dense, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1)
        self.conv6 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)


        self.gelu = nn.GELU()

    def forward(self, x):


        x1 = self.conv1(x)
        x1 = self.gelu(x1+x)

        x2 = self.conv2(x)
        x2 = self.gelu(x2+x1+x)

        x3 =self.conv3(x)
        x3 = self.gelu(x3+x2+x1+x)

        x4 = self.conv4(x)
        x4 = self.gelu(x4+x3+x2+x1+x)

        x5 =self.conv5(x)
        x5 = self.gelu(x5+x4+x3+x2+x1+x)

        x6 = self.conv6(x)
        x6 = self.gelu(x6+x5+x4+x3+x2+x1+x)

        return x6
class AFFM(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(AFFM, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)
        self.dense = Dense(in_channels=in_channels)
    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)
        fusion_out = self.dense(feats_V)
        return fusion_out

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GIPB(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(GIPB, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = BiMamba2_2D(64, 64, 64)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class FRPB(nn.Module):
    def __init__(self, num_layers=3):
        super(FRPB, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)



import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        _, _, H, W = x.shape

        x_freq = torch.fft.rfft2(x, norm='backward')

        ori_mag = torch.abs(x_freq)
        mag = self.processmag(ori_mag)
        mag = ori_mag + mag

        ori_pha = torch.angle(x_freq)
        pha = self.processpha(ori_pha)
        pha = ori_pha + pha


        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_freq_spatial+x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.Q = nn.Sequential(Frequency_Spectrum_Dynamic_Aggregation(nc=dim), nn.Conv2d(dim, dim, 1))
        self.K = nn.Sequential(Frequency_Spectrum_Dynamic_Aggregation(nc=dim), nn.Conv2d(dim, dim, 1))
        self.V = nn.Sequential(Frequency_Spectrum_Dynamic_Aggregation(nc=dim), nn.Conv2d(dim, dim, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm=nn.BatchNorm2d):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):

        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class LGFB(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(LGFB, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.embed = OverlapPatchEmbed(1, 64, bias=True)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(d, dim, 1, bias=True)
        )
        self.gate = nn.Sigmoid()
        self.edg = EdgeEnhancer(dim, norm=nn.BatchNorm2d)
        self.weight_edg1 = EdgeEnhancer(1, norm=nn.BatchNorm2d)
        self.weight_edg2 = EdgeEnhancer(dim, norm=nn.BatchNorm2d)
        self.embed1 = OverlapPatchEmbed(1, 64, bias=True)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(d, dim, 1, bias=True)
        )
        self.gate1 = nn.Sigmoid()

        self.weight_edg1_1 = EdgeEnhancer(1, norm=nn.BatchNorm2d)
        self.weight_edg2_1 = EdgeEnhancer(dim, norm=nn.BatchNorm2d)

    def forward(self, in_feats1, in_feats2, weight):

        attn = self.weight_edg2(self.mlp(self.embed(self.weight_edg1(weight))))
        attn = self.gate(attn)

        re1 = self.weight_edg2_1(self.mlp1(self.embed1(self.weight_edg1_1(weight))))
        re2 = self.gate1(re1)
        out = in_feats1 * re2 + in_feats2 * attn
        out = self.edg(out)
        return out


class Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Encoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.TI = DWTInverse(mode='zero', wave='haar')
        self.fusion2 = AFFM(in_channels=dim, height=3, reduction=8)
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.GIPB = GIPB(dim=dim, num_heads=heads[2])
        self.FRPB = FRPB()
        self.blk3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.Conv2d(dim, dim * 3, 3, padding=1, bias=False),
            nn.Conv2d(dim * 3, dim * 3, 3, padding=1, bias=False),

        )

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.GIPB(out_enc_level1)

        yL, yH = self.wt(out_enc_level1)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        detail_feature_low = self.FRPB(yL)
        detail_feature_high = self.fusion2([y_HL, y_LH, y_HH])
        detail_feature_high = self.blk3(detail_feature_high)
        y_HL, y_LH, y_HH = torch.chunk(detail_feature_high, chunks=3, dim=1)
        high_freq = torch.stack([y_HL, y_LH, y_HH], dim=2)
        detail_feature = self.TI((detail_feature_low, [high_freq]))

        return base_feature, detail_feature, out_enc_level1


class Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0





