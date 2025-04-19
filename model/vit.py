import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

# Decoder Block: Used to upsample and convert features to an image.
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Vision Transformer Decoder
class ViTDecoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head = 64, dropout=0., emb_dropout=0., out_channels = 3):
        super(ViTDecoder, self).__init__()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Decoder network to reconstruct image
        self.decoder = nn.Sequential(
            DecoderBlock(dim, 128, kernel_size=4, stride=2, padding=5), # 116x116
            DecoderBlock(128, 64, kernel_size=4, stride=2, padding=1), # 232x232
            DecoderBlock(64, 32, kernel_size=4, stride=2, padding=1), # 464x464
            # DecoderBlock(32, 16, kernel_size=4, stride=2, padding=4), # 928x928
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2),  # Final layer to match the original image size 930x930
            nn.Sigmoid()
        )
    
    def forward(self, img):
        # Step 1: Patch embedding and adding positional encoding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Step 2: Pass through the transformer
        x = self.transformer(x)

        # Step 3: Decoder: Use transformer output to reconstruct image
        patch_tokens = x[:, 1:, :]  # Remove CLS token → [B, 196, 768]
    
        # B, N, C = patch_tokens.shape  # B=batch, N=196, C=768
        # H = W = int(N ** 0.5)         # Assumes square patches, H=W=14 here

        x = patch_tokens.permute(0, 2, 1).reshape(-1, 768, 62, 62)  # reshape to image-like
        x = self.decoder(x)  # Use decoder to reconstruct image

        return x
    
# Vision Transformer Decoder, 3 encoder approach
class ViTDecoder_v2(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head = 64, dropout=0., emb_dropout=0., out_channels = 3):
        super(ViTDecoder_v2, self).__init__()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding_X = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding_X = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_X = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_X = nn.Dropout(emb_dropout)
        self.transformer_X = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_patch_embedding_Y = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding_Y = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_Y = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_Y = nn.Dropout(emb_dropout)
        self.transformer_Y = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_patch_embedding_Z = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding_Z = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_Z = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_Z = nn.Dropout(emb_dropout)
        self.transformer_Z = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        # Decoder network to reconstruct image
        self.decoder = nn.Sequential(
            DecoderBlock(dim, 128, kernel_size=4, stride=2, padding=5), # 116x116
            DecoderBlock(128, 64, kernel_size=4, stride=2, padding=1), # 232x232
            DecoderBlock(64, 32, kernel_size=4, stride=2, padding=1), # 464x464
            # DecoderBlock(32, 16, kernel_size=4, stride=2, padding=4), # 928x928
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2),  # Final layer to match the original image size 930x930
            nn.Sigmoid()
        )
    
    def forward(self, imgs, names):
        # Step 1: Patch embedding and adding positional encoding
        img1 = imgs[0]
        img2 = imgs[1]
        img3 = imgs[2]

        counter = 0
        for name in names:
            if "current.png" in name:
                img1 = imgs[counter]
            elif "eff_dist" in name:
                img2 = imgs[counter]
            elif "pdn_density" in name:
                img3 = imgs[counter]
            counter += 1

        x = self.to_patch_embedding_X(img1)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token_X, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding_X[:, :(n + 1)]
        x = self.dropout_X(x)

        y = self.to_patch_embedding_Y(img2)
        b, n, _ = y.shape
        cls_tokens = repeat(self.cls_token_Y, '1 1 d -> b 1 d', b=b)
        y = torch.cat((cls_tokens, y), dim=1)
        y += self.pos_embedding_Y[:, :(n + 1)]
        y = self.dropout_Y(y)

        z = self.to_patch_embedding_Z(img3)
        b, n, _ = z.shape
        cls_tokens = repeat(self.cls_token_Z, '1 1 d -> b 1 d', b=b)
        z = torch.cat((cls_tokens, z), dim=1)
        z += self.pos_embedding_Z[:, :(n + 1)]
        z = self.dropout_Z(x)

        # Step 2: Pass through the transformer
        x = self.transformer_X(x)
        y = self.transformer_Y(y)
        z = self.transformer_Z(z)

        # Step 3: Decoder: Use transformer output to reconstruct image
        patch_tokens_X = x[:, 1:, :]  # Remove CLS token → [B, 196, 768]
        patch_tokens_Y = y[:, 1:, :]
        patch_tokens_Z = z[:, 1:, :]

        w = nn.functional.softmax(self.weights, dim=0)
        patch_tokens = w[0]*patch_tokens_X + w[1]*patch_tokens_Y + w[2]*patch_tokens_Z

        a = patch_tokens.permute(0, 2, 1).reshape(-1, 768, 62, 62)  # reshape to image-like
        a = self.decoder(a)  # Use decoder to reconstruct image

        return a
    
class ViTDecoder_v3(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head = 64, dropout=0., emb_dropout=0., out_channels = 3):
        super(ViTDecoder_v3, self).__init__()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding_X = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding_X = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_X = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_X = nn.Dropout(emb_dropout)
        self.transformer_X = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_patch_embedding_Y = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding_Y = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_Y = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_Y = nn.Dropout(emb_dropout)
        self.transformer_Y = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_patch_embedding_Z = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding_Z = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_Z = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_Z = nn.Dropout(emb_dropout)
        self.transformer_Z = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_patch_embedding_Q = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim*4),
            nn.Linear(patch_dim*4, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding_Q = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_Q = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_Q = nn.Dropout(emb_dropout)
        self.transformer_Q = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))

        # Decoder network to reconstruct image
        self.decoder = nn.Sequential(
            DecoderBlock(dim, 128, kernel_size=4, stride=2, padding=5), # 116x116
            DecoderBlock(128, 64, kernel_size=4, stride=2, padding=1), # 232x232
            DecoderBlock(64, 32, kernel_size=4, stride=2, padding=1), # 464x464
            # DecoderBlock(32, 16, kernel_size=4, stride=2, padding=4), # 928x928
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2),  # Final layer to match the original image size 930x930
            nn.Sigmoid()
        )
    
    def forward(self, imgs, names):
        # Step 1: Patch embedding and adding positional encoding
        img1 = imgs[0]
        img2 = imgs[1]
        img3 = imgs[2]

        counter = 0
        for name in names:
            if "current.png" in name:
                img1 = imgs[counter]
            elif "eff_dist" in name:
                img2 = imgs[counter]
            elif "pdn_density" in name:
                img3 = imgs[counter]
            counter += 1

        x = self.to_patch_embedding_X(img1)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token_X, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding_X[:, :(n + 1)]
        x = self.dropout_X(x)

        y = self.to_patch_embedding_Y(img2)
        b, n, _ = y.shape
        cls_tokens = repeat(self.cls_token_Y, '1 1 d -> b 1 d', b=b)
        y = torch.cat((cls_tokens, y), dim=1)
        y += self.pos_embedding_Y[:, :(n + 1)]
        y = self.dropout_Y(y)

        z = self.to_patch_embedding_Z(img3)
        b, n, _ = z.shape
        cls_tokens = repeat(self.cls_token_Z, '1 1 d -> b 1 d', b=b)
        z = torch.cat((cls_tokens, z), dim=1)
        z += self.pos_embedding_Z[:, :(n + 1)]
        z = self.dropout_Z(x)

        q = self.to_patch_embedding_Z(img3)
        b, n, _ = q.shape
        cls_tokens = repeat(self.cls_token_Q, '1 1 d -> b 1 d', b=b)
        q = torch.cat((cls_tokens, q), dim=1)
        q += self.pos_embedding_Q[:, :(n + 1)]
        q = self.dropout_Q(x)

        # Step 2: Pass through the transformer
        x = self.transformer_X(x)
        y = self.transformer_Y(y)
        z = self.transformer_Z(z)
        q = self.transformer_Q(q)

        # Step 3: Decoder: Use transformer output to reconstruct image
        patch_tokens_X = x[:, 1:, :]  # Remove CLS token → [B, 196, 768]
        patch_tokens_Y = y[:, 1:, :]
        patch_tokens_Z = z[:, 1:, :]
        patch_tokens_Q = q[:, 1:, :]


        w = nn.functional.softmax(self.weights, dim=0)
        patch_tokens = w[0]*patch_tokens_X + w[1]*patch_tokens_Y  + w[2]*patch_tokens_Z + w[3]*patch_tokens_Q

        a = patch_tokens.permute(0, 2, 1).reshape(-1, 768, 62, 62)  # reshape to image-like
        a = self.decoder(a)  # Use decoder to reconstruct image

        return a