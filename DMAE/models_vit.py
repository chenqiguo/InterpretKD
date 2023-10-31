from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        """
        # for debug:
        print('****** debug 2')
        print(x.shape) # torch.Size([128, 3, 224, 224])
        #assert(False)
        """
        x = self.patch_embed(x)
        """
        # for debug:
        print('****** debug 3')
        print(x.shape) # torch.Size([128, 196, 768])
        #assert(False)
        """
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        """
        # for debug:
        print('****** debug 4')
        print(x.shape) # torch.Size([128, 197, 768])
        #assert(False)
        """
        for blk in self.blocks:
            """
            # for debug:
            print('****** debug 5')
            print(blk) # ...LayerNorm((768,)...
            """
            x = blk(x)
        """
        # for debug:
        print('****** debug 6')
        print(x.shape) # torch.Size([128, 197, 768])
        #assert(False)
        """
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        """
        # for debug:
        print('****** debug 7')
        print(x.shape) # torch.Size([128, 768])
        print(outcome.shape) # torch.Size([128, 768])
        #assert(False)
        """
        return outcome
    
    # newly added by Chenqi:
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    
    

def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
    
    
    
def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
