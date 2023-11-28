import math
import time
import torch
import torch.nn as nn
from functools import partial

from models.modeling_finetune import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from models.modeling_cae_helper import *

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class ContextAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None, init_std=0.02, 
                 decoder_embed_dim=768, regressor_depth=4, decoder_num_classes=8192, decoder_num_heads=12, 
                 decoder_layer_scale_init_value=0.1, decoder_depth=4, fix_init_weight=False , **kwargs):
        super().__init__()
        self.model_type = kwargs['args'].model_type
        if kwargs['args'].regressor_depth != regressor_depth: regressor_depth = kwargs['args'].regressor_depth
        if kwargs['args'].decoder_embed_dim != decoder_embed_dim: decoder_embed_dim = kwargs['args'].decoder_embed_dim
        if kwargs['args'].decoder_depth != decoder_depth: decoder_depth = kwargs['args'].decoder_depth
        print("regressor_depth: ", regressor_depth)
        print("decoder_embed_dim: ", decoder_embed_dim)
        print("decoder_depth: ", decoder_depth)

        self.encoder = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                 embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim, init_std=init_std)
        
        # alignment branch
        if self.model_type == 'caev2':
            self.alignment_encoder = None
        else:
            self.alignment_encoder = VisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                    embed_dim=embed_dim, depth=depth,
                    num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim, init_std=init_std)
        
        self.init_std = init_std
        self.num_patches = self.encoder.patch_embed.num_patches

        
        # from encoder to regresser projection, borrowed from mae.
        if decoder_embed_dim != embed_dim:
            self.encoder_to_regresser = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.encoder_to_regresser_norm = norm_layer(decoder_embed_dim)
        else:
            self.encoder_to_regresser = None
        

        # generate position embeddings for regresser and deocder (rd) .
        self.rd_pos_embed = self.encoder.build_2d_sincos_position_embedding(decoder_embed_dim, use_cls_token=True)
        
        
        # context regresser 
        self.regresser = LatentRegresser(embed_dim=decoder_embed_dim, regresser_depth=regressor_depth, num_heads=decoder_num_heads, 
                                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                        drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=decoder_layer_scale_init_value, init_std=init_std,
                                        model_type=self.model_type)
        
        # regress is cross attention, mask tokens are querries.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

        # decoder for recontrcution
        if self.model_type == 'caev2':
            self.num_out_dim = kwargs['args'].num_out_dim
            self.decoder = Decoder(num_classes=self.num_out_dim, embed_dim=decoder_embed_dim, decoder_depth=0,
                                norm_layer=norm_layer, init_values=decoder_layer_scale_init_value, init_std=init_std)
        else:
            self.decoder = Decoder(num_classes=decoder_num_classes, embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
                                num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
                                norm_layer=norm_layer, init_values=decoder_layer_scale_init_value, init_std=init_std)                
       
        ### whether to use 'rescale' to init the weight, borrowed from beit.
        if not fix_init_weight:
            self.apply(self._init_weights)
        if self.model_type != 'caev2':
            self._init_alignment_encoder()
        
    def _init_alignment_encoder(self):
        # init the weights of alignment_encoder with those of backbone
        for param_encoder, param_alignment_encoder in zip(self.encoder.parameters(), self.alignment_encoder.parameters()):
            param_alignment_encoder.detach()
            param_alignment_encoder.data.copy_(param_encoder.data)
            param_alignment_encoder.requires_grad = False


    def alignment_parameter_update(self):
        """parameter update of the alignment_encoder network."""
        for param_encoder, param_alignment_encoder in zip(self.encoder.parameters(),
                                                self.alignment_encoder.parameters()):
            param_alignment_encoder.data = param_encoder.data # completely copy


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    '''
    Input shape:
        x: [bs, 3, 224, 224]
        bool_masked_pos: [bs, num_patch * num_patch]
    '''
    def forward(self, x, bool_masked_pos):
        
        batch_size = x.size(0)

        '''
        Encoder
        Output shape:
            [bs, num_visible + 1, C]
        '''
        x_unmasked = self.encoder(x, bool_masked_pos=bool_masked_pos)

        # encoder to regresser projection
        if self.encoder_to_regresser is not None:
            x_unmasked = self.encoder_to_regresser(x_unmasked)
            x_unmasked = self.encoder_to_regresser_norm(x_unmasked)

        '''
        Alignment branch
        '''
        if self.model_type == 'caev2':
            latent_target = None
        else:
            with torch.no_grad():
                latent_target = self.alignment_encoder(x, bool_masked_pos=(~bool_masked_pos))
                latent_target = latent_target[:, 1:, :] # remove class token
                if self.encoder_to_regresser is not None:
                    latent_target = self.encoder_to_regresser_norm(self.encoder_to_regresser(latent_target.detach()))

                self.alignment_parameter_update()

        '''
        Latent contextual regressor
        1. prepare masked, unmasked pos embed, and masked mebedding
        '''
        _, num_visible_plus1, dim = x_unmasked.shape
        
        x_cls_token = x_unmasked[:, :1, :]
        x_unmasked = x_unmasked[:, 1:, :] # remove class token

        pos_embed = self.rd_pos_embed.expand(batch_size, self.num_patches+1, dim).cuda(x_unmasked.device)
        pos_embed_masked = pos_embed[:,1:][bool_masked_pos].reshape(batch_size, -1, dim)  # pos embed for masked patches
        pos_embed_unmasked = pos_embed[:,1:][~bool_masked_pos].reshape(batch_size, -1, dim)  # pos embed for unmasked patches

        num_masked_patches = self.num_patches - (num_visible_plus1-1)
        x_masked = self.mask_token.expand(batch_size, num_masked_patches, -1) # masked embedding 
        
        '''
        2. regress masked latent via regresser
        '''
        x_masked_predict = self.regresser(x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked)

        ## preserve for alignment
        if self.model_type != 'caev2':
            latent_predict = x_masked_predict 
        
        '''
        decoder for reconstruction
        '''
        if self.model_type == 'caev2':
            logits, latent_predict = self.decoder(x_masked_predict, pos_embed_masked, x_cls_token=x_cls_token, x_unmasked=x_unmasked)
            logits = logits / logits.norm(dim=-1, keepdim=True)
        else:
            logits = self.decoder(x_masked_predict, pos_embed_masked)
            logits = logits.view(-1, logits.shape[2]) # flatten

        return logits, latent_predict, latent_target

@register_model
def cae_tiny_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = ContextAutoencoderViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def cae_small_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = ContextAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def cae_base_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = ContextAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def cae_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = ContextAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
