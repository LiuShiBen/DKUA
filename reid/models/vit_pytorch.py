import os
import torch
import torch.nn as nn
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from collections import OrderedDict
from reid.models.vit import TransReID
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from reid.models.Prompt import TextPrompt
from functools import partial
import sys
import os.path as osp
import  copy
sys.path.append(osp.abspath(osp.join(__file__, '..')))

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class Encode_text_img(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.positional_embedding = clip_model.positional_embedding
        self.text_projection = clip_model.text_projection
        #self.text_projection = nn.Parameter(torch.empty(512, 768))
        self.end_id = clip_model.end_id
    def forward(self, text_tokens, img_tokens=None):
        #Original text
        text_feat = []
        for items in text_tokens:
            text_token = clip.tokenize(items["text"])
            text_token = text_token.cuda()
            text_feat.append(text_token)

        text = torch.cat(text_feat, dim=0)
        text = text.cuda()  # 64,77
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        # x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class CrossAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        head_dim = model_dim // num_heads
        self.scale = head_dim ** -0.5
        self.kv_linear = nn.Linear(model_dim, model_dim * 2, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(model_dim, model_dim)


    def forward(self, query, key_value):
        #print("query:", query.shape, key_value.shape)
        B, N, C = key_value.shape
        # Split by heads
        query_proj = query.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv_proj = self.kv_linear(key_value).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        key_proj, value_proj = kv_proj[0], kv_proj[1]   # make torchscript happy (cannot use tensor as tuple)
        attn = (query_proj @ key_proj.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value_proj).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Self_Attention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        head_dim = model_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv1_linear = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.qkv2_linear = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn1_drop = nn.Dropout(attn_drop)
        self.attn2_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(model_dim, model_dim)
        drop_path = 0.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, qkv):
        B, N, C = qkv.shape

        #first self-attention
        qkv_proj1 = self.qkv1_linear(qkv).reshape(B, N,3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query_proj1, key_proj1, value_proj1 = qkv_proj1[0], qkv_proj1[1], qkv_proj1[2]  # make torchscript happy (cannot use tensor as tuple)
        attn1 = (query_proj1 @ key_proj1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn1_drop(attn1)
        attn1 = (attn1 @ value_proj1).transpose(1, 2).reshape(B, N, C)
        attn1 = qkv + self.drop_path(attn1)
        #second self-attention
        qkv_proj2 = self.qkv2_linear(attn1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query_proj2, key_proj2, value_proj2 = qkv_proj2[0], qkv_proj2[1], qkv_proj2[2]  # make torchscript happy (cannot use tensor as tuple)
        attn2 = (query_proj2 @ key_proj2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn2_drop(attn2)
        attn2 = (attn2 @ value_proj2).transpose(1, 2).reshape(B, N, C)
        attn2 = attn2 + self.drop_path(attn1)
        x = self.proj(attn2)
        x = self.proj_drop(x)
        return x


def get_reload_weight(model_path, model, pth='ckpt_max.pth'):
    model_path = os.path.join(model_path, pth)

    load_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    # print(load_dict)
    if isinstance(load_dict, OrderedDict):
        pretrain_dict = load_dict
    else:
        pretrain_dict = load_dict['state_dicts']
        print(f"best performance {load_dict['metric']} in epoch : {load_dict['epoch']}")

    #model.load_state_dict(pretrain_dict, strict=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in pretrain_dict.items()})

    return model

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

class Normalize(nn.Module):
    def __init__(self, power=2, dim=1):
        super(Normalize, self).__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-4)
        return out

class GatingNet(nn.Module):
    def __init__(self, input_size = 384, num_task=5):
        super(GatingNet, self).__init__()
        self.linear = nn.Linear(input_size, num_task)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        return x

class Distribution_layer(nn.Module):
    def __init__(self, model, input_size = 384):
        super(Distribution_layer, self).__init__()
        self.model = model
        self.linear = nn.Linear(input_size, input_size)
        norm_layer= partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(input_size)
        self.linear = nn.Sequential(*[nn.Linear(input_size, input_size) for _ in range(5)])
    def forward(self, x):
        x_combine = []
        for i in range(5):
            x1 = self.model[i](x)
            x2 = self.norm(x1)
            x3 = self.linear[i](x2)[:, 0]
            x_combine.append(x3)
        x_combine = torch.stack(x_combine, dim=2)
        return x_combine

class build_transformer(nn.Module):
    def __init__(self, args, num_classes, arch, img_size, sie_coef, camera_num, view_num, stride_size, drop_path_rate, drop_rate,
                 attn_drop_rate, pretrain_path, hw_ratio, gem_pool, stem_conv, num_parts, has_early_feature, has_head,
                 global_feature_type, granularities, branch, enable_early_norm, **kwargs):

        super(build_transformer, self).__init__()
        self.model_name = "ViT-B-16"
        self.num_task_experts = args.num_task_experts
        self.total_experts = args.total_experts
        self.prompt_param = args.prompt_param
        self.batchsize = args.batch_size


        self.in_planes = 384
        self.num_classes = num_classes

        self.base = TransReID(img_size=img_size, patch_size=16, stride_size=stride_size, in_chans=3, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera_num, view=view_num,
                 drop_path_rate=drop_path_rate, has_early_feature=has_early_feature, sie_coef=sie_coef,
                 gem_pool=gem_pool, stem_conv=stem_conv, enable_early_norm=enable_early_norm, **kwargs)
        print("pretrain_path:", pretrain_path)
        if pretrain_path != '':
            if osp.exists(pretrain_path):
                self.base.load_param(pretrain_path, hw_ratio)
                print('Loading pretrained weights from {} ...'.format(pretrain_path))
            else:
                raise FileNotFoundError('Cannot find {}'.format(pretrain_path))
        else:
            print('Initialize weights randomly.')

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck_base = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_base.bias.requires_grad_(False)
        self.bottleneck_base.apply(weights_init_kaiming)
        self.bottleneck_task = nn.BatchNorm1d(self.in_planes)  #(self.in_planes_proj)
        self.bottleneck_task.bias.requires_grad_(False)
        self.bottleneck_task.apply(weights_init_kaiming)

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.DE = nn.ModuleList([
            nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)) for _ in range(5)
        ])

        self.Dis_layer = Distribution_layer(self.DE, input_size = 384)

        self.gate = GatingNet(num_task=5)

    def forward(self, x = None, training_step = None):

        base_feat = self.base(x)
        #print("training_step", base_feat.shape)
        task_outputs = self.Dis_layer(base_feat)   #128,384,5
        gate_output = self.gate(base_feat)  #128,5
        #print("training_step", expert_outputs.shape, gate_output.shape)
        # 计算门控网络的输出

        com_output = torch.sum(gate_output.unsqueeze(-2) * task_outputs, dim=-1)
        start = training_step-1

        #current_output = torch.sum(gate_output[:,int(start):int(end)].unsqueeze(-2) * task_outputs[:,:,int(start):int(end)], dim=-1)
        current_output =task_outputs[:, :, int(start)]
        #print("current_output", task_outputs.shape, current_output.shape)
        current_feat = self.bottleneck_base(current_output)
        finally_feat = self.bottleneck_base(com_output)
        #print("expert_outputs", expert_outputs.shape)
        if self.training:
            cls_score_fin = self.classifier(finally_feat)
            #print("expert_outputs", cls_score_fin.shape, cls_score_adapt.shape, output.shape, output_adapt.shape, expert_outputs.shape)
            return cls_score_fin, [com_output], [current_output], task_outputs#, topk_indices
        else:
            return finally_feat
            #return torch.cat([finally_feat, current_feat], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


def build_vit_backbone(num_class, args):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    model = build_transformer(args, num_class, arch='tmgf',
              img_size=[args.height, args.width], sie_coef=3.0,
              camera_num=6, view_num=0,
              stride_size=[16, 16], drop_path_rate=0.1,
              drop_rate=0.0, attn_drop_rate=0.0,
              pretrain_path=args.pretraining_path, hw_ratio=1,
              gem_pool=False, stem_conv=True, num_parts=5,
              has_head=True, global_feature_type='mean',
              granularities=[2, 3], branch="all", has_early_feature=True,
              enable_early_norm=False)

    return model




