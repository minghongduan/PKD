import argparse
import json
import warnings
import os
import time
import numpy as np
#import statistics
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import util

import models

import math

from Clinical_Dataset.dataset_PANDA_4classes_new import PANDA_feat, PANDA_feat_test
import datetime
import utliz
import random
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn import metrics
import copy

from models.model_utils import *
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer


warnings.simplefilter("ignore", UserWarning)
EVAL_THRESHOLD = 0.5
F1_SEARCH_THRESHOLDS = tuple(i / 100.0 for i in range(1, 100))
THRESHOLD_METRIC_NAMES = ("acc", "f1", "sensitivity", "specificity")
CONFUSION_COUNT_NAMES = ("tn", "fp", "fn", "tp")
ALL_THRESHOLD_METRIC_NAMES = THRESHOLD_METRIC_NAMES + CONFUSION_COUNT_NAMES


def _to_numpy_1d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def minmax_normalize_score(score):
    score_min = score.min()
    score_range = score.max() - score_min
    if score_range.item() == 0:
        return torch.zeros_like(score)
    return (score - score_min) / score_range


def cal_confusion_metrics(label, pred, threshold=EVAL_THRESHOLD):
    label = _to_numpy_1d(label).astype(np.int64)
    pred = _to_numpy_1d(pred)
    pred_label = (pred > threshold).astype(np.int64)

    tp = int(np.sum((pred_label == 1) & (label == 1)))
    tn = int(np.sum((pred_label == 0) & (label == 0)))
    fp = int(np.sum((pred_label == 1) & (label == 0)))
    fn = int(np.sum((pred_label == 0) & (label == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def cal_pr_auc(label, pred):
    label_np = _to_numpy_1d(label).astype(np.int64)
    pred_np = _to_numpy_1d(pred)

    if np.sum(label_np == 1) == 0:
        return 0.0
    precision, recall, _ = metrics.precision_recall_curve(label_np, pred_np, pos_label=1)
    return metrics.auc(recall, precision)


def cal_metrics_at_threshold(label, pred, threshold):
    label_np = _to_numpy_1d(label).astype(np.int64)
    pred_np = _to_numpy_1d(pred)
    pred_label = (pred_np > threshold).astype(np.int64)
    return {
        "acc": float(np.mean(pred_label == label_np)),
        "f1": float(metrics.f1_score(label_np, pred_label, average='macro', zero_division=0)),
        **cal_confusion_metrics(label_np, pred_np, threshold),
    }


def find_best_f1_threshold(label, pred):
    """Search validation macro-F1; ties select the smallest threshold."""
    label_np = _to_numpy_1d(label).astype(np.int64)
    pred_np = _to_numpy_1d(pred)
    if label_np.size == 0 or label_np.size != pred_np.size:
        raise ValueError("Threshold search requires non-empty, equally sized labels and predictions.")
    if not np.all((label_np == 0) | (label_np == 1)) or not np.all(np.isfinite(pred_np)):
        raise ValueError("Threshold search requires binary labels and finite prediction scores.")

    # Count all 99 thresholds from sorted scores without 99 full sklearn
    # evaluations on potentially millions of validation patches.
    # Match NumPy's scalar comparison precision and the strict > decision rule.
    thresholds = np.asarray(F1_SEARCH_THRESHOLDS, dtype=np.result_type(pred_np.dtype, np.float32))
    negative_scores = np.sort(pred_np[label_np == 0])
    positive_scores = np.sort(pred_np[label_np == 1])
    tn = np.searchsorted(negative_scores, thresholds, side='right')
    fn = np.searchsorted(positive_scores, thresholds, side='right')
    fp = negative_scores.size - tn
    tp = positive_scores.size - fn
    negative_denominator = 2 * tn + fp + fn
    positive_denominator = 2 * tp + fp + fn
    negative_f1 = np.divide(2 * tn, negative_denominator, out=np.zeros(thresholds.shape), where=negative_denominator > 0)
    positive_f1 = np.divide(2 * tp, positive_denominator, out=np.zeros(thresholds.shape), where=positive_denominator > 0)
    # Match sklearn's macro average over classes present in truth or prediction.
    class_count = (negative_denominator > 0).astype(int) + (positive_denominator > 0).astype(int)
    f1_values = (negative_f1 + positive_f1) / class_count
    best_threshold = F1_SEARCH_THRESHOLDS[int(np.argmax(f1_values))]
    return best_threshold, cal_metrics_at_threshold(label_np, pred_np, best_threshold)


def save_validation_threshold(path, threshold, validation_metrics, epoch):
    with open(path, 'w') as handle:
        json.dump({
            "threshold": float(threshold),
            "validation_f1": float(validation_metrics["f1"]),
            "f1_average": "macro",
            "epoch": int(epoch),
        }, handle, indent=2)


def load_validation_threshold(path):
    # Require the threshold saved with this checkpoint; never tune on test data.
    with open(path) as handle:
        return float(json.load(handle)["threshold"])


def print_selected_threshold_metrics(prefix, threshold, values):
    #print(f'{prefix} threshold (selected on validation) : {threshold:.2f}')
    for metric_name in ALL_THRESHOLD_METRIC_NAMES:
        value = values[metric_name]
        formatted = f'{value:d}' if metric_name in CONFUSION_COUNT_NAMES else f'{value:.6f}'
        print(f'{prefix} {metric_name} : {formatted}')


def add_selected_threshold_metrics_to_writer(writer, prefix, threshold, values, step):
    writer.add_scalar(f'{prefix}_threshold', threshold, step)
    for metric_name in ALL_THRESHOLD_METRIC_NAMES:
        writer.add_scalar(f'{prefix}_{metric_name}', values[metric_name], step)


def make_eval_result(auc, pr_auc, threshold, selected_metrics):
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "metrics": selected_metrics,
    }


def init_eval_metric_storage():
    return {
        "auc": [],
        "pr_auc": [],
        "threshold": [],
        "metrics": {metric_name: [] for metric_name in ALL_THRESHOLD_METRIC_NAMES},
    }


def append_eval_metric_storage(storage, eval_result):
    storage["auc"].append(eval_result["auc"])
    storage["pr_auc"].append(eval_result["pr_auc"])
    storage["threshold"].append(eval_result["threshold"])
    for metric_name in ALL_THRESHOLD_METRIC_NAMES:
        storage["metrics"][metric_name].append(eval_result["metrics"][metric_name])


def print_eval_metric_storage_for_fold(prefix, storage, fold_index):
    print(f'{prefix} auc : {storage["auc"][fold_index]:.6f}')
    print(f'{prefix} pr_auc : {storage["pr_auc"][fold_index]:.6f}')
    selected_metrics = {
        metric_name: storage["metrics"][metric_name][fold_index]
        for metric_name in ALL_THRESHOLD_METRIC_NAMES
    }
    print_selected_threshold_metrics(prefix, storage["threshold"][fold_index], selected_metrics)


def print_eval_metric_summary(prefix, storage):
    auc_values = np.array(storage["auc"])
    pr_auc_values = np.array(storage["pr_auc"])
    print(f'Average test {prefix} auc : {np.mean(auc_values):.6f}')
    print(f'Std test {prefix} auc : {np.std(auc_values):.6f}')
    print(f'Average test {prefix} pr_auc : {np.mean(pr_auc_values):.6f}')
    print(f'Std test {prefix} pr_auc : {np.std(pr_auc_values):.6f}')
    for metric_name in ALL_THRESHOLD_METRIC_NAMES:
        metric_values = np.array(storage["metrics"][metric_name])
        print(f'Average test {prefix} {metric_name} : {np.mean(metric_values):.6f}')
        print(f'Std test {prefix} {metric_name} : {np.std(metric_values):.6f}')

'''
Please download the pretrained CONCH model weights and put it in the folder.
'''
def load_conch(backbone_name="conch_ViT-B-16"):
    model_path = './PKD/checkpoints/CONCH/pytorch_model.bin'
    model, _ = create_model_from_pretrained(backbone_name, model_path)
    return model

class PromptLearner(nn.Module):
    def __init__(self, n_ctx=0, ctx_init="", all_ctx_trainable=True, csc=True, classnames=["Negative", "Positive", "Positive", "Positive"], conch_model='conch_ViT-B-16', p_drop_out=0.0):
        # n_ctx: number of context vectors
        # ctx_init: list of strings to initialize context vectors
        # all_ctx_trainable: whether all context vectors are trainable
        # csc: whether to use class-specific context vectors
        # classnames: list of class names
        # clip_model: clip model
        super().__init__()
        self.all_ctx_trainable = all_ctx_trainable
        n_cls = len(classnames)
       
        conch_model = load_conch(backbone_name=conch_model)
        tokenizer = get_tokenizer()
        #dtype = conch_model.dtype
        dtype = torch.float32
        ctx_dim = conch_model.text.ln_final.weight.shape[0]

        if type(ctx_init) == list:
            self.use_class_specific_ctx = True
        else:
            self.use_class_specific_ctx = False

        if ctx_init:
            if not self.use_class_specific_ctx:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                n_fixed_ctx = len(ctx_init.replace(" *", "").split(" "))
                n_learnable_ctx = ctx_init.count("*")
                prompt = tokenize(texts=ctx_init, tokenizer=tokenizer)
                with torch.no_grad():
                    embedding = conch_model.text.token_embedding(prompt[:, :-1]).type(dtype)
                num_nonzero_token = prompt.nonzero().max()
                ctx_vectors = embedding[0, 1: num_nonzero_token , :]
                prompt_prefix = ctx_init
            else:
                prompt_prefix = []
                ctx_vectors = []
                for ctx_init_i in ctx_init:
                    # use given words to initialize context vectors
                    ctx_init_i = ctx_init_i.replace("_", " ")
                    prompt_i = tokenize(texts=[ctx_init_i], tokenizer=tokenizer)
                    with torch.no_grad():
                        embedding_i = conch_model.text.token_embedding(prompt_i[:, :-1]).type(dtype)
                    num_nonzero_token = prompt_i.nonzero().max()
                    idx_special_character_i = torch.where(prompt_i == 7281)[1]
                    if all_ctx_trainable:
                        ctx_vectors_i = embedding_i[0, 1: num_nonzero_token, :]  # keep ctx between SOS and EOS
                    else:
                        ctx_vectors_i = embedding_i[0, idx_special_character_i, :]
                    prompt_prefix_i = ctx_init_i
                    prompt_prefix.append(prompt_prefix_i)
                    ctx_vectors.append(ctx_vectors_i)

        else:
            # random initialization
            if csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        if not self.use_class_specific_ctx:
            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        else:
            self.ctx = nn.ParameterList([nn.Parameter(ctx_vector) for ctx_vector in ctx_vectors])

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(tokenize(texts=[name], tokenizer=tokenizer)) for name in classnames]
        if not self.use_class_specific_ctx:
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            prompts = [prompt_prefix_i + " " + name + "." for prompt_prefix_i, name in zip(prompt_prefix, classnames)]

        tokenized_prompts = torch.cat([tokenize(texts=[p], tokenizer=tokenizer) for p in prompts])
        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts[:, :-1]).type(dtype)
            cls_mask = conch_model.text.build_cls_mask(tokenized_prompts[:, :-1], dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        if not self.use_class_specific_ctx:
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_ctx = n_ctx
        else:
            if all_ctx_trainable:
                self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            else:
                for i in range(n_cls):
                    special_character_pos = torch.where(tokenized_prompts[i] == 7281)[0].min()  # find first character "*"
                    self.register_buffer("token_prefix_{}".format(i), embedding[i, :special_character_pos, :])
            for i in range(n_cls):
                CLS_pos = torch.where(tokenized_prompts[i] == 7281)[0].max()  # find last character "*"
                self.register_buffer("token_suffix_{}".format(i), embedding[i, CLS_pos+1:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.cls_mask = cls_mask
        self.drop_layer = torch.nn.Dropout(p=p_drop_out)

    def forward(self):
        if not self.use_class_specific_ctx:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        else:
            ctx = self.ctx
            prompts = []
            for i in range(self.n_cls):
                if self.all_ctx_trainable:
                    prompt_i = torch.cat(
                        [
                            self.token_prefix[i].unsqueeze(0),  # (n_cls, 1, dim)
                            ctx[i].unsqueeze(0),  # (n_cls, n_ctx, dim)
                            getattr(self, "token_suffix_{}".format(i)).unsqueeze(0)  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )
                else:
                    prompt_i = torch.cat(
                        [
                            getattr(self, "token_prefix_{}".format(i)).unsqueeze(0),  # (n_cls, 1, dim)
                            ctx[i].unsqueeze(0),  # (n_cls, n_ctx, dim)
                            getattr(self, "token_suffix_{}".format(i)).unsqueeze(0)  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )
                prompts.append(prompt_i)
            prompts = torch.cat(prompts, dim=0)
            prompts = self.drop_layer(prompts)
        return prompts


class TextEncoder(nn.Module):
    def __init__(self, conch_model):
        super().__init__()
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        self.attn_mask = conch_model.text.attn_mask
        self.cls_emb = conch_model.text.cls_emb
        self.build_cls_mask = conch_model.text.build_cls_mask
        self._repeat = conch_model.text._repeat
        self.dtypec = torch.float32
        
        #self.dtype = torch.float64
    def forward(self, prompts, cls_mask):
        x = prompts
        seq_len = x.shape[1]
        seq_len += 1
        x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
        #cls_mask = self.build_cls_mask(x, self.dtypec)
        attn_mask = self.attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(self.dtypec)
        #x = prompts + self.positional_embedding.type(self.dtypec)
        x = x.permute(1, 0, 2)  # NLD -> LND
        #x = self.transformer(x, self.attn_mask.to("cuda"))
        x = self.transformer(x, attn_mask=attn_mask.to("cuda"))
        x = x.permute(1, 0, 2)  # LND -> NLD
        #pooled, tokens = x[:, -1], x[:, :-1]
        #pooled = self.ln_final(pooled)
        x = self.ln_final(x[:, -1])
        #pooled = pooled[torch.arange(pooled.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        #x = F.normalize(pooled, dim=-1)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # argmax = tokenized_prompts.argmax(dim=-1)
        x = x @ self.text_projection
        #x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        #x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = F.normalize(x, dim=-1)
        return x

class feat_projecter(nn.Module):
    def __init__(self, input_feat_dim=512, output_feat_dim=512):
        super(feat_projecter, self).__init__()

        self.projecter = nn.Sequential(
            nn.Linear(input_feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_feat_dim)
        )
        self.dtype = torch.float64

    def forward(self, x):
        x = self.projecter(x)
        return x


def panda_feat_projecter(input_dim, output_dim):
    model = feat_projecter(input_dim, output_dim)
    return model

class Feat_predictor_student(nn.Module):
    def __init__(self, prompt_learner, num_feat=512):
        super(Feat_predictor_student, self).__init__()
        #self.projector = nn.Sequential(nn.Linear(num_feat, 512),
        #                    nn.LayerNorm(normalized_shape=512))

        self.prompt_learner = prompt_learner
        conch_model = load_conch(backbone_name="conch_ViT-B-16")
        self.text_encoder = TextEncoder(conch_model)
        self.logit_scale = conch_model.logit_scale
        self.dtype = torch.float64
        #self.predictor = nn.Sequential(nn.Linear(512, 2))

    def forward(self, x):
        instance_prompts = self.prompt_learner()
        instance_text_features = self.text_encoder(instance_prompts, self.prompt_learner.cls_mask)

        return x.type(self.dtype), instance_text_features.type(self.dtype)


class Feat_predictor_teacher(nn.Module):
    def __init__(self, prompt_learner, num_feat=512):
        super(Feat_predictor_teacher, self).__init__()
        #self.projector = nn.Sequential(nn.Linear(num_feat, 512),
        #                    nn.LayerNorm(normalized_shape=512))

        self.prompt_learner = prompt_learner
        conch_model = load_conch(backbone_name="conch_ViT-B-16")
        self.text_encoder = TextEncoder(conch_model)
        self.logit_scale = conch_model.logit_scale
        self.dtype = torch.float64
        #self.predictor = nn.Sequential(nn.Linear(512, 2))

    def forward(self, x):
        bag_prompts = self.prompt_learner()
        bag_text_features = self.text_encoder(bag_prompts, self.prompt_learner.cls_mask)

        return x, bag_text_features.type(self.dtype)


class Text_attention(nn.Module):
    def __init__(self, num_classes=[2], init=True, withoutAtten=False, input_feat_dim=512):
        super(Text_attention, self).__init__()
        self.input_feat_dim = input_feat_dim
        self.withoutAtten=withoutAtten

        self.L = 512
        self.D = 256
        self.K = 1

        self.attention = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(input_feat_dim, 512),
                            nn.LayerNorm(normalized_shape=512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(512, 256),
                            nn.LayerNorm(normalized_shape=256),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(256, 512),
                            nn.LayerNorm(normalized_shape=512),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.L, self.D),
                            nn.Tanh(),
                            nn.Linear(self.D, self.K)
        )
        
        self.headcount = len(num_classes)
        self.return_features = False
        
        if len(num_classes) == 1:
            self.top_layer = nn.Linear(512, num_classes[0])
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(2048, i))
            self.top_layer = None  # this way headcount can act as switch.
        
        if init:
            self._initialize_weights()

    def forward(self, x, xo, returnBeforeSoftMaxA=False, scores_replaceAS=None):
        x = x.view(x.size(0), self.input_feat_dim)
        #x = self.classifier(x)
        #x = self.classifier2(x)
        # Attention module
        A_ = self.attention(x)  # NxK
        #A_ = A_[:, 1].unsqueeze(0)
        A_ = torch.transpose(A_, 1, 0)  # KxN
        A = F.softmax(A_, dim=1)  # softmax over N

        if scores_replaceAS is not None:
            A_ = scores_replaceAS
            A = F.softmax(A_, dim=1)  # softmax over N

        if self.withoutAtten:
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            x = torch.mm(A, xo)  # KxL

        if self.return_features: # switch only used for CIFAR-experiments
            return x

        #x = self.top_layer(x)
        if returnBeforeSoftMaxA:
            return x, 0, A, A_
        return x, 0, A

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class map_abmil(nn.Module):
    def __init__(self, model):
        super(map_abmil, self).__init__()
        self.model_gleason5 = model
        #self.model_gleason4 = model
        #self.model_gleason5 = model
        self.dtype = torch.float64
        self.dtypec = torch.float32
        self.input_dim = 512
        self.logit_scale = self.input_dim ** -0.5
        self.cross_attention = MultiheadAttention(embed_dim=self.input_dim, num_heads=1)
        self.to_q = nn.Linear(self.input_dim, self.input_dim)
        self.to_k = nn.Linear(self.input_dim, self.input_dim)
        self.to_v = nn.Linear(self.input_dim, self.input_dim)
        self.norm = nn.LayerNorm(normalized_shape=512)
    
    def forward(self, x, concept_feats_student):
        #sim_scores_teacher = x @ concept_feats_student.type(self.dtypec).T
        '''
        q = self.to_q(x)
        k = self.to_k(concept_feats_student.type(self.dtypec))
        v = self.to_v(concept_feats_student.type(self.dtypec))
        '''
        xo = x.type(self.dtypec)
        q = x.unsqueeze(1).type(self.dtypec)
        k = concept_feats_student.type(self.dtypec)
        v = concept_feats_student.type(self.dtypec)
        x, _ = self.cross_attention(q, k, v) 

        x = self.norm(x + q)

        bag_prediction_gleason5, _, _, instance_attn_score_gleason5 = self.model_gleason5(x.squeeze(1), xo, returnBeforeSoftMaxA=True, scores_replaceAS=None)

        bag_prediction = [bag_prediction_gleason5.type(self.dtype)]
        instance_attn_score = [instance_attn_score_gleason5.type(self.dtype)]

        return bag_prediction, instance_attn_score

    def get_loss(self, output_bag, bag_label):
        # output shape: 1
        # label shape: 1
        output_bag = output_bag[0, 3]
        bag_label = bag_label.squeeze()
        loss = -1. * (bag_label * torch.log(output_bag+1e-5) + (1. - bag_label) * torch.log(1. - output_bag+1e-5))
        return loss


class map_dsmil(nn.Module):
    def __init__(self, model):
        super(map_dsmil, self).__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        instance_attn_score, bag_prediction, _, _ = self.model(x)
        instance_attn_score = torch.softmax(instance_attn_score, dim=1)
        instance_attn_score = instance_attn_score[:, 1].unsqueeze(0)
        return bag_prediction, instance_attn_score

    def get_loss(self, output_bag, output_inst, bag_label):
        # output shape: 1
        # label shape: 1
        max_id = torch.argmax(output_inst.squeeze(0))
        bag_pred_byMax = output_inst.squeeze(0)[max_id]
        bag_loss = self.criterion(output_bag, bag_label)

        bag_label = bag_label.squeeze()
        bag_loss_byMax = -1. * (bag_label * torch.log(bag_pred_byMax+1e-5) + (1. - bag_label) * torch.log(1. - bag_pred_byMax+1e-5))
        loss = 0.5 * bag_loss + 0.5 * bag_loss_byMax
        return loss


class Optimizer:
    def __init__(self, model_encoder, model_student, model_teacher, model_teacherHead, model_teacherHead_distill, train_bagloader, train_instanceloader, val_bagloader, val_instanceloader, test_bagloader, test_instanceloader, n_epochs, lr, fold_id, weight_decay=1e-5, ckpt_dir="./results_CAMELYON_checkpoints", resume=''):
        self.num_epochs = n_epochs
        self.lr = lr
        self.lr_schedule = lambda epoch: ((epoch < 9999) * (self.lr * (0.5 ** (epoch // args.lrdrop)))
                                          + (epoch >= 9999) * self.lr * 0.1 ** 3)

        self.momentum = 0.9
        self.weight_decay = weight_decay
        self.checkpoint_dir = ckpt_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.smoothE = 0
        self.log_period = 10
        self.resume = resume
        self.writer = None
        self.StuFilterType = 'PseudoBag_95_05_2'
        # model stuff

        self.model_encoder = model_encoder
        self.model_student = model_student
        self.model_teacher = model_teacher
        self.model_teacherHead = model_teacherHead
        self.model_teacherHead_distill = model_teacherHead_distill
        #self.teacher_pseudo_label_merge_weight = 0.5
        self.stu_loss_weight_neg = 0.1

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nmodel_gpus = len(args.modeldevice)
        #self.pseudo_loader = t_loader  # can also be DataLoader with less aug.
        self.train_loader = train_instanceloader
        self.val_loader = val_instanceloader  # evaluating dataloader(could be None)
        self.test_loader = test_instanceloader
        
        self.train_bagloader = train_bagloader
        self.val_bagloader = val_bagloader
        self.test_bagloader = test_bagloader

        self.dtype = torch.float64 if not args.cpu else np.float64
        
        # activations of previous to last layer to be saved if using multiple heads.

        self.bag_auc_best_gleason5 = 0
        self.instance_auc_best_gleason5 = 0
        
        self.fold_id = fold_id

        self.bag_auc_best_encoder_ckpt_name_gleason5 = 'bag_auc_best_encoder_ckpt_gleason5_{}.pt'.format(self.fold_id)
        self.bag_auc_best_teacher_ckpt_name_gleason5 = 'bag_auc_best_teacher_ckpt_gleason5_{}.pt'.format(self.fold_id)
        self.bag_auc_best_teacherhead_ckpt_name_gleason5 = 'bag_auc_best_teacherhead_ckpt_gleason5_{}.pt'.format(self.fold_id)
        self.bag_auc_best_teacherhead_distill_ckpt_name_gleason5 = 'bag_auc_best_teacherhead_distill_ckpt_gleason5_{}.pt'.format(self.fold_id)
        self.bag_auc_best_student_ckpt_name_gleason5 = 'bag_auc_best_student_ckpt_gleason5_{}.pt'.format(self.fold_id)
        self.bag_auc_best_threshold_name_gleason5 = 'bag_auc_best_threshold_gleason5_{}.json'.format(self.fold_id)
        self.instance_auc_best_encoder_ckpt_name_gleason5 = 'instance_auc_best_encoder_ckpt_gleason5_{}.pt'.format(self.fold_id)
        self.instance_auc_best_teacher_ckpt_name_gleason5 = 'instance_auc_best_teacher_ckpt_gleason5_{}.pt'.format(self.fold_id)
        self.instance_auc_best_teacherhead_ckpt_name_gleason5 = 'instance_auc_best_teacherhead_ckpt_gleason5_{}.pt'.format(self.fold_id)
        self.instance_auc_best_teacherhead_distill_ckpt_name_gleason5 = 'instance_auc_best_teacherhead_distill_ckpt_gleason5_{}.pt'.format(self.fold_id)
        self.instance_auc_best_student_ckpt_name_gleason5 = 'instance_auc_best_student_ckpt_gleason5_{}.pt'.format(self.fold_id)
        self.instance_auc_best_threshold_name_gleason5 = 'instance_auc_best_threshold_gleason5_{}.json'.format(self.fold_id)
        
        self.network_logger = utliz.Network_Logger(model=model_student)

        #self.num_teacher = 1
        #self.num_teacher = len(model_teacherHead)
        self.tokenizer = get_tokenizer()

        self.optimizer_encoder = torch.optim.SGD(self.model_encoder.parameters(), weight_decay=self.weight_decay, momentum=self.momentum, lr=self.lr)

        self.optimizer_teacherHead = torch.optim.SGD(self.model_teacherHead.parameters(), weight_decay=self.weight_decay, momentum=self.momentum, lr=self.lr)
        
        self.optimizer_teacherHead_distill = torch.optim.SGD(self.model_teacherHead_distill.parameters(), weight_decay=self.weight_decay, momentum=self.momentum, lr=self.lr)

        for param in self.model_student.text_encoder.parameters():
            param.requires_grad = False
        self.optimizer_student = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model_student.parameters()),
                                    weight_decay=self.weight_decay,
                                    momentum=self.momentum,
                                    lr=self.lr)

        for param in self.model_teacher.text_encoder.parameters():
            param.requires_grad = False
        self.optimizer_teacher = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model_teacher.parameters()),
                                    weight_decay=self.weight_decay,
                                    momentum=self.momentum,
                                    lr=self.lr)

    def optimize_teacher(self, epoch):
        self.model_encoder.train()
        self.model_teacher.train()
        self.model_teacherHead.train()
        self.model_teacherHead_distill.train()
        self.model_student.eval()
        ## optimize teacher with bag-dataloader
        # 1. change loader to bag-loader
        loader = self.train_bagloader
        # 2. optimize
        patch_label_gt_gleason5 = []
        bag_label_gt_gleason5 = []
        patch_label_pred_gleason5 = []
        bag_label_pred_gleason5 = []
        patch_corresponding_bag_label_gleason5 = []
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Teacher training')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            feat = self.model_encoder(data.squeeze(0))

            if epoch > self.smoothE:
                with torch.no_grad():
                    final_student, concept_feats_student = self.model_student(feat)
                    sim_scores_student = final_student @ concept_feats_student.T * self.model_student.logit_scale.exp()
                    #pred_byStudent = torch.softmax(sim_scores_student, dim=1)[:, 3]
                    pred_byStudent = torch.sigmoid(sim_scores_student)[:, 3]

                if '_Top' in self.StuFilterType:
                    # strategy A: remove the topK most negative instance
                    idx_to_keep = torch.topk(-pred_byStudent, k=int(self.StuFilterType.split('_Top')[-1]))[1]
                elif '_ThreProb' in self.StuFilterType:
                    # strategy B: remove the negative instance above prob K
                    idx_to_keep = torch.where(pred_byStudent <= int(self.StuFilterType.split('_ThreProb')[-1]) / 100.0)[0]
                    if idx_to_keep.shape[0] == 0:  # if all instance are dropped, keep the most positive one
                        idx_to_keep = torch.topk(pred_byStudent, k=1)[1]
                elif 'PseudoBag' in self.StuFilterType:
                    hardPos_hardNeg_thresholds = [int(self.StuFilterType.split('_')[1]) / 100.0,
                                                  int(self.StuFilterType.split('_')[2]) / 100.0]

                    idx_easy_pos = torch.where(pred_byStudent >= hardPos_hardNeg_thresholds[0])[0]
                    # if idx_easy_pos.shape[0] == 0:  # if all instance are dropped, keep the most Positive one
                    #     idx_easy_pos = torch.topk(pred_byStudent, k=1)[1]

                    idx_hard_pos = torch.where((pred_byStudent < hardPos_hardNeg_thresholds[0]) & (pred_byStudent >= 0.5))[0]
                    # if idx_hard_pos.shape[0] == 0:  # if all instance are dropped, keep the most Positive one
                    #     idx_hard_pos = torch.topk(pred_byStudent, k=1)[1]

                    idx_easy_neg = torch.where(pred_byStudent <= hardPos_hardNeg_thresholds[-1])[0]
                    # if idx_easy_neg.shape[0] == 0:  # if all instance are dropped, keep the most Negative one
                    #     idx_easy_neg = torch.topk(-1 * pred_byStudent, k=1)[1]

                    idx_hard_neg = torch.where((pred_byStudent > hardPos_hardNeg_thresholds[-1]) & (pred_byStudent < 0.5))[0]
                    # if idx_hard_neg.shape[0] == 0:  # if all instance are dropped, keep the most Negative one
                    #     idx_hard_neg = torch.topk(-1 * pred_byStudent, k=1)[1]

                    # split a bag into 4 part: 1: easy neg, 2: hard neg, 3: hard pos, 4: easy pos

                    pseudo_bag_easy_pos = feat[idx_easy_pos]
                    pseudo_bag_hard_pos = feat[idx_hard_pos]
                    pseudo_bag_easy_neg = feat[idx_easy_neg]
                    pseudo_bag_hard_neg = feat[idx_hard_neg]
 
                    pseudo_bag_013 = torch.cat([pseudo_bag_easy_pos, pseudo_bag_hard_pos, pseudo_bag_hard_neg], dim=0) 
 
                    pseudo_bag_123 = torch.cat([pseudo_bag_hard_pos, pseudo_bag_easy_neg, pseudo_bag_hard_neg], dim=0)  

                    pseudo_bag_0123 = torch.cat([pseudo_bag_easy_pos, pseudo_bag_hard_pos, pseudo_bag_easy_neg, pseudo_bag_hard_neg], dim=0)


                feat_PseudoBag = [pseudo_bag_013, pseudo_bag_123, pseudo_bag_0123] 

                bag_pseudo_feat_distill = []

                loss_teacher = 0
                for feat_new_i in feat_PseudoBag:
                    if feat_new_i.shape[0] == 0:  # if drop all, don't drop
                        feat_new_i = feat

                    #bag_features_teacher_i, instance_attn_score_teacher_i = self.model_teacherHead(feat_new_i, concept_feats_student.detach())
                    bag_features_teacher_i, instance_attn_score_teacher_i = self.model_teacherHead(feat_new_i, concept_feats_student.detach())
                    
                    bag_pseudo_feat_distill.append(bag_features_teacher_i[0])
                    
                    _, concept_feats_teacher = self.model_teacher(bag_features_teacher_i)
                    
                    sim_scores_teacher_i_gleason5 = bag_features_teacher_i[0] @ concept_feats_teacher.detach().T * self.model_teacher.logit_scale.exp()

                    bag_prediction_teacher_i_gleason5 = torch.sigmoid(sim_scores_teacher_i_gleason5)
                    
                    loss_teacher_i_gleason5 = self.model_teacherHead.get_loss(bag_prediction_teacher_i_gleason5, label[1].squeeze(0)[2])
                    loss_teacher = loss_teacher + loss_teacher_i_gleason5

                    patch_label_pred_gleason5.append(instance_attn_score_teacher_i[0].detach().squeeze(0))
                    bag_label_pred_gleason5.append(bag_prediction_teacher_i_gleason5.detach()[0, 3])

                self.optimizer_encoder.zero_grad()
                self.optimizer_teacher.zero_grad()
                self.optimizer_teacherHead.zero_grad()
                loss_teacher.backward(retain_graph=True)
                bag_pseudo_feat_distill = torch.cat(bag_pseudo_feat_distill, dim=-2)  ### numGroup x fs 

                bag_features_teacher_distill, _ = self.model_teacherHead_distill(bag_pseudo_feat_distill, concept_feats_teacher.detach())

                sim_scores_teacher_distill = bag_features_teacher_distill[0] @ concept_feats_teacher.T * self.model_teacher.logit_scale.exp()
                #bag_prediction_teacher_distill = torch.softmax(sim_scores_teacher_distill, 1)
                bag_prediction_teacher_distill = torch.sigmoid(sim_scores_teacher_distill)
                loss_teacher_distill = self.model_teacherHead_distill.get_loss(bag_prediction_teacher_distill, label[1].squeeze(0)[2])

                self.optimizer_teacherHead_distill.zero_grad()
                loss_teacher_distill.backward()
                
                self.optimizer_encoder.step()
                self.optimizer_teacher.step()
                self.optimizer_teacherHead.step()
                self.optimizer_teacherHead_distill.step()
                
            else:
                with torch.no_grad():
                    _, concept_feats_student = self.model_student(feat)

                bag_features_teacher, instance_attn_score_teacher = self.model_teacherHead(feat, concept_feats_student.detach())
                #bag_features_teacher, instance_attn_score_teacher = self.model_teacherHead(feat, concept_feats_student.detach())
                _, concept_feats_teacher = self.model_teacher(bag_features_teacher)
                
                sim_scores_teacher = bag_features_teacher[0] @ concept_feats_teacher.detach().T * self.model_teacher.logit_scale.exp()

                bag_prediction_teacher_gleason5 = torch.sigmoid(sim_scores_teacher)
                
                loss_teacher_gleason5 = self.model_teacherHead.get_loss(bag_prediction_teacher_gleason5, label[1].squeeze(0)[2])

                patch_label_pred_gleason5.append(instance_attn_score_teacher[0].detach().squeeze(0))
                bag_label_pred_gleason5.append(bag_prediction_teacher_gleason5.detach()[0, 3])

                loss_teacher = loss_teacher_gleason5

                self.optimizer_encoder.zero_grad()
                self.optimizer_teacher.zero_grad()
                self.optimizer_teacherHead.zero_grad()
                loss_teacher.backward()
                self.optimizer_encoder.step()
                self.optimizer_teacher.step()
                self.optimizer_teacherHead.step()

            patch_label_gt_gleason5.append(label[0].squeeze(0)[:, 2])
            bag_label_gt_gleason5.append(label[1].squeeze(0)[2])
            patch_corresponding_bag_label_gleason5.append(torch.ones([data.shape[1]]).to(self.dev)*label[1].squeeze(0)[2])
   
            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Teacher', loss_teacher.item(), niter)

        patch_label_pred_gleason5 = torch.cat(patch_label_pred_gleason5) 
        bag_label_pred_gleason5 = torch.tensor(bag_label_pred_gleason5)
        patch_label_gt_gleason5 = torch.cat(patch_label_gt_gleason5)
        bag_label_gt_gleason5 = torch.tensor(bag_label_gt_gleason5)
        patch_corresponding_bag_label_gleason5 = torch.cat(patch_corresponding_bag_label_gleason5)

        self.estimated_AttnScore_norm_para_min_gleason5 = patch_label_pred_gleason5.min()
        self.estimated_AttnScore_norm_para_max_gleason5 = patch_label_pred_gleason5.max() 

        return 0
    
    def norm_AttnScore2Prob_gleason5(self, attn_score):
        prob = (attn_score - self.estimated_AttnScore_norm_para_min_gleason5) / \
               (self.estimated_AttnScore_norm_para_max_gleason5 - self.estimated_AttnScore_norm_para_min_gleason5).clamp(min=1e-8)
        return prob

    def optimize_student(self, epoch):
        self.model_teacher.eval()
        self.model_teacherHead.eval()
        self.model_teacherHead_distill.eval()
        self.model_encoder.train()
        self.model_student.train()
        ## optimize teacher with instance-dataloader
        # 1. change loader to instance-loader
        loader = self.train_loader
        # 2. optimize
        patch_label_gt_gleason5 = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred_gleason5 = torch.zeros([loader.dataset.__len__(), 1]).type(torch.float64).to(self.dev)
        bag_label_gt_gleason5 = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)

        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student training')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            # get teacher output of instance
            feat = self.model_encoder(data)

            final_student, concept_feats_student = self.model_student(feat)
       
            sim_scores_student_gleason5 = (final_student @ concept_feats_student.T * self.model_student.logit_scale.exp())

            patch_prediction_gleason5 = torch.sigmoid(sim_scores_student_gleason5)

            with torch.no_grad():
                _, instance_attn_score_teacher = self.model_teacherHead(feat, concept_feats_student.detach())
            
            pseudo_instance_label_gleason5 = self.norm_AttnScore2Prob_gleason5(instance_attn_score_teacher[0]).clamp(min=1e-5, max=1-1e-5).squeeze(0)

            # set true negative patch label to [1, 0]
            pseudo_instance_label_gleason5[label[1].squeeze(0)[:, 2] == 0] = 0
            ###pseudo_instance_label_gleason4[label[1].squeeze(0)[:, 1] == 0] = 0
            ###pseudo_instance_label_gleason5[label[1].squeeze(0)[:, 2] == 0] = 0
            # # DEBUG: Assign GT patch label
            # pseudo_instance_label = label[0]
            # get student output of instance

            train_loss_neg_gleason5 = -1. * torch.mean(self.stu_loss_weight_neg * (1-pseudo_instance_label_gleason5) * torch.log(1 - patch_prediction_gleason5[:, 3] + 1e-5))
            train_loss_pos_gleason5 = -1. * torch.mean((1-self.stu_loss_weight_neg) * pseudo_instance_label_gleason5 * torch.log(patch_prediction_gleason5[:, 3] + 1e-5))
            # cal loss
            loss_student_gleason5 = -1. * torch.mean(self.stu_loss_weight_neg * (1-pseudo_instance_label_gleason5) * torch.log(1 - patch_prediction_gleason5[:, 3] + 1e-5) +
                                            (1-self.stu_loss_weight_neg) * pseudo_instance_label_gleason5 * torch.log(patch_prediction_gleason5[:, 3] + 1e-5))

            loss_student = loss_student_gleason5
            self.optimizer_encoder.zero_grad()
            self.optimizer_student.zero_grad()
            loss_student.backward()
            self.optimizer_encoder.step()
            self.optimizer_student.step()
            
            patch_corresponding_slide_idx[selected, 0] = label[2]
            
            patch_label_pred_gleason5[selected, 0] = patch_prediction_gleason5.detach()[:, 3]
            patch_label_gt_gleason5[selected, 0] = label[0][:, 2]
            bag_label_gt_gleason5[selected, 0] = label[1][:, 2]

            if niter % self.log_period == 0:
                self.writer.add_scalar('train_loss_Student', loss_student.item(), niter)
                self.writer.add_scalar('train_loss_neg_Student_gleason5', train_loss_neg_gleason5.item(), niter)
                self.writer.add_scalar('train_loss_pos_Student_gleason5', train_loss_pos_gleason5.item(), niter)

        instance_auc_ByStudent_gleason5 = utliz.cal_auc(patch_label_gt_gleason5.reshape(-1), patch_label_pred_gleason5.reshape(-1))
        self.writer.add_scalar('train_instance_AUC_byStudent_gleason5', instance_auc_ByStudent_gleason5, epoch)

        # cal bag-level auc

        bag_label_gt_coarse_gleason5 = []
        bag_label_prediction_gleason5 = []
        available_bag_idx = patch_corresponding_slide_idx.unique()
        for bag_idx_i in available_bag_idx:
            idx_same_bag_i = torch.where(patch_corresponding_slide_idx == bag_idx_i)
            if bag_label_gt_gleason5[idx_same_bag_i].max() != bag_label_gt_gleason5[idx_same_bag_i].max():
                raise

            bag_label_gt_coarse_gleason5.append(bag_label_gt_gleason5[idx_same_bag_i].max())
            bag_label_prediction_gleason5.append(patch_label_pred_gleason5[idx_same_bag_i].max())

        bag_label_gt_coarse_gleason5 = torch.tensor(bag_label_gt_coarse_gleason5)
        bag_label_prediction_gleason5 = torch.tensor(bag_label_prediction_gleason5)
        bag_auc_ByStudent_gleason5 = utliz.cal_auc(bag_label_gt_coarse_gleason5.reshape(-1), bag_label_prediction_gleason5.reshape(-1))
        self.writer.add_scalar('train_bag_AUC_byStudent_gleason5', bag_auc_ByStudent_gleason5, epoch)

        return 0

    def evaluate_teacher(self, epoch):
        self.model_encoder.eval()
        self.model_teacher.eval()
        self.model_teacherHead.eval()
        self.model_teacherHead_distill.eval()
        self.model_student.eval()
        ## optimize teacher with bag-dataloader
        # 1. change loader to bag-loader
        loader = self.val_bagloader
        # 2. optimize

        bag_label_gt_gleason5 = []

        bag_label_prediction_withAttnScore_gleason5 = []
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Teacher evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)
            with torch.no_grad():
                feat = self.model_encoder(data.squeeze(0))
                final_student, concept_feats_student = self.model_student(feat)
                sim_scores_student = final_student @ concept_feats_student.T * self.model_student.logit_scale.exp()
                #pred_byStudent = torch.softmax(sim_scores_student, dim=1)[:, 3]
                pred_byStudent = torch.sigmoid(sim_scores_student)[:, 3]

                hardPos_hardNeg_thresholds = [int(self.StuFilterType.split('_')[1]) / 10000.0,
                                                int(self.StuFilterType.split('_')[2]) / 100.0]

                idx_easy_pos = torch.where(pred_byStudent >= hardPos_hardNeg_thresholds[0])[0]
                # if idx_easy_pos.shape[0] == 0:  # if all instance are dropped, keep the most Positive one
                #     idx_easy_pos = torch.topk(pred_byStudent, k=1)[1]

                idx_hard_pos = torch.where((pred_byStudent < hardPos_hardNeg_thresholds[0]) & (pred_byStudent >= 0.5))[0]
                # if idx_hard_pos.shape[0] == 0:  # if all instance are dropped, keep the most Positive one
                #     idx_hard_pos = torch.topk(pred_byStudent, k=1)[1]

                idx_easy_neg = torch.where(pred_byStudent <= hardPos_hardNeg_thresholds[-1])[0]
                # if idx_easy_neg.shape[0] == 0:  # if all instance are dropped, keep the most Negative one
                #     idx_easy_neg = torch.topk(-1 * pred_byStudent, k=1)[1]

                idx_hard_neg = torch.where((pred_byStudent > hardPos_hardNeg_thresholds[-1]) & (pred_byStudent < 0.5))[0]
                # if idx_hard_neg.shape[0] == 0:  # if all instance are dropped, keep the most Negative one
                #     idx_hard_neg = torch.topk(-1 * pred_byStudent, k=1)[1]

                # split a bag into 4 part: 1: easy neg, 2: hard neg, 3: hard pos, 4: easy pos

                pseudo_bag_easy_pos = feat[idx_easy_pos]
                pseudo_bag_hard_pos = feat[idx_hard_pos]
                pseudo_bag_easy_neg = feat[idx_easy_neg]
                pseudo_bag_hard_neg = feat[idx_hard_neg]

                pseudo_bag_013 = torch.cat([pseudo_bag_easy_pos, pseudo_bag_hard_pos, pseudo_bag_hard_neg], dim=0) 
 
                pseudo_bag_123 = torch.cat([pseudo_bag_hard_pos, pseudo_bag_easy_neg, pseudo_bag_hard_neg], dim=0)  

                pseudo_bag_0123 = torch.cat([pseudo_bag_easy_pos, pseudo_bag_hard_pos, pseudo_bag_easy_neg, pseudo_bag_hard_neg], dim=0)

                # hard group
                #feat_PseudoBag = [pseudo_bag_012, pseudo_bag_013, pseudo_bag_023, pseudo_bag_123, pseudo_bag_0123] 
                feat_PseudoBag = [pseudo_bag_013, pseudo_bag_123, pseudo_bag_0123] 

                bag_pseudo_feat_distill = []
                for feat_new_i in feat_PseudoBag:
                    if feat_new_i.shape[0] == 0:  # if drop all, don't drop
                        feat_new_i = feat

                    #bag_features_teacher_i, instance_attn_score_teacher_i = self.model_teacherHead(feat_new_i, concept_feats_student.detach())
                    
                    bag_features_teacher_i, instance_attn_score_teacher_i = self.model_teacherHead(feat_new_i, concept_feats_student.detach())
                    
                    bag_pseudo_feat_distill.append(bag_features_teacher_i[0])

                bag_pseudo_feat_distill = torch.cat(bag_pseudo_feat_distill, dim=-2)  ### numGroup x fs 
                
                _, concept_feats_teacher_distill = self.model_teacher(bag_pseudo_feat_distill)
                
                bag_features_teacher_distill, _ = self.model_teacherHead_distill(bag_pseudo_feat_distill, concept_feats_teacher_distill.detach())

                sim_scores_teacher_distill = bag_features_teacher_distill[0] @ concept_feats_teacher_distill.T * self.model_teacher.logit_scale.exp()
                #bag_prediction_teacher_distill = torch.softmax(sim_scores_teacher_distill, 1)
                bag_prediction_teacher_distill = torch.sigmoid(sim_scores_teacher_distill)

                bag_label_prediction_withAttnScore_gleason5.append(bag_prediction_teacher_distill.detach()[0, 3])

            bag_label_gt_gleason5.append(label[1].squeeze(0)[2])

        bag_label_prediction_withAttnScore_gleason5 = torch.tensor(bag_label_prediction_withAttnScore_gleason5)

        bag_label_gt_gleason5 = torch.tensor(bag_label_gt_gleason5)

        #patch_label_pred_normed_gleason3 = (patch_label_pred_gleason3 - patch_label_pred_gleason3.min()) / (patch_label_pred_gleason3.max() - patch_label_pred_gleason3.min())
        #instance_auc_ByTeacher_gleason3 = utliz.cal_auc(patch_label_gt_gleason3.reshape(-1), patch_label_pred_normed_gleason3.reshape(-1))
        bag_auc_ByTeacher_withAttnScore_gleason5 = utliz.cal_auc(
            bag_label_gt_gleason5.reshape(-1),
            bag_label_prediction_withAttnScore_gleason5.reshape(-1),
        )
        bag_pr_auc = cal_pr_auc(
            bag_label_gt_gleason5.reshape(-1),
            bag_label_prediction_withAttnScore_gleason5.reshape(-1),
        )
        bag_threshold, bag_validation_metrics = find_best_f1_threshold(
            bag_label_gt_gleason5.reshape(-1),
            bag_label_prediction_withAttnScore_gleason5.reshape(-1),
        )

        if bag_auc_ByTeacher_withAttnScore_gleason5 >= self.bag_auc_best_gleason5:
            print(f'Validation gleason5 bag auc increased ({self.bag_auc_best_gleason5:.6f} --> {bag_auc_ByTeacher_withAttnScore_gleason5:.6f}).  Saving model ...')
            print_selected_threshold_metrics(
                f'Fold{self.fold_id} Validation bag', bag_threshold, bag_validation_metrics
            )
            self.bag_auc_best_gleason5 = bag_auc_ByTeacher_withAttnScore_gleason5
            torch.save(self.model_encoder.state_dict(), os.path.join(self.checkpoint_dir, self.bag_auc_best_encoder_ckpt_name_gleason5)) 
            torch.save(self.model_teacher.state_dict(), os.path.join(self.checkpoint_dir, self.bag_auc_best_teacher_ckpt_name_gleason5)) 
            torch.save(self.model_teacherHead.state_dict(), os.path.join(self.checkpoint_dir, self.bag_auc_best_teacherhead_ckpt_name_gleason5)) 
            torch.save(self.model_teacherHead_distill.state_dict(), os.path.join(self.checkpoint_dir, self.bag_auc_best_teacherhead_distill_ckpt_name_gleason5)) 
            torch.save(self.model_student.state_dict(), os.path.join(self.checkpoint_dir, self.bag_auc_best_student_ckpt_name_gleason5)) 
            save_validation_threshold(
                os.path.join(self.checkpoint_dir, self.bag_auc_best_threshold_name_gleason5),
                bag_threshold, bag_validation_metrics, epoch,
            )

        #self.writer.add_scalar('test_instance_AUC_byTeacher_gleason3', instance_auc_ByTeacher_gleason3, epoch)
        self.writer.add_scalar('val_bag_AUC', bag_auc_ByTeacher_withAttnScore_gleason5, epoch)
        self.writer.add_scalar('val_bag_PR_AUC', bag_pr_auc, epoch)
        add_selected_threshold_metrics_to_writer(
            self.writer, 'val_bag_best_f1', bag_threshold, bag_validation_metrics, epoch
        )

        return 0


    def optimize(self):
        """Perform full optimization."""

        first_epoch = 0

        # Perform optmization ###############################################################
        epoch = first_epoch

        while epoch < self.num_epochs:
            self.optimize_teacher(epoch)
            self.evaluate_teacher(epoch)
            self.optimize_student(epoch)
            self.evaluate_student(epoch)

            epoch += 1
            
        return self.model_student

    def evaluate_student(self, epoch):
        self.model_encoder.eval()
        self.model_student.eval()
        ## optimize teacher with instance-dataloader
        # 1. change loader to instance-loader
        loader = self.val_loader
        # 2. optimize
        patch_label_gt_gleason5 = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred_gleason5 = torch.zeros([loader.dataset.__len__(), 1]).type(torch.float64).to(self.dev)
        bag_label_gt_gleason5 = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)

        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)
            niter = epoch * len(loader) + iter

            data = data.to(self.dev)

            # get student output of instance
            with torch.no_grad():
                feat = self.model_encoder(data)
                final_student, concept_feats_student = self.model_student(feat)

                sim_scores_student_gleason5 = (final_student @ concept_feats_student.T * self.model_student.logit_scale.exp())
                #patch_prediction_gleason5 = torch.softmax(sim_scores_student_gleason5, dim=1)
                patch_prediction_gleason5 = torch.sigmoid(sim_scores_student_gleason5) 

            patch_corresponding_slide_idx[selected, 0] = label[2]
            
            patch_label_pred_gleason5[selected, 0] = patch_prediction_gleason5.detach()[:, 3]
            patch_label_gt_gleason5[selected, 0] = label[0][:, 2]
            bag_label_gt_gleason5[selected, 0] = label[1][:, 2]

        instance_auc_ByStudent_gleason5 = utliz.cal_auc(
            patch_label_gt_gleason5.reshape(-1), patch_label_pred_gleason5.reshape(-1)
        )
        instance_pr_auc = cal_pr_auc(
            patch_label_gt_gleason5.reshape(-1), patch_label_pred_gleason5.reshape(-1)
        )
        instance_threshold, instance_validation_metrics = find_best_f1_threshold(
            patch_label_gt_gleason5.reshape(-1), patch_label_pred_gleason5.reshape(-1)
        )
        ###instance_auc_ByStudent_gleason4 = utliz.cal_auc(patch_label_gt_gleason4.reshape(-1), patch_label_pred_gleason4.reshape(-1))
        ###instance_auc_ByStudent_gleason5 = utliz.cal_auc(patch_label_gt_gleason5.reshape(-1), patch_label_pred_gleason5.reshape(-1))
        
        if instance_auc_ByStudent_gleason5 >= self.instance_auc_best_gleason5:
            print(f'Validation gleason5 instance auc increased ({self.instance_auc_best_gleason5:.6f} --> {instance_auc_ByStudent_gleason5:.6f}).  Saving model ...')
            print_selected_threshold_metrics(
                f'Fold{self.fold_id} Validation instance', instance_threshold, instance_validation_metrics
            )
            self.instance_auc_best_gleason5 = instance_auc_ByStudent_gleason5
            torch.save(self.model_encoder.state_dict(), os.path.join(self.checkpoint_dir, self.instance_auc_best_encoder_ckpt_name_gleason5)) 
            torch.save(self.model_teacher.state_dict(), os.path.join(self.checkpoint_dir, self.instance_auc_best_teacher_ckpt_name_gleason5)) 
            torch.save(self.model_teacherHead.state_dict(), os.path.join(self.checkpoint_dir, self.instance_auc_best_teacherhead_ckpt_name_gleason5)) 
            torch.save(self.model_teacherHead_distill.state_dict(), os.path.join(self.checkpoint_dir, self.instance_auc_best_teacherhead_distill_ckpt_name_gleason5)) 
            torch.save(self.model_student.state_dict(), os.path.join(self.checkpoint_dir, self.instance_auc_best_student_ckpt_name_gleason5)) 
            save_validation_threshold(
                os.path.join(self.checkpoint_dir, self.instance_auc_best_threshold_name_gleason5),
                instance_threshold, instance_validation_metrics, epoch,
            )

        self.writer.add_scalar('val_instance_AUC', instance_auc_ByStudent_gleason5, epoch)
        self.writer.add_scalar('val_instance_PR_AUC', instance_pr_auc, epoch)
        add_selected_threshold_metrics_to_writer(
            self.writer, 'val_instance_best_f1', instance_threshold, instance_validation_metrics, epoch
        )
        return 0

    def eval(self):
        """Evaluate the best instance-level and bag-level checkpoints."""
        return self.evaluate_instance_gleason5(), self.evaluate_bag_gleason5()


    def evaluate_instance_gleason5(self):
        instance_threshold = load_validation_threshold(
            os.path.join(self.checkpoint_dir, self.instance_auc_best_threshold_name_gleason5)
        )
        self.model_encoder.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, self.instance_auc_best_encoder_ckpt_name_gleason5), weights_only=False))
        self.model_teacher.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, self.instance_auc_best_teacher_ckpt_name_gleason5), weights_only=False))
        self.model_teacherHead.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, self.instance_auc_best_teacherhead_ckpt_name_gleason5), weights_only=False))
        self.model_teacherHead_distill.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, self.instance_auc_best_teacherhead_distill_ckpt_name_gleason5), weights_only=False))
        self.model_student.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, self.instance_auc_best_student_ckpt_name_gleason5), weights_only=False))
        self.model_encoder.eval()
        self.model_student.eval()
        ## optimize teacher with instance-dataloader
        # 1. change loader to instance-loader
        loader = self.test_loader
        # 2. optimize
        patch_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)  # only for patch-label available dataset
        patch_label_pred = torch.zeros([loader.dataset.__len__(), 1]).type(torch.float64).to(self.dev)
        bag_label_gt = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        patch_corresponding_slide_idx = torch.zeros([loader.dataset.__len__(), 1]).long().to(self.dev)
        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Student evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)

            data = data.to(self.dev)

            # get student output of instance
            with torch.no_grad():
                feat = self.model_encoder(data)
                final_student, concept_feats_student = self.model_student(feat)
                #concept_feats_student_gleason3 = concept_feats_student[[0,1],:]
                
                sim_scores_student_gleason5 = (final_student @ concept_feats_student.T * self.model_student.logit_scale.exp())
                #patch_prediction_gleason5 = torch.softmax(sim_scores_student_gleason5, dim=1)
                patch_prediction_gleason5 = torch.sigmoid(sim_scores_student_gleason5)

            patch_corresponding_slide_idx[selected, 0] = label[2]
            patch_label_pred[selected, 0] = patch_prediction_gleason5.detach()[:, 3]
            patch_label_gt[selected, 0] = label[0][:, 2]
            bag_label_gt[selected, 0] = label[1][:, 2]

        instance_auc_ByStudent = utliz.cal_auc(
            patch_label_gt.reshape(-1), patch_label_pred.reshape(-1)
        )
        instance_pr_auc = cal_pr_auc(
            patch_label_gt.reshape(-1), patch_label_pred.reshape(-1)
        )
        instance_test_metrics = cal_metrics_at_threshold(
            patch_label_gt.reshape(-1), patch_label_pred.reshape(-1), instance_threshold
        )

        print(f'Fold{self.fold_id} Test instance auc : {instance_auc_ByStudent:.6f}')
        print(f'Fold{self.fold_id} Test instance pr_auc : {instance_pr_auc:.6f}')
        print_selected_threshold_metrics(
            f'Fold{self.fold_id} Test instance', instance_threshold, instance_test_metrics
        )
        add_selected_threshold_metrics_to_writer(
            self.writer, 'test_instance_best_val_f1', instance_threshold, instance_test_metrics, self.fold_id
        )
        return make_eval_result(
            instance_auc_ByStudent, instance_pr_auc, instance_threshold, instance_test_metrics
        )


    def evaluate_bag_gleason5(self):
        bag_threshold = load_validation_threshold(
            os.path.join(self.checkpoint_dir, self.bag_auc_best_threshold_name_gleason5)
        )
        self.model_encoder.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, self.bag_auc_best_encoder_ckpt_name_gleason5), weights_only=False))
        self.model_teacher.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, self.bag_auc_best_teacher_ckpt_name_gleason5), weights_only=False))
        self.model_teacherHead.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, self.bag_auc_best_teacherhead_ckpt_name_gleason5), weights_only=False))
        self.model_teacherHead_distill.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, self.bag_auc_best_teacherhead_distill_ckpt_name_gleason5), weights_only=False))
        self.model_student.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, self.bag_auc_best_student_ckpt_name_gleason5), weights_only=False))
        self.model_encoder.eval()
        self.model_teacher.eval()
        self.model_teacherHead.eval()
        self.model_teacherHead_distill.eval()
        self.model_student.eval()
        ## optimize teacher with bag-dataloader
        # 1. change loader to bag-loader
        loader = self.test_bagloader
        # 2. optimize
        bag_label_gt_gleason5 = []
        bag_label_prediction_withAttnScore_gleason5 = []

        for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Teacher evaluating')):
            for i, j in enumerate(label):
                if torch.is_tensor(j):
                    label[i] = j.to(self.dev)
            selected = selected.squeeze(0)

            data = data.to(self.dev)
            with torch.no_grad():
                feat = self.model_encoder(data.squeeze(0))
                final_student, concept_feats_student = self.model_student(feat)
                sim_scores_student = final_student @ concept_feats_student.T * self.model_student.logit_scale.exp()
                #pred_byStudent = torch.softmax(sim_scores_student, dim=1)[:, 3]
                pred_byStudent = torch.sigmoid(sim_scores_student)[:, 3]

                hardPos_hardNeg_thresholds = [int(self.StuFilterType.split('_')[1]) / 10000.0,
                                                int(self.StuFilterType.split('_')[2]) / 100.0]

                idx_easy_pos = torch.where(pred_byStudent >= hardPos_hardNeg_thresholds[0])[0]
                # if idx_easy_pos.shape[0] == 0:  # if all instance are dropped, keep the most Positive one
                #     idx_easy_pos = torch.topk(pred_byStudent, k=1)[1]

                idx_hard_pos = torch.where((pred_byStudent < hardPos_hardNeg_thresholds[0]) & (pred_byStudent >= 0.5))[0]
                # if idx_hard_pos.shape[0] == 0:  # if all instance are dropped, keep the most Positive one
                #     idx_hard_pos = torch.topk(pred_byStudent, k=1)[1]

                idx_easy_neg = torch.where(pred_byStudent <= hardPos_hardNeg_thresholds[-1])[0]
                # if idx_easy_neg.shape[0] == 0:  # if all instance are dropped, keep the most Negative one
                #     idx_easy_neg = torch.topk(-1 * pred_byStudent, k=1)[1]

                idx_hard_neg = torch.where((pred_byStudent > hardPos_hardNeg_thresholds[-1]) & (pred_byStudent < 0.5))[0]
                # if idx_hard_neg.shape[0] == 0:  # if all instance are dropped, keep the most Negative one
                #     idx_hard_neg = torch.topk(-1 * pred_byStudent, k=1)[1]

                # split a bag into 4 part: 1: easy neg, 2: hard neg, 3: hard pos, 4: easy pos

                pseudo_bag_easy_pos = feat[idx_easy_pos]
                pseudo_bag_hard_pos = feat[idx_hard_pos]
                pseudo_bag_easy_neg = feat[idx_easy_neg]
                pseudo_bag_hard_neg = feat[idx_hard_neg]
 
                pseudo_bag_013 = torch.cat([pseudo_bag_easy_pos, pseudo_bag_hard_pos, pseudo_bag_hard_neg], dim=0) 
                pseudo_bag_123 = torch.cat([pseudo_bag_hard_pos, pseudo_bag_easy_neg, pseudo_bag_hard_neg], dim=0)  

                pseudo_bag_0123 = torch.cat([pseudo_bag_easy_pos, pseudo_bag_hard_pos, pseudo_bag_easy_neg, pseudo_bag_hard_neg], dim=0)

                # hard group
                #feat_PseudoBag = [pseudo_bag_012, pseudo_bag_013, pseudo_bag_023, pseudo_bag_123, pseudo_bag_0123] 
                feat_PseudoBag = [pseudo_bag_013, pseudo_bag_123, pseudo_bag_0123] 

                bag_pseudo_feat_distill = []

                for feat_new_i in feat_PseudoBag:
                    if feat_new_i.shape[0] == 0:  # if drop all, don't drop
                        feat_new_i = feat

                    #bag_features_teacher_i, instance_attn_score_teacher_i = self.model_teacherHead(feat_new_i, concept_feats_student.detach())
                    
                    bag_features_teacher_i, instance_attn_score_teacher_i = self.model_teacherHead(feat_new_i, concept_feats_student.detach())
                    
                    bag_pseudo_feat_distill.append(bag_features_teacher_i[0])
                    
                bag_pseudo_feat_distill = torch.cat(bag_pseudo_feat_distill, dim=-2)  ### numGroup x fs 
                
                _, concept_feats_teacher_distill = self.model_teacher(bag_pseudo_feat_distill)
                
                bag_features_teacher_distill, _ = self.model_teacherHead_distill(bag_pseudo_feat_distill, concept_feats_teacher_distill.detach())

                sim_scores_teacher_distill = bag_features_teacher_distill[0] @ concept_feats_teacher_distill.T * self.model_teacher.logit_scale.exp()
                #bag_prediction_teacher_distill = torch.softmax(sim_scores_teacher_distill, 1)
                bag_prediction_teacher_distill = torch.sigmoid(sim_scores_teacher_distill)

                bag_label_prediction_withAttnScore_gleason5.append(bag_prediction_teacher_distill.detach()[0, 3])

            bag_label_gt_gleason5.append(label[1].squeeze(0)[2])

        #patch_label_pred_gleason3 = torch.cat(patch_label_pred_gleason3)
        bag_label_prediction_withAttnScore = torch.tensor(bag_label_prediction_withAttnScore_gleason5)

        #patch_label_gt_gleason3 = torch.cat(patch_label_gt_gleason3)
        bag_label_gt = torch.tensor(bag_label_gt_gleason5)

        #patch_label_pred_normed_gleason3 = (patch_label_pred_gleason3 - patch_label_pred_gleason3.min()) / (patch_label_pred_gleason3.max() - patch_label_pred_gleason3.min())
        #instance_auc_ByTeacher_gleason3 = utliz.cal_auc(patch_label_gt_gleason3.reshape(-1), patch_label_pred_normed_gleason3.reshape(-1))
        bag_auc_ByTeacher_withAttnScore = utliz.cal_auc(
            bag_label_gt.reshape(-1), bag_label_prediction_withAttnScore.reshape(-1)
        )
        bag_pr_auc = cal_pr_auc(
            bag_label_gt.reshape(-1), bag_label_prediction_withAttnScore.reshape(-1)
        )
        bag_test_metrics = cal_metrics_at_threshold(
            bag_label_gt.reshape(-1), bag_label_prediction_withAttnScore.reshape(-1), bag_threshold
        )

        print(f'Fold{self.fold_id} Test bag auc : {bag_auc_ByTeacher_withAttnScore:.6f}')
        print(f'Fold{self.fold_id} Test bag pr_auc : {bag_pr_auc:.6f}')
        print_selected_threshold_metrics(
            f'Fold{self.fold_id} Test bag', bag_threshold, bag_test_metrics
        )
        add_selected_threshold_metrics_to_writer(
            self.writer, 'test_bag_best_val_f1', bag_threshold, bag_test_metrics, self.fold_id
        )
        return make_eval_result(
            bag_auc_ByTeacher_withAttnScore, bag_pr_auc, bag_threshold, bag_test_metrics
        )



def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: (-5)')

    parser.add_argument('--cpu', default=True, action='store_true', help='use CPU variant (slow) (default: off)')

    # architecture
    parser.add_argument('--arch', default='Feat_PANDA', type=str, help='alexnet or resnet (default: alexnet)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')

    parser.add_argument('--workers', default=0, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='SeLA_TCGA_WholeSlide_BasedOnSimCLRFeat', type=str, help='name for tensorboardX')
    parser.add_argument('--resume', default='', type=str, help='resume ckpt path')

    parser.add_argument('--seed', default=10, type=int, help='random seed')

    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default='./task_grading_panda',
                        help='manually specify the set of splits to use, instead of infering from the task and label_frac argument (default: None)')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_%s" % args.comment.replace('/', '_') + \
           "_Seed{}_lr{}".format(args.seed, args.lr)
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device
    util.setup_runtime(seed=42, cuda_dev_id=list(np.unique(args.modeldevice + args.device)))
    print(args, flush=True)
    print()
    print(name, flush=True)
    time.sleep(5)

    writer = SummaryWriter('./runs_PANDA_gleason5/%s'%name)
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    folds = np.arange(start, end)
    
    instance_metric_storage = init_eval_metric_storage()
    bag_metric_storage = init_eval_metric_storage()

    for i in folds:

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        assert args.arch in ['resnetv2', 'resnetv1', 'alexnet_PANDA', 'resnet_PANDA', 'Feat_PANDA']
        if args.arch == 'resnet_PANDA':
            model = models.__dict__[args.arch]()
        elif args.arch == 'Feat_PANDA':
            learnablePrompt_ctx_init = [
                "An image shows benign prostate glands with folded epithelium, featuring pale cytoplasm and small, regular nuclei. The glands are grouped together. * * * *",
                "An image shows the tumor retaining glandular differentiation, with small, distinct glands that contain lumina infiltrating among benign glands. The nuclei are enlarged and exhibit dark chromatin. * * * *",
                "An image shows the tumor with partial glandular differentiation, featuring poorly formed or fused glands and cribriform sheets containing multiple lumina. * * * *",
                "An image shows the tumor with almost no glandular differentiation, where cancer cells are dispersed individually or arranged in solid sheets or strands. * * * *",
            ]
            instance_prompt_learner = PromptLearner(n_ctx=16, ctx_init=learnablePrompt_ctx_init, all_ctx_trainable=False, csc=True,
                                                    classnames=["normal", "tumor", "tumor", "tumor"],
                                                    conch_model='conch_ViT-B-16', p_drop_out=0.0)
            
            learnablePrompt_ctx_init_bag = [
                "A whole slide image shows benign prostate glands with folded epithelium. The cytoplasm appears pale, and the nuclei are small and regular. The glands are grouped together, forming a typical benign architectural pattern. * * * *",
                "A whole slide image contains the tumor that retains glandular differentiation, characterized by small, distinct glands that contain lumina and infiltrate among the benign glands. The nuclei are enlarged with dark chromatin. * * * *",
                "A whole slide image contains the tumor that shows partial glandular differentiation. Glands are poorly formed or fused, with cribriform sheets showing multiple lumina. * * * *",
                "A whole slide image contains the tumor that shows almost no glandular differentiation. Cancer cells are dispersed individually or in solid sheets or strands. * * * *",
            ]
            bag_prompt_learner = PromptLearner(n_ctx=16, ctx_init=learnablePrompt_ctx_init_bag, all_ctx_trainable=False, csc=True,
                                                    classnames=["normal", "tumor", "tumor", "tumor"],
                                                    conch_model='conch_ViT-B-16', p_drop_out=0.0)
            
            model_student = Feat_predictor_student(instance_prompt_learner)
            model_teacher = Feat_predictor_teacher(bag_prompt_learner)
            # Share the frozen text encoder; keep prompts and logit scales independent.
            model_teacher.text_encoder = model_student.text_encoder

            model_encoder = panda_feat_projecter(input_dim=512, output_dim=512)

            model_teacherHead = map_abmil(Text_attention(input_feat_dim=512)).to('cuda:0')
            model_teacherHead_distill = map_abmil(Text_attention(input_feat_dim=512)).to('cuda:0')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_student.to(device)
        model_teacher.to(device)
        model_encoder.to(device)
        trans = transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=90),
                                    transforms.ToTensor()])

        train_ds_return_instance = PANDA_feat(data_csv='{}/splits_{}.csv'.format(args.split_dir, i), train=True, transform=None, downsample=1.0, drop_threshold=0, preload=True, return_bag=False)

        train_ds_return_bag = copy.deepcopy(train_ds_return_instance)
        train_ds_return_bag.return_bag = True
        val_ds_return_instance = PANDA_feat(data_csv='{}/splits_{}.csv'.format(args.split_dir, i), train=False, transform=None, downsample=1.0, drop_threshold=0, preload=True, return_bag=False)
        val_ds_return_bag = PANDA_feat(data_csv='{}/splits_{}.csv'.format(args.split_dir, i), train=False, transform=None, downsample=1.0, drop_threshold=0, preload=True, return_bag=True)

        test_ds_return_instance = PANDA_feat_test(data_csv='{}/splits_{}.csv'.format(args.split_dir, i), transform=None, downsample=1.0, drop_threshold=0, preload=True, return_bag=False)
        test_ds_return_bag = PANDA_feat_test(data_csv='{}/splits_{}.csv'.format(args.split_dir, i), transform=None, downsample=1.0, drop_threshold=0, preload=True, return_bag=True)

        train_loader_instance = torch.utils.data.DataLoader(train_ds_return_instance, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers, drop_last=False, pin_memory=True)
        train_loader_bag = torch.utils.data.DataLoader(train_ds_return_bag, batch_size=1, shuffle=True, num_workers=args.workers, drop_last=False, pin_memory=True)
        val_loader_instance = torch.utils.data.DataLoader(val_ds_return_instance, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
        val_loader_bag = torch.utils.data.DataLoader(val_ds_return_bag, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

        test_loader_instance = torch.utils.data.DataLoader(test_ds_return_instance, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
        test_loader_bag = torch.utils.data.DataLoader(test_ds_return_bag, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

        print("[Data] {} training samples".format(len(train_loader_instance.dataset)))
        print("[Data] {} evaluating samples".format(len(val_loader_instance.dataset)))
        print("[Data] {} testing samples".format(len(test_loader_instance.dataset)))
        
        if torch.cuda.device_count() > 1:
            print("Let's use", len(args.modeldevice), "GPUs for the model")
            if len(args.modeldevice) == 1:
                print('single GPU model', flush=True)
            else:
                model.features = nn.DataParallel(model.features, device_ids=list(range(len(args.modeldevice))))

        o = Optimizer(model_encoder=model_encoder, model_student=model_student, model_teacher=model_teacher, model_teacherHead=model_teacherHead, model_teacherHead_distill=model_teacherHead_distill, train_bagloader=train_loader_bag, train_instanceloader=train_loader_instance, val_bagloader=val_loader_bag, val_instanceloader=val_loader_instance,
                    test_bagloader=test_loader_bag, test_instanceloader=test_loader_instance, n_epochs=args.epochs, lr=args.lr, fold_id=i, weight_decay=10**args.wd, 
                    ckpt_dir="./results_PANDA_checkpoints_gleason5", resume=args.resume)
        o.writer = writer
        # Optimize
        o.optimize()
        instance_results, bag_results = o.eval()
        append_eval_metric_storage(instance_metric_storage, instance_results)
        append_eval_metric_storage(bag_metric_storage, bag_results)

    print("")
    print("All folds completed!")
    print("")

    for fold_pos, fold_id in enumerate(folds):
        print_eval_metric_storage_for_fold(
            f'Fold{fold_id} Test instance', instance_metric_storage, fold_pos
        )
        print_eval_metric_storage_for_fold(
            f'Fold{fold_id} Test bag', bag_metric_storage, fold_pos
        )

    print_eval_metric_summary('instance', instance_metric_storage)
    print_eval_metric_summary('bag', bag_metric_storage)
