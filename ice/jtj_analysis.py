import functools
import itertools
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from training.networks import Generator
from torch_utils import misc

import legacy

from wrapper import StyleGanWrapper, FaceSegmenter, KeyPointDetector
from criterions import * 

def find_jtj_direct(f, dataloader, bases=None):
    JtJs = None
    for u in (dataloader):
        f.reset()
        batch_size = len(u)
        u_param = nn.Parameter(u, requires_grad=True)
        ys = f(u_param)
        if not isinstance(ys, (list, tuple)):
            ys = [ys]
        ys_fl = [_.flatten(1, -1) for _ in ys]

        Js = [u.new_zeros(_.shape + (u.flatten(1, -1).shape[1],)) for _ in ys_fl]
        for i, y_fl in enumerate(ys_fl):
            for j in tqdm(range(y_fl.shape[1])):
                u_param.grad is None or u_param.grad.zeros_()
                y_grad = u.new_tensor(j).repeat(batch_size).long()
                y_grad = F.one_hot(y_grad, num_classes=y_fl.shape[1]).float()
                u_grad = torch.autograd.grad(y_fl, u_param,
                        grad_outputs=y_grad, retain_graph=True)
                Js[i][:, j, :] = u_grad[0].flatten(1, -1).detach()

        Js = [_.flatten(0, 1) for _ in Js]
        if JtJs is None:
            JtJs = [0 for _ in ys]
        for i in range(len(JtJs)):
            JtJs[i] += Js[i].T @ Js[i]

    JtJs = [_ / len(dataloader.dataset) for _ in JtJs]
    return JtJs

def find_jtj_approx(f, dataloader, bases=None):
    JtJs = None
    for u in (dataloader):
        f.reset()
        with torch.no_grad():
            y0s = f(u)
        if not isinstance(y0s, (list, tuple)):
            y0s = [y0s]

        u_fl = u.flatten(1, -1)
        JtJs_cur = [u.new_zeros((u_fl.shape[1],) * 2) for _ in range(len(y0s))]
        alpha = u_fl.norm(dim=1).mean().item() * 1e-4
        eye = torch.eye(*(u_fl.shape[1],) * 2, device=u.device)
        for iu in tqdm(range(u_fl.shape[1])):
            n_param = nn.Parameter(eye[iu], requires_grad=True)
            u1 = (u_fl + n_param * alpha).reshape(u.shape)
            y1s = f(u1)
            if not isinstance(y1s, (list, tuple)):
                y1s = [y1s]
            for iy in range(len(y1s)):
                loss = (y1s[iy] - y0s[iy]).square().sum() / 2
                f.zero_grad()
                n_param.grad is None or n_param.grad.zero_()
                loss.backward(retain_graph=True)
                JtJs_cur[iy][iu, :] = n_param.grad.clone().detach()

        if JtJs is None:
            JtJs = [0 for _ in y0s]
        for i in range(len(JtJs)):
            JtJs[i] += JtJs_cur[i]


    for i in range(len(JtJs)):
        JtJs[i] /= len(dataloader.dataset) * alpha ** 2
        JtJs[i] = (JtJs[i] + JtJs[i].T) / 2
    return JtJs


def load_or_compute(f, compute, recompute=False, map_location=None):
    if not Path(f).is_file() or recompute:
        res = compute()
        torch.save(res, f)
    else:
        res = torch.load(f, map_location=map_location)
    return res

def projected_pca(JtJ, bases=None):
    if bases is not None:
        JtJ = bases.T @ JtJ @ bases
    S, V = torch.symeig(JtJ, eigenvectors=True)
    index = S.abs().argsort(descending=True)
    S, V = S[index], V[:, index]
    if bases is not None:
        V = bases @ V
    return S, V

def trim_stack(*V, dim=1):
    min_len = min(_.shape[dim] for _ in V)
    assert dim == 1
    stacked = torch.stack(tuple(_[:, :min_len] for _ in V))
    return stacked


# https://discuss.pytorch.org/t/nullspace-of-a-tensor/69980/4
def nullspace(A, rcond=None):
    At = A.T
    ut, st, vht = torch.Tensor.svd(At, some=False,compute_uv=True)
    vht=vht.T        
    Mt, Nt = ut.shape[0], vht.shape[1] 
    if rcond is None:
        rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
    tolt = torch.max(st) * rcondt
    numt= torch.sum(st > tolt, dtype=int)
    nullspace = vht[numt:,:].T.conj()
    # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
    return nullspace

def subspace_intersect(*As):
    null_as = [nullspace(_) for _ in As]
    combined = torch.cat(null_as, dim=1)
    intersected = nullspace(combined)
    return intersected

def compute_early_projected_pca(jtj_act, jtjs_sup, sup_ratio=1e-2, act_ratio=1e-2):
    vs_sup = []
    if not isinstance(sup_ratio, (list, tuple)):
        sup_ratio = [sup_ratio] * len(jtjs_sup)
    for jtj_sup, ratio in zip(jtjs_sup, sup_ratio):
        s_sup, v_sup = projected_pca(jtj_sup)
        mask = s_sup.abs() < s_sup.abs().max() * ratio
        v_sup = v_sup[:, mask]
        vs_sup.append(v_sup)

    if not vs_sup:
        vs_sup = None
    else:
        vs_sup = subspace_intersect(*vs_sup)
    s_act, v_act = projected_pca(jtj_act, bases=vs_sup)
    if act_ratio is not None:
        mask = s_act.abs() > s_act.abs().max() * act_ratio
        s_act = s_act[mask]
        v_act = v_act[:, mask]
    return s_act, v_act
        
def compute_late_projected_pca(jtj_act, jtjs_sup, sup_ratio=1e-2, act_ratio=1e-2):
    vs_sup = []
    if not isinstance(sup_ratio, (list, tuple)):
        sup_ratio = [sup_ratio] * len(jtjs_sup)
    for jtj_sup, ratio in zip(jtjs_sup, sup_ratio):
        s_sup, v_sup = projected_pca(jtj_sup)
        mask = s_sup.abs() < s_sup.abs().max() * ratio
        v_sup = v_sup[:, mask]
        vs_sup.append(v_sup)

    if not vs_sup:
        vs_sup = None
    else:
        vs_sup = subspace_intersect(*vs_sup)
    s_act, v_act = projected_pca(jtj_act)
    v_act = v_sup @ v_sup.T @ v_act
    v_act = F.normalize(v_act, dim=1)
    if act_ratio is not None:
        mask = s_act.abs() > s_act.abs().max() * act_ratio
        s_act = s_act[mask]
        v_act = v_act[:, mask]
    return s_act, v_act

def compute_projected_pca(jtj_act, early_jtjs_sup, late_jtjs_sup, early_sup_ratio, late_sup_ratio=[], act_ratio=1e-2):
    early_vs_sup = []
    if not isinstance(early_sup_ratio, (list, tuple)):
        early_sup_ratio = [early_sup_ratio] * len(jtjs_sup)
    for jtj_sup, ratio in zip(early_jtjs_sup, early_sup_ratio):
        s_sup, v_sup = projected_pca(jtj_sup)
        mask = s_sup.abs() < s_sup.abs().max() * ratio
        v_sup = v_sup[:, mask]
        early_vs_sup.append(v_sup)
    if not early_vs_sup:
        early_vs_sup = None
    else:
        early_vs_sup = subspace_intersect(*early_vs_sup)

    s_act, v_act = projected_pca(jtj_act, bases=early_vs_sup)

    late_vs_sup = []
    if not isinstance(late_sup_ratio, (list, tuple)):
        late_sup_ratio = [late_sup_ratio] * len(jtjs_sup)
    for jtj_sup, ratio in zip(late_jtjs_sup, late_sup_ratio):
        s_sup, v_sup = projected_pca(jtj_sup)
        mask = s_sup.abs() < s_sup.abs().max() * ratio
        v_sup = v_sup[:, mask]
        late_vs_sup.append(v_sup)
    if not late_vs_sup:
        late_vs_sup = None
    else:
        late_vs_sup = subspace_intersect(*late_vs_sup)
        v_act = late_vs_sup @ late_vs_sup.T @ v_act

    if act_ratio is not None:
        mask = s_act.abs() > s_act.abs().max() * act_ratio
        s_act = s_act[mask]
        v_act = v_act[:, mask]
    return s_act, v_act
        

    


