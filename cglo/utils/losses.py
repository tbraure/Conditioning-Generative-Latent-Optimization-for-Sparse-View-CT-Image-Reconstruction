import torch
import torch.nn.functional as F
import numpy as np

from cglo.utils.physics import center_radon_transform  as f_op

def l2_dose_loss(x, y, nb_ang, reduction='sum'):
    xy_dose = f_op(torch.cat([x, y]), nb_ang)
    x_dose, y_dose = xy_dose[:x.shape[0]], xy_dose[x.shape[0]:]
    return F.mse_loss(x_dose, y_dose, reduction=reduction)

def l1_dose_loss(x, y, nb_ang, reduction='sum'):
    xy_dose = f_op(torch.cat([x, y]), nb_ang)
    x_dose, y_dose = xy_dose[:x.shape[0]], xy_dose[x.shape[0]:]
    return F.l1_loss(x_dose, y_dose, reduction=reduction)

def l1_dose_half_loss(x_dose, y, nb_ang, reduction='mean'):
    return F.l1_loss(f_op(y, nb_ang), x_dose, reduction=reduction)

def l2_dose_half_loss(x_dose, y, nb_ang, reduction='mean'):
    return F.mse_loss(f_op(y, nb_ang), x_dose, reduction=reduction)

