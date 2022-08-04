import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

def kl_loss_compute(pred, soft_targets, reduce=False):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def SemiLoss(outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
    probs_u = torch.softmax(outputs_u, dim=1)

    Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
    Lu = torch.mean((probs_u - targets_u)**2)

    return Lx, Lu, linear_rampup(epoch, warm_up)

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return 25*float(current)


def Icvar_p(args, intra_domain):

    intra_weight = torch.tensor(intra_domain)
    intra_weight = intra_weight.cpu().numpy() * args.p_icvar
    intra_weight = intra_weight.astype(float)
    intra_weight = torch.from_numpy(intra_weight)
    intra_domain_weight = F.softmax(intra_weight)
    return intra_domain_weight

# Loss functions
def loss_teaching(y_1, y_ind, t1, forget_rate, alpha):

    task_criterion = nn.CrossEntropyLoss().cuda()

    y_10 = torch.softmax(y_1, dim=1)
    _, y10_pse = torch.max(y_10.data, 1)
    _, label_pse = torch.max(t1.data, 1)

    loss_2 = F.cross_entropy(y_ind, label_pse, reduce=False)##

    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()

    remember_rate = 1 - (1-alpha) *forget_rate
    num_remember = math.ceil(remember_rate * len(ind_2_sorted))

    ind_2_update = ind_2_sorted[:num_remember]
    ind_20_update = ind_2_sorted[(num_remember):]#/*2+num_neg
    loss_1_update = task_criterion(y_1[ind_2_update], label_pse[ind_2_update])

    return ind_2_update, ind_20_update, loss_1_update

def mse_loss(out1, out2):
    quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
    return quad_diff / out1.data.nelement()

def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)

def one_hot(input, num_classes):
    output = torch.zeros(input.cuda().size(0), num_classes).cuda() \
        .scatter_(1, input.cuda().view(-1, 1), 1)
    return output