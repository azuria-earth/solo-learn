# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Dict, List, Sequence

import torch
#from torchmetrics import CohenKappa
import sklearn.metrics

def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def BadPred(outputs, targets, indexes) :   

    with torch.no_grad():

        _, pred = outputs.topk(1, 1, True, True)
        
        pred = pred.t()[0].cpu().tolist()
        targets = targets.cpu().tolist()
        bad_pred_ind = []
        for i in range(len(pred)):
            if pred[i] != targets[i] :
                bad_pred_ind+= [indexes[i].tolist()]
        return bad_pred_ind




def Coef_Kappa(
    outputs: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> Sequence[int]:
    """Computes Coef Kappa

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.

    Returns:
        coeff 
    """

    # cohenkappa = CohenKappa(num_classes=num_classes)
    import sklearn
    import numpy as np
    with torch.no_grad():

        _, pred = outputs.topk(1, 1, True, True)
        
        pred = pred.t()[0].cpu()
        targets = targets.cpu() 

        #coef_kappa = sklearn.metrics.cohen_kappa_score(targets, pred)
        
        Classes = np.unique(np.array(targets.tolist() + pred.tolist()))

        cm = sklearn.metrics.confusion_matrix(targets, pred, labels=Classes)

        # Sample size
        n = np.sum(cm)
        # Expected matrix
        sum0 = np.sum(cm, axis=0)
        sum1 = np.sum(cm, axis=1)
        expected = np.outer(sum0, sum1) / n
        # Number of classes
        n_classes = len(Classes)
        # Calculate p_o (the observed proportionate agreement) and
        # p_e (the probability of random agreement)
        identity = np.identity(n_classes)
        p_o = np.sum((identity * cm) / n)
        p_e = np.sum((identity * expected) / n)
        # Calculate Cohen's kappa
        coef_kappa = (p_o - p_e) / (1 - p_e)
        if p_e == 1 :
            print('p_e == 1, coef=', coef_kappa)
            print('targets', targets)
            print('pred', pred)

            print('cm', cm)

        #print(f'p_o = {p_o}, p_e = {p_e}, kappa = {coef_kappa:3.1f}')

        return coef_kappa, pred, targets


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
  
    return value.squeeze(0)
