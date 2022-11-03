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

import logging
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.methods.base import BaseMethod
from solo.utils.lars import LARS
from solo.utils.metrics import accuracy_at_k, weighted_mean, Coef_Kappa, BadPred
from solo.utils.misc import param_groups_layer_decay
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau

from torchmetrics import ConfusionMatrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

class LinearModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        backbone_name: str,
        loss_func: Callable,
        num_classes: int,
        fold: str,
        train_data_path: str,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lr: float,
        weight_decay: float,
        extra_optimizer_args: dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        finetune: bool = False,
        mixup_func: Callable = None,
        scheduler_interval: str = "step",
        lr_decay_steps: Optional[Sequence[int]] = None,
        no_channel_last: bool = False,
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            backbone (nn.Module): backbone architecture for feature extraction.
            loss_func (Callable): loss function to use (for mixup, label smoothing or default).
            num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.
            batch_size (int): batch size.
            optimizer (str): optimizer to use.
            weight_decay (float): weight decay.
            extra_optimizer_args (dict): extra optimizer arguments.
            scheduler (str): learning rate scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            mixup_func (Callable, optional). function to convert data and targets with mixup/cutmix. Defaults to None.
            finetune (bool): whether or not to finetune the backbone. Defaults to False.
            scheduler_interval (str): interval to update the lr scheduler. Defaults to 'step'.
            lr_decay_steps (Optional[Sequence[int]], optional): list of epochs where the learning
                rate will be decreased. Defaults to None.
            no_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models
        """

        super().__init__()

        self.backbone = backbone
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features

        self.num_classes = num_classes

        self.fold = fold
        self.train_data_path = train_data_path



        ###### classifier ###############
        if backbone_name.startswith("efficientnet") :
            #remove fc layer
            self.features_dim = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()
            self.backbone._swish = nn.Identity()

            self.classifier = nn.Sequential(nn.Linear(in_features=features_dim, out_features=num_classes, bias=True), nn.ReLU6())

        else:

            #self.classifier = nn.Linear(features_dim, num_classes)  # type: ignore
            #proj_hidden_dim = [512, 128, 64]
            proj_hidden_dim = [1024, 512, 64]

            self.classifier = nn.Sequential(
                nn.Linear(features_dim, proj_hidden_dim[0]),
                nn.BatchNorm1d(proj_hidden_dim[0]),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim[0], proj_hidden_dim[1]),
                nn.BatchNorm1d(proj_hidden_dim[1]),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim[1], proj_hidden_dim[2]),
                nn.Linear(proj_hidden_dim[2], num_classes))
                # nn.BatchNorm1d(proj_hidden_dim[2]),
                # nn.ReLU(),
                # nn.Linear(proj_hidden_dim[2], num_classes))

        


        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()


        self.loss_func = loss_func

        # training related
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.finetune = finetune
        self.mixup_func = mixup_func
        assert scheduler_interval in ["step", "epoch"]
        self.scheduler_interval = scheduler_interval
        self.lr_decay_steps = lr_decay_steps
        self.no_channel_last = no_channel_last

        # all the other parameters
        self.extra_args = kwargs

        if not finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

        # can provide up to ~20% speed up
        if not no_channel_last:
            self = self.to(memory_format=torch.channels_last)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds basic linear arguments.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("linear")

        # backbone args
        parser.add_argument("--backbone", choices=BaseMethod._BACKBONES, type=str)
        # for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # if we want to finetune the backbone
        parser.add_argument("--finetune", action="store_true")

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        parser.add_argument(
            "--optimizer", choices=LinearModel._OPTIMIZERS.keys(), type=str, required=True
        )
        # lars args
        parser.add_argument("--exclude_bias_n_norm", action="store_true")
        # adamw args
        parser.add_argument("--adamw_beta1", default=0.9, type=float)
        parser.add_argument("--adamw_beta2", default=0.999, type=float)

        parser.add_argument(
            "--scheduler", choices=LinearModel._SCHEDULERS, type=str, default="reduce"
        )
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)
        parser.add_argument(
            "--scheduler_interval", choices=["step", "epoch"], default="step", type=str
        )

        # disables channel last optimization
        parser.add_argument("--no_channel_last", action="store_true")

        return parent_parser

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configures the optimizer for the linear layer.

        Raises:
            ValueError: if the optimizer is not in (sgd, adam).
            ValueError: if the scheduler is not in not in (warmup_cosine, cosine, reduce, step,
                exponential).

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        layer_decay = self.extra_args.get("layer_decay", 0)
        if layer_decay > 0:
            assert self.finetune, "Only with use layer weight decay with finetune on."
            parameters = param_groups_layer_decay(
                self.backbone,
                self.weight_decay,
                no_weight_decay_list=self.backbone.no_weight_decay(),
                layer_decay=layer_decay,
            )
            parameters.append({"name": "classifier", "params": self.classifier.parameters()})
        else:
            parameters = (
                self.classifier.parameters()
                if not self.finetune
                else [
                    {"name": "backbone", "params": self.backbone.parameters()},
                    {"name": "classifier", "params": self.classifier.parameters()},
                ]
            )


        optimizer = optimizer(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)

        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        n=0
        if isinstance(X, dict) :
            n=1
            batch = X
            index, image, label = batch.keys()            
            indexes, X, target = batch[index], batch[image], batch[label]
       
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        with torch.set_grad_enabled(self.finetune):
            feats = self.backbone(X)

        logits = self.classifier(feats)
        if n > 0 :
            return {"logits": logits, "feats": feats, "target": target}
        else:
            return {"logits": logits, "feats": feats}

    def shared_step(
        self, batch: Tuple, batch_idx: int, train: bool()
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """

        if isinstance(batch, dict) :
            
            index, image, label = batch.keys()            
            indexes, X, target = batch[index], batch[image], batch[label]
        else :
            X, target = batch


    
        metrics = {"batch_size": X.size(0)}
        if self.training and self.mixup_func is not None:

            X, target = self.mixup_func(X, target)
            out = self(X)["logits"]
            loss = self.loss_func(out, target)
            metrics.update({"loss": loss})
        else:

            out = self(X)["logits"]
            loss = F.cross_entropy(out, target)
            acc1, acc3 = accuracy_at_k(out, target, top_k=(1, 3))
            coef_kappa, pred, targets = Coef_Kappa(out, target, self.num_classes)   

            if not train :  
                bad_pred_ind = BadPred(out, target, indexes)       
                metrics.update({"loss": loss, "acc1": acc1, "acc3": acc3, "coef_kappa": coef_kappa, "pred": pred, "targets": targets, "bad_pred_ind": bad_pred_ind})
            else:
                metrics.update({"loss": loss, "acc1": acc1, "acc3": acc3, "coef_kappa": coef_kappa, "pred": pred, "targets": targets})

            
        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set backbone to eval mode
        if not self.finetune:
            self.backbone.eval()

        out = self.shared_step(batch, batch_idx, train=True)

        log = {"train_loss": out["loss"]}
        if self.mixup_func is None:
            log.update({"train_acc1": out["acc1"], "train_acc3": out["acc3"], "train_coef_kappa": out["coef_kappa"]})

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return out["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        out = self.shared_step(batch, batch_idx, train=False)

        results = {
            "batch_size": out["batch_size"],
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc3": out["acc3"],
            "val_coef_kappa": out["coef_kappa"],
            "pred" : out["pred"],
            "targets" : out["targets"],
            "bad_pred_ind":out["bad_pred_ind"]
        }

        return results

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc3 = weighted_mean(outs, "val_acc3", "batch_size")
        val_coef_kappa = weighted_mean(outs, "val_coef_kappa", "batch_size")


        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc3": val_acc3, "val_coef_kappa": val_coef_kappa}
        self.log_dict(log, sync_dist=True)

       

    def prediction_step(trainer, model, test_loader, device, classes):

        num_classes = len(classes)

        print('trainer.predict(model, test_loader)', trainer.predict(model, test_loader))

        outs = trainer.predict(model, test_loader)[0]
        out = outs["logits"].to(device)
        target = outs["target"]
        target = torch.as_tensor(target).to(device)

        loss = F.cross_entropy(out, target)
        acc1, acc3 = accuracy_at_k(out, target, top_k=(1, 3))
        coef_kappa, pred, targets = Coef_Kappa(out, target, num_classes)   

        _, pred = out.topk(1, 1, True, True)
        pred = pred.t()[0]

        print('pred', len(pred))
        print('pred', pred)
        print('targets', targets)

        labels = np.arange(num_classes)
        cm = sklearn.metrics.confusion_matrix(targets.cpu(), pred.cpu(), labels=labels, normalize='true')
                
        plt.figure(figsize=(15,10))

        df_cm = pd.DataFrame(cm, index = classes, columns = classes)

        plt.figure(figsize = (10, 7))
        fig_ = sns.heatmap(df_cm, annot = True, cmap = 'Spectral').get_figure()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        trainer.logger.experiment.add_figure("Confusion matrix", fig_)
        plt.close(fig_)

        print('acc1', acc1)
        print('acc3', acc3)
        print('loss', loss)
        print('coef_kappa', coef_kappa)



