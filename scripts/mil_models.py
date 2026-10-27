from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchFeatureExtractor(nn.Module):
    """Shared patch-image encoder for end-to-end MIL models.

    Input
    -----
    bag : torch.Tensor
        Tensor with shape [B, N, C, H, W].

    Output
    ------
    torch.Tensor
        Patch features with shape [B, N, D].
    """

    def __init__(self, CFG, base_model: str, pretrain: bool = True):
        super().__init__()
        self.CFG = CFG
        self.feature_dim = int(CFG.model.features_num)
        self.model = timm.create_model(
            base_model,
            pretrained=pretrain,
            img_size=CFG.model.img_size,
            num_classes=0,
            in_chans=3,
        )

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = bag.shape
        x = bag.reshape(-1, channels, height, width)

        features = self.model.forward_features(x)

        # Swin models in timm typically return [B*N, H', W', D].
        # The channel-first branch also supports compatible CNN backbones.
        if features.ndim == 4:
            if features.shape[-1] == self.feature_dim:
                features = features.mean(dim=(1, 2))
            elif features.shape[1] == self.feature_dim:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            else:
                raise ValueError(
                    f"Unexpected feature shape from backbone: {tuple(features.shape)}"
                )
        elif features.ndim != 2:
            raise ValueError(
                f"Unexpected feature shape from backbone: {tuple(features.shape)}"
            )

        return features.reshape(batch_size, num_instances, -1)


class MeanPoolingMIL(nn.Module):
    """End-to-end mean-pooling MIL baseline."""

    def __init__(
        self,
        CFG,
        base_model: str = "swin_small_patch4_window7_224",
        pretrain: bool = True,
    ):
        super().__init__()
        self.CFG = CFG
        self.encoder = PatchFeatureExtractor(CFG, base_model=base_model, pretrain=pretrain)
        self.feature_dim = int(CFG.model.features_num)
        self.classifier = nn.Linear(self.feature_dim, int(CFG.model.n_classes))

    def forward(self, bag, labels=None):
        features = self.encoder(bag)
        slide_features = features.mean(dim=1)
        logits = self.classifier(slide_features)
        return {"logits": logits}


class AttentionMIL(nn.Module):
    """End-to-end ABMIL model with gated attention pooling."""

    def __init__(
        self,
        CFG,
        base_model: str = "swin_small_patch4_window7_224",
        pretrain: bool = True,
    ):
        super().__init__()
        self.CFG = CFG
        self.encoder = PatchFeatureExtractor(CFG, base_model=base_model, pretrain=pretrain)
        self.feature_dim = int(CFG.model.features_num)
        attention_dim = int(CFG.model.attention_dim)

        self.attention_V = nn.Sequential(
            nn.Linear(self.feature_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.feature_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(self.feature_dim, int(CFG.model.n_classes))

    def forward(self, bag, labels=None):
        features = self.encoder(bag)
        attention_raw = self.attention_w(
            self.attention_V(features) * self.attention_U(features)
        )
        attention = F.softmax(attention_raw, dim=1)
        slide_features = torch.sum(attention * features, dim=1)
        logits = self.classifier(slide_features)
        return {
            "logits": logits,
            "attention": attention,
            "attention_raw": attention_raw,
        }


class CLAMSBMIL(nn.Module):
    """End-to-end CLAM-SB model with instance-level clustering loss."""

    def __init__(
        self,
        CFG,
        base_model: str = "swin_small_patch4_window7_224",
        pretrain: bool = True,
    ):
        super().__init__()
        self.CFG = CFG
        self.encoder = PatchFeatureExtractor(CFG, base_model=base_model, pretrain=pretrain)
        self.feature_dim = int(CFG.model.features_num)
        self.n_classes = int(CFG.model.n_classes)
        self.k_sample = int(CFG.model.clam_k_sample)
        attention_dim = int(CFG.model.attention_dim)

        self.attention_V = nn.Sequential(
            nn.Linear(self.feature_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.feature_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_dim, 1)

        self.classifier = nn.Linear(self.feature_dim, self.n_classes)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(self.feature_dim, 2) for _ in range(self.n_classes)]
        )
        self.instance_loss_fn = nn.CrossEntropyLoss()

    def _instance_eval_one_class(
        self,
        features: torch.Tensor,
        attention: torch.Tensor,
        classifier: nn.Module,
    ) -> torch.Tensor:
        k = min(self.k_sample, features.size(0))
        if k == 0:
            return torch.tensor(0.0, device=features.device)

        top_idx = torch.topk(attention, k=k, largest=True).indices
        bottom_idx = torch.topk(attention, k=k, largest=False).indices
        selected = torch.cat([features[top_idx], features[bottom_idx]], dim=0)
        targets = torch.cat(
            [
                torch.ones(k, dtype=torch.long, device=features.device),
                torch.zeros(k, dtype=torch.long, device=features.device),
            ],
            dim=0,
        )
        return self.instance_loss_fn(classifier(selected), targets)

    def _compute_instance_loss(
        self,
        features: torch.Tensor,
        attention: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        labels = labels.view(-1).long()

        for batch_idx in range(features.size(0)):
            label = int(labels[batch_idx].item())
            if not 0 <= label < self.n_classes:
                raise ValueError(
                    f"CLAMSBMIL received label {label}, but n_classes={self.n_classes}."
                )

            losses.append(
                self._instance_eval_one_class(
                    features[batch_idx],
                    attention[batch_idx, :, 0],
                    self.instance_classifiers[label],
                )
            )

        return torch.stack(losses).mean()

    def forward(self, bag, labels=None):
        features = self.encoder(bag)
        attention_raw = self.attention_w(
            self.attention_V(features) * self.attention_U(features)
        )
        attention = F.softmax(attention_raw, dim=1)
        slide_features = torch.sum(attention * features, dim=1)
        logits = self.classifier(slide_features)

        if self.training and labels is not None:
            instance_loss = self._compute_instance_loss(features, attention, labels)
        else:
            instance_loss = torch.tensor(0.0, device=logits.device)

        return {
            "logits": logits,
            "attention": attention,
            "attention_raw": attention_raw,
            "instance_loss": instance_loss,
        }


class DSMIL(nn.Module):
    """End-to-end DSMIL-style dual-stream MIL model."""

    def __init__(
        self,
        CFG,
        base_model: str = "swin_small_patch4_window7_224",
        pretrain: bool = True,
    ):
        super().__init__()
        self.CFG = CFG
        self.encoder = PatchFeatureExtractor(CFG, base_model=base_model, pretrain=pretrain)
        self.feature_dim = int(CFG.model.features_num)
        self.n_classes = int(CFG.model.n_classes)
        self.hidden_dim = int(CFG.model.dsmil_hidden_dim)
        self.use_logit_average = bool(CFG.model.dsmil_logit_average)

        self.instance_classifier = nn.Linear(self.feature_dim, self.n_classes)
        self.v_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.q_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.bag_classifiers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(self.n_classes)]
        )

    def forward(self, bag, labels=None):
        features = self.encoder(bag)
        batch_size = features.size(0)

        instance_logits = self.instance_classifier(features)
        max_instance_logits, max_indices = instance_logits.max(dim=1)

        values = self.v_proj(features)
        queries = self.q_proj(features)

        bag_logits_per_class = []
        attention_per_class = []

        for class_idx in range(self.n_classes):
            critical_indices = max_indices[:, class_idx]
            critical_query = queries[
                torch.arange(batch_size, device=queries.device), critical_indices
            ]

            scores = torch.sum(
                queries * critical_query.unsqueeze(1), dim=-1
            ) / math.sqrt(self.hidden_dim)
            attention = F.softmax(scores, dim=1)
            bag_repr = torch.sum(attention.unsqueeze(-1) * values, dim=1)
            bag_logit = self.bag_classifiers[class_idx](bag_repr)

            bag_logits_per_class.append(bag_logit)
            attention_per_class.append(attention.unsqueeze(-1))

        bag_logits = torch.cat(bag_logits_per_class, dim=1)
        attention = torch.cat(attention_per_class, dim=2)

        logits = (
            0.5 * (bag_logits + max_instance_logits)
            if self.use_logit_average
            else bag_logits
        )

        return {
            "logits": logits,
            "attention": attention,
            "instance_logits": instance_logits,
        }


def build_model(CFG, pretrain: bool = True):
    """Build the model specified by CFG.model.mil_name."""
    mil_name = str(CFG.model.mil_name).lower()

    model_classes = {
        "mean_pooling": MeanPoolingMIL,
        "abmil": AttentionMIL,
        "clam_sb": CLAMSBMIL,
        "dsmil": DSMIL,
    }
    if mil_name not in model_classes:
        raise ValueError(
            f"Unknown CFG.model.mil_name: {mil_name}. "
            f"Available models: {', '.join(model_classes)}."
        )

    return model_classes[mil_name](
        CFG,
        base_model=CFG.model.name,
        pretrain=pretrain,
    )
