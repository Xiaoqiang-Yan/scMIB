import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        num_genes,
        hidden_size=128,
        dropout=0,
        masked_data_weight=.75,
        mask_loss_weight=0.7,
        class_num=10,
        cluster_parameter = 0.2,
        recon_rate = 0.1
    ):
        super().__init__()
        self.num_genes = num_genes
        self.masked_data_weight = masked_data_weight
        self.mask_loss_weight = mask_loss_weight
        self.hidden_size = hidden_size
        self.class_num = class_num
        self.contrastive = cluster_parameter
        self.recon_rate = recon_rate

        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_genes, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )

        self.label_contrastive_latent = nn.Sequential(
            nn.Linear(hidden_size, class_num),
            nn.Softmax(dim=1)
        )

        self.label_contrastive_clean = nn.Sequential(
            nn.Linear(hidden_size, class_num),
            nn.Softmax(dim=1)
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            Swish(),
            nn.Linear(256, num_genes)
        )
        self.decoder = nn.Linear(
            in_features=hidden_size + num_genes, out_features=num_genes)

        self.clean_decoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.Mish(inplace=True),
            nn.Linear(512, num_genes)
        )

        self.encoders = []
        for i in range(2):
            self.encoders.append(self.encoder)

        self.encoders = nn.ModuleList(self.encoders)

    def forward_mask(self, x, y):
        latent = self.encoders[0](x)
        clean_latent = self.encoders[1](y)
        predicted_mask = self.predictor(latent)
        recon = self.decoder(
            torch.cat([latent, predicted_mask], dim=1))
        clean_recon = self.clean_decoder(clean_latent)
        label_latent = self.label_contrastive_latent(latent)
        label_clean = self.label_contrastive_clean(clean_latent)

        return latent, clean_latent, recon, clean_recon, predicted_mask, label_latent, label_clean

    def loss_mask(self, x, y, mask):
        latent, clean_latent, recon, clean_recon, predicted_mask, label_latent, label_clean = self.forward_mask(x, y)
        w_nums = mask * self.masked_data_weight + (1 - mask) * (1 - self.masked_data_weight)
        reconstruction_latent = torch.mul(
            w_nums, mse(recon, y, reduction='none'))
        reconstruction_clean = mse(clean_recon, y, reduction='none')
        reconstruction_loss = reconstruction_latent + self.recon_rate * reconstruction_clean
        reconstruction_loss = reconstruction_loss.mean()
        mask_loss = bce_logits(predicted_mask, mask, reduction="mean")
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        contrastive_loss = self.contrastive * cross_entropy_loss(label_latent, label_clean)

        return latent, clean_latent, reconstruction_loss, mask_loss, contrastive_loss

    def feature(self, x):
        latent = self.encoders[0](x)
        return latent



