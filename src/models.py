import torch
import torchaudio
from torch import nn

from modules import ResFrontEnd, Transformer


class MusicTaggingTransformer(nn.Module):
    def __init__(
        self,
        conv_ndim=16,
        n_mels=128,
        sample_rate=22050,
        n_fft=1024,
        f_min=0,
        f_max=11025,
        attention_ndim=256,
        attention_nheads=8,
        attention_nlayers=4,
        attention_max_len=512,
        dropout=0.1,
        n_seq_cls=1,
        n_token_cls=1,
    ):
        super(MusicTaggingTransformer, self).__init__()
        # Input preprocessing
        self.spec = self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                     n_fft=n_fft,
                                                                     f_min=f_min,
                                                                     f_max=f_max,
                                                                     n_mels=n_mels,
                                                                     power=2)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Input embedding
        self.frontend = ResFrontEnd(conv_ndim, attention_ndim, n_mels)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, attention_max_len + 1, attention_ndim))
        self.cls_token = nn.Parameter(torch.randn(attention_ndim))

        # transformer
        self.transformer = Transformer(
            attention_ndim,
            attention_nlayers,
            attention_nheads,
            attention_ndim // attention_nheads,
            attention_ndim * 4,
            dropout,
        )
        self.to_latent = nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # projection for sequence classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(attention_ndim), nn.Linear(attention_ndim, n_seq_cls)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): (batch, time)

        Returns:
            x (torch.Tensor): (batch, n_seq_cls)

        """
        # Input preprocessing
        x = self.spec(x)
        x = self.amplitude_to_db(x)
        x = x.unsqueeze(1)

        # Input embedding
        x = self.frontend(x)

        # Positional embedding with a [CLS] token
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)

        # transformer
        x = self.transformer(x)

        # projection for sequence classification
        x = self.to_latent(x[:, 0])
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        return x

