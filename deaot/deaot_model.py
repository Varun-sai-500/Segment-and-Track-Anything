import torch.nn as nn

from deaot.resnet_encoder import ResNet
from deaot.transformers import DualBranchGPM
from .fpn_decoder import FPNSegmentationHead


class DeAOT(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_obj_num = 10
        self.encoder = ResNet()

        self.encoder_projector = nn.Conv2d(
            1024,
            256,
            kernel_size=1,
        )

        self.LSTT = DualBranchGPM()

        self.decoder = FPNSegmentationHead()

        self.patch_wise_id_bank = nn.Conv2d(
            self.max_obj_num + 1,
            256,
            kernel_size=17,
            stride=16,
            padding=8,
        )

        self.id_norm = nn.LayerNorm(256)

        self._init_weight()

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)

        id_emb = self.id_norm(
            id_emb.permute(2, 3, 0, 1)
        ).permute(2, 3, 0, 1)

        return id_emb

    def encode_image(self, img):
        xs = self.encoder(img)
        xs[-1] = self.encoder_projector(xs[-1])
        return xs

    def decode_id_logits(self, lstt_emb, shortcuts):
        n, _, h, w = shortcuts[-1].size()

        decoder_inputs = [shortcuts[-1]]

        for emb in lstt_emb:
            decoder_inputs.append(
                emb.view(h, w, n, -1)
                .permute(2, 3, 0, 1)
            )

        return self.decoder(
            decoder_inputs,
            shortcuts,
        )

    def LSTT_forward(
        self,
        curr_embs,
        long_term_memories,
        short_term_memories,
        curr_id_emb=None,
        size_2d=(30, 30),
    ):
        n, c, h, w = curr_embs[-1].size()

        curr_emb = (
            curr_embs[-1]
            .view(n, c, h * w)
            .permute(2, 0, 1)
        )

        lstt_embs, lstt_memories = self.LSTT(
            curr_emb,
            long_term_memories,
            short_term_memories,
            curr_id_emb=curr_id_emb,
            size_2d=size_2d,
        )

        (
            lstt_curr_memories,
            lstt_long_memories,
            lstt_short_memories,
        ) = zip(*lstt_memories)

        return (
            lstt_embs,
            lstt_curr_memories,
            lstt_long_memories,
            lstt_short_memories,
        )

    def _init_weight(self):
        nn.init.xavier_uniform_(
            self.encoder_projector.weight
        )

        nn.init.orthogonal_(
            self.patch_wise_id_bank.weight.view(
                256,
                -1,
            ),
            gain=17 ** -2,
        )