import torch
import torch.nn as nn

from deaot.ops import GroupNorm1D, seq_to_2d
from deaot.attention import (
    GatedPropagation,
    LocalGatedPropagation,
    silu,
)

class DualBranchGPM(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            GatedPropagationModule(0),
            GatedPropagationModule(1),
            GatedPropagationModule(2),
        ])

        # Keep checkpoint-compatible parameter naming:
        # LSTT.decoder_norms.0.gn.weight
        # LSTT.decoder_norms.0.gn.bias
        self.decoder_norms = nn.ModuleList([
            GroupNorm1D(
                512,
                groups=2,
            )
        ])

    def forward(
        self,
        tgt,
        long_term_memories,
        short_term_memories,
        curr_id_emb=None,
        self_pos=None,
        size_2d=None,
    ):
        output = tgt

        intermediate = []
        intermediate_memories = []
        output_id = None

        for idx, layer in enumerate(self.layers):
            output, output_id, memories = layer(
                output,
                output_id,
                long_term_memories[idx]
                if long_term_memories is not None
                else None,
                short_term_memories[idx]
                if short_term_memories is not None
                else None,
                curr_id_emb=curr_id_emb,
                self_pos=self_pos,
                size_2d=size_2d,
            )

            cat_output = torch.cat(
                [output, output_id],
                dim=2,
            )

            intermediate.append(cat_output)
            intermediate_memories.append(memories)

        # Original inference configuration has final_norm=True
        # and intermediate_norm=False.
        cat_output = self.decoder_norms[0](
            cat_output
        )

        intermediate[-1] = cat_output

        return (
            intermediate,
            intermediate_memories,
        )


class GatedPropagationModule(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx

        self.d_model = 256
        self.att_nhead = 1
        self.d_att = 128
        self.expand_d_model = 512

        self.norm1 = nn.LayerNorm(256)

        # 128 Q/K + 512 V
        self.linear_QV = nn.Linear(
            256,
            640,
        )

        self.linear_U = nn.Linear(
            256,
            512,
        )

        if layer_idx == 0:
            self.linear_ID_V = nn.Linear(
                256,
                512,
            )
        else:
            self.id_norm1 = nn.LayerNorm(256)

            self.linear_ID_V = nn.Linear(
                512,
                512,
            )

            self.linear_ID_U = nn.Linear(
                256,
                512,
            )
        self.long_term_attn = GatedPropagation(
            d_qk=256,
            d_vu=512,
            num_head=1,
            use_linear=False,
            d_att=128,
            expand_ratio=2.0,
        )

        self.short_term_attn = LocalGatedPropagation(
            d_qk=256,
            d_vu=512,
            num_head=1,
            dilation=1,
            use_linear=False,
            d_att=128,
            max_dis=7,
            expand_ratio=2.0,
        )
        self.norm2 = nn.LayerNorm(256)
        self.id_norm2 = nn.LayerNorm(256)

        self.self_attn = GatedPropagation(
            d_qk=512,
            d_vu=512,
            num_head=1,
            d_att=128,
            expand_ratio=2.0,
        )
        
    def forward(
        self,
        tgt,
        tgt_id=None,
        long_term_memory=None,
        short_term_memory=None,
        curr_id_emb=None,
        self_pos=None,
        size_2d=(30, 30),
    ):
        _tgt = self.norm1(tgt)

        curr_QV = self.linear_QV(_tgt)

        curr_Q, curr_V = torch.split(
            curr_QV,
            [128, 512],
            dim=2,
        )

        curr_K = curr_Q

        local_Q = seq_to_2d(
            curr_Q,
            size_2d,
        )

        curr_V = silu(curr_V)

        curr_U = self.linear_U(_tgt)

        if tgt_id is None:
            tgt_id = torch.zeros_like(_tgt)

            cat_curr_U = torch.cat(
                [
                    silu(curr_U),
                    torch.ones_like(curr_U),
                ],
                dim=-1,
            )

            curr_ID_V = None

        else:
            _tgt_id = self.id_norm1(tgt_id)

            curr_ID_V = _tgt_id

            curr_ID_U = self.linear_ID_U(
                _tgt_id
            )

            cat_curr_U = silu(
                torch.cat(
                    [
                        curr_U,
                        curr_ID_U,
                    ],
                    dim=-1,
                )
            )

        if curr_id_emb is not None:
            global_K = curr_K
            global_V = curr_V

            local_K = seq_to_2d(
                global_K,
                size_2d,
            )

            local_V = seq_to_2d(
                global_V,
                size_2d,
            )

            _, global_ID_V = self.fuse_key_value_id(
                None,
                curr_ID_V,
                curr_id_emb,
            )

            local_ID_V = seq_to_2d(
                global_ID_V,
                size_2d,
            )

        else:
            (
                global_K,
                global_V,
                _,
                global_ID_V,
            ) = long_term_memory

            (
                local_K,
                local_V,
                _,
                local_ID_V,
            ) = short_term_memory

        cat_global_V = torch.cat(
            [
                global_V,
                global_ID_V,
            ],
            dim=-1,
        )

        cat_local_V = torch.cat(
            [
                local_V,
                local_ID_V,
            ],
            dim=1,
        )

        cat_tgt2, _ = self.long_term_attn(
            curr_Q,
            global_K,
            cat_global_V,
            cat_curr_U,
            size_2d,
        )

        cat_tgt3, _ = self.short_term_attn(
            local_Q,
            local_K,
            cat_local_V,
            cat_curr_U,
            size_2d,
        )

        tgt2, tgt_id2 = torch.split(
            cat_tgt2,
            256,
            dim=-1,
        )

        tgt3, tgt_id3 = torch.split(
            cat_tgt3,
            256,
            dim=-1,
        )

        tgt = tgt + tgt2 + tgt3
        tgt_id = tgt_id + tgt_id2 + tgt_id3

        _tgt = self.norm2(tgt)
        _tgt_id = self.id_norm2(tgt_id)

        q = k = v = u = torch.cat(
            [
                _tgt,
                _tgt_id,
            ],
            dim=-1,
        )

        cat_tgt2, _ = self.self_attn(
            q,
            k,
            v,
            u,
            size_2d,
        )

        tgt2, tgt_id2 = torch.split(
            cat_tgt2,
            256,
            dim=-1,
        )

        tgt = tgt + tgt2
        tgt_id = tgt_id + tgt_id2

        return (
            tgt,
            tgt_id,
            [
                [
                    curr_K,
                    curr_V,
                    None,
                    curr_ID_V,
                ],
                [
                    global_K,
                    global_V,
                    None,
                    global_ID_V,
                ],
                [
                    local_K,
                    local_V,
                    None,
                    local_ID_V,
                ],
            ],
        )

    def fuse_key_value_id(
        self,
        key,
        value,
        id_emb,
    ):
        if value is not None:
            ID_V = silu(
                self.linear_ID_V(
                    torch.cat(
                        [
                            value,
                            id_emb,
                        ],
                        dim=2,
                    )
                )
            )
        else:
            ID_V = silu(
                self.linear_ID_V(
                    id_emb
                )
            )

        return None, ID_V