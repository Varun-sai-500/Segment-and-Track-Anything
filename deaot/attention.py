import torch
import torch.nn as nn
import torch.nn.functional as F

from deaot.ops import DWConv2d

def silu(x):
    return x * torch.sigmoid(x)

class GatedPropagation(nn.Module):
    def __init__(
        self,
        d_qk,
        d_vu,
        num_head=8,
        dropout=0.0,
        use_linear=True,
        d_att=None,
        expand_ratio=2.0,
    ):
        super().__init__()

        self.d_vu = d_vu
        self.num_head = num_head
        self.expand_d_vu = int(
            d_vu * expand_ratio
        )

        self.d_att = (
            d_qk // num_head
            if d_att is None
            else d_att
        )

        self.hidden_dim = (
            self.expand_d_vu // num_head
        )

        self.temperature = self.d_att ** 0.5
        self.use_linear = use_linear

        self.d_middle = self.d_att * num_head

        if use_linear:
            self.linear_QK = nn.Linear(
                d_qk,
                self.d_middle,
            )

            half_d_vu = (
                self.hidden_dim
                * num_head
                // 2
            )

            self.linear_V1 = nn.Linear(
                d_vu // 2,
                half_d_vu,
            )

            self.linear_V2 = nn.Linear(
                d_vu // 2,
                half_d_vu,
            )

            self.linear_U1 = nn.Linear(
                d_vu // 2,
                half_d_vu,
            )

            self.linear_U2 = nn.Linear(
                d_vu // 2,
                half_d_vu,
            )

        self.dropout = nn.Dropout(dropout)

        self.dw_conv = DWConv2d(
            self.expand_d_vu,
        )

        self.projection = nn.Linear(
            self.expand_d_vu,
            d_vu,
        )

    def forward(
        self,
        Q,
        K,
        V,
        U,
        size_2d,
    ):
        length, batch_size, _ = Q.size()

        if self.use_linear:
            Q = K = self.linear_QK(Q)

            V1, V2 = torch.split(
                V,
                self.d_vu // 2,
                dim=-1,
            )

            V = silu(
                torch.cat(
                    [
                        self.linear_V1(V1),
                        self.linear_V2(V2),
                    ],
                    dim=-1,
                )
            )

            U1, U2 = torch.split(
                U,
                self.d_vu // 2,
                dim=-1,
            )

            U = silu(
                torch.cat(
                    [
                        self.linear_U1(U1),
                        self.linear_U2(U2),
                    ],
                    dim=-1,
                )
            )

        Q = Q / self.temperature

        Q = Q.view(
            -1,
            batch_size,
            self.num_head,
            self.d_att,
        ).permute(
            1,
            2,
            0,
            3,
        )

        K = K.view(
            -1,
            batch_size,
            self.num_head,
            self.d_att,
        ).permute(
            1,
            2,
            3,
            0,
        )

        V = V.view(
            -1,
            batch_size,
            self.num_head,
            self.hidden_dim,
        ).permute(
            1,
            2,
            0,
            3,
        )

        attention = torch.softmax(
            torch.matmul(Q, K),
            dim=-1,
        )

        attention = self.dropout(
            attention
        )

        output = torch.matmul(
            attention,
            V,
        ).permute(
            2,
            0,
            1,
            3,
        )

        output = output.reshape(
            length,
            batch_size,
            self.expand_d_vu,
        )

        output = output * U

        output = self.dw_conv(
            output,
            size_2d,
        )

        return (
            self.projection(output),
            attention,
        )


class LocalGatedPropagation(nn.Module):
    def __init__(
        self,
        d_qk,
        d_vu,
        num_head,
        dropout=0.0,
        max_dis=7,
        dilation=1,
        use_linear=True,
        d_att=None,
        expand_ratio=2.0,
    ):
        super().__init__()

        self.d_qk = d_qk
        self.d_vu = d_vu
        self.num_head = num_head

        self.dilation = dilation
        self.max_dis = max_dis
        self.window_size = (
            2 * max_dis + 1
        )

        self.expand_d_vu = int(
            d_vu * expand_ratio
        )

        self.hidden_dim = (
            self.expand_d_vu // num_head
        )

        self.d_att = (
            d_qk // num_head
            if d_att is None
            else d_att
        )

        self.temperature = self.d_att ** 0.5
        self.use_linear = use_linear

        self.d_middle = (
            self.d_att * num_head
        )

        if use_linear:
            self.linear_QK = nn.Conv2d(
                d_qk,
                self.d_middle,
                kernel_size=1,
            )

            self.linear_V = nn.Conv2d(
                d_vu,
                self.expand_d_vu,
                kernel_size=1,
                groups=2,
            )

            self.linear_U = nn.Conv2d(
                d_vu,
                self.expand_d_vu,
                kernel_size=1,
                groups=2,
            )

        self.relative_emb_k = nn.Conv2d(
            self.d_middle,
            num_head * self.window_size
            * self.window_size,
            kernel_size=1,
            groups=num_head,
        )

        self.dw_conv = DWConv2d(
            self.expand_d_vu,
        )

        self.projection = nn.Linear(
            self.expand_d_vu,
            d_vu,
        )

        self.local_mask = None
        self.last_size_2d = None
        self.qk_mask = None

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q,
        k,
        v,
        u,
        size_2d,
    ):
        n, _, h, w = v.size()

        if self.use_linear:
            q = k = self.linear_QK(q)

            v = silu(
                self.linear_V(v)
            )

            u = silu(
                self.linear_U(u)
            )

            if self.num_head > 1:
                v = (
                    v.view(
                        -1,
                        2,
                        self.num_head,
                        self.hidden_dim // 2,
                        h * w,
                    )
                    .permute(
                        0,
                        2,
                        1,
                        3,
                        4,
                    )
                    .reshape(
                        n,
                        -1,
                        h,
                        w,
                    )
                )

                u = (
                    u.view(
                        -1,
                        2,
                        self.num_head,
                        self.hidden_dim // 2,
                        h * w,
                    )
                    .permute(
                        4,
                        0,
                        2,
                        1,
                        3,
                    )
                    .reshape(
                        h * w,
                        n,
                        -1,
                    )
                )
            else:
                u = (
                    u.permute(2, 3, 0, 1)
                    .reshape(
                        h * w,
                        n,
                        -1,
                    )
                )

        if (
            self.qk_mask is not None
            and (h, w) == self.last_size_2d
        ):
            qk_mask = self.qk_mask
        else:
            memory_mask = torch.ones(
                (1, 1, h, w),
                device=v.device,
            )

            unfolded_k_mask = (
                self.pad_and_unfold(
                    memory_mask
                ).view(
                    1,
                    1,
                    self.window_size
                    * self.window_size,
                    h * w,
                )
            )

            qk_mask = 1 - unfolded_k_mask
            self.qk_mask = qk_mask

        relative_emb = self.relative_emb_k(q)

        q = q / self.temperature

        q = q.view(
            -1,
            self.d_att,
            h,
            w,
        )

        k = k.view(
            -1,
            self.d_att,
            h,
            w,
        )

        v = v.view(
            n,
            self.num_head,
            self.hidden_dim,
            h * w,
        )

        relative_emb = relative_emb.view(
            n,
            self.num_head,
            self.window_size
            * self.window_size,
            h * w,
        )

        unfolded_k = self.pad_and_unfold(
            k
        ).view(
            n * self.num_head,
            self.d_att,
            self.window_size
            * self.window_size,
            h,
            w,
        )

        qk = (
            q.unsqueeze(2)
            * unfolded_k
        ).sum(dim=1).view(
            n,
            self.num_head,
            self.window_size
            * self.window_size,
            h * w,
        )

        qk = qk + relative_emb

        qk = qk.masked_fill(
            qk_mask.bool(),
            -1e8
            if qk.dtype == torch.float32
            else -1e4,
        )

        local_attention = torch.softmax(
            qk,
            dim=2,
        )

        local_attention = self.dropout(
            local_attention
        )

        global_attention = self.local2global(
            local_attention,
            h,
            w,
        )

        output = (
            global_attention
            @ v.transpose(-2, -1)
        ).permute(
            2,
            0,
            1,
            3,
        ).reshape(
            h * w,
            n,
            -1,
        )

        output = output * u

        output = self.dw_conv(
            output,
            size_2d,
        )

        output = self.projection(
            output
        )

        self.last_size_2d = (h, w)

        return (
            output,
            local_attention,
        )

    def local2global(
        self,
        local_attention,
        height,
        width,
    ):
        batch_size = (
            local_attention.size(0)
        )

        pad_height = (
            height + 2 * self.max_dis
        )

        pad_width = (
            width + 2 * self.max_dis
        )

        if (
            self.local_mask is not None
            and (height, width)
            == self.last_size_2d
        ):
            local_mask = self.local_mask

        else:
            ky, kx = torch.meshgrid(
                torch.arange(
                    pad_height,
                    device=local_attention.device,
                ),
                torch.arange(
                    pad_width,
                    device=local_attention.device,
                ),
                indexing="ij",
            )

            qy, qx = torch.meshgrid(
                torch.arange(
                    height,
                    device=local_attention.device,
                ),
                torch.arange(
                    width,
                    device=local_attention.device,
                ),
                indexing="ij",
            )

            offset_y = (
                qy.reshape(-1, 1)
                - ky.reshape(1, -1)
                + self.max_dis
            )

            offset_x = (
                qx.reshape(-1, 1)
                - kx.reshape(1, -1)
                + self.max_dis
            )

            local_mask = (
                (offset_y.abs() <= self.max_dis)
                & (offset_x.abs() <= self.max_dis)
            )

            local_mask = local_mask.view(
                1,
                1,
                height * width,
                pad_height,
                pad_width,
            )

            self.local_mask = local_mask

        global_attention = torch.zeros(
            (
                batch_size,
                self.num_head,
                height * width,
                pad_height,
                pad_width,
            ),
            device=local_attention.device,
        )

        global_attention[
            local_mask.expand(
                batch_size,
                self.num_head,
                -1,
                -1,
                -1,
            )
        ] = local_attention.transpose(
            -1,
            -2,
        ).reshape(-1)

        global_attention = global_attention[
            :,
            :,
            :,
            self.max_dis:-self.max_dis,
            self.max_dis:-self.max_dis,
        ].reshape(
            batch_size,
            self.num_head,
            height * width,
            height * width,
        )

        return global_attention

    def pad_and_unfold(self, x):
        pad_pixel = (
            self.max_dis * self.dilation
        )

        x = F.pad(
            x,
            (
                pad_pixel,
                pad_pixel,
                pad_pixel,
                pad_pixel,
            ),
            mode="constant",
            value=0,
        )

        return F.unfold(
            x,
            kernel_size=(
                self.window_size,
                self.window_size,
            ),
            stride=1,
            dilation=self.dilation,
        )