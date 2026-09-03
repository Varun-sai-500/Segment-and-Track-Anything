import torch
import torch.nn as nn
import torch.nn.functional as F

from deaot.ops import seq_to_2d

def one_hot_mask(mask, cls_num):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    indices = torch.arange(
        cls_num + 1,
        device=mask.device,
    ).view(1, -1, 1, 1)

    return (mask == indices).float()


class DeAOTEngine(nn.Module):
    def __init__(
        self,
        deaot_model,
        long_term_mem_gap=9999,
        short_term_mem_skip=1,
        max_len_long_term=9999,
    ):
        super().__init__()

        self.DeAOT = deaot_model
        self.max_obj_num = deaot_model.max_obj_num

        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.max_len_long_term = max_len_long_term

        self.restart_engine()

    # ------------------------------------------------------------------
    # Encoding / identity
    # ------------------------------------------------------------------

    def encode_one_img_mask(self, img, mask=None):
        if img is None:
            curr_enc_embs = None
        else:
            curr_enc_embs = self.DeAOT.encode_image(img)

        if mask is not None:
            curr_one_hot_mask = one_hot_mask(
                mask,
                self.max_obj_num,
            )
        else:
            curr_one_hot_mask = None

        return curr_enc_embs, curr_one_hot_mask

    def assign_identity(self, one_hot_mask):
        id_emb = self.DeAOT.get_id_emb(
            one_hot_mask
        ).view(
            self.batch_size,
            -1,
            self.enc_hw,
        ).permute(
            2,
            0,
            1,
        )

        return id_emb

    # ------------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------------

    def add_reference_frame(
        self,
        img,
        mask,
        obj_nums,
        frame_step=-1,
        img_embs=None,
    ):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]

        self.obj_nums = obj_nums

        if frame_step != -1:
            self.frame_step = frame_step

        if img_embs is None:
            curr_enc_embs, curr_one_hot_mask = (
                self.encode_one_img_mask(
                    img,
                    mask,
                )
            )
        else:
            curr_enc_embs = img_embs
            _, curr_one_hot_mask = (
                self.encode_one_img_mask(
                    img,
                    mask,
                )
            )

        if self.input_size_2d is None:
            if img is None:
                raise RuntimeError(
                    "Input size is not initialized and no reference "
                    "image was provided."
                )

            self.update_size(
                img.shape[-2:],
                curr_enc_embs[-1].shape[-2:],
            )

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        self.curr_id_embs = self.assign_identity(
            curr_one_hot_mask
        )

        self.curr_lstt_output = (
            self.DeAOT.LSTT_forward(
                curr_enc_embs,
                None,
                None,
                self.curr_id_embs,
                size_2d=self.enc_size_2d,
            )
        )

        (
            _,
            _,
            lstt_long_memories,
            lstt_short_memories,
        ) = self.curr_lstt_output

        if self.long_term_memories is None:
            self.long_term_memories = (
                lstt_long_memories
            )
        else:
            self.update_long_term_memory(
                lstt_long_memories
            )

        self.last_mem_step = self.frame_step

        self.short_term_memories_list = [
            lstt_short_memories
        ]

        self.short_term_memories = (
            lstt_short_memories
        )

    @torch.no_grad()
    def add_reference_frame_incremental(
        self,
        mask,
        obj_nums,
        frame_step=-1,
    ):
        if self.obj_nums is None:
            raise RuntimeError(
                "Cannot add incremental objects before "
                "tracking is initialized"
            )

        if obj_nums <= self.obj_nums:
            raise ValueError(
                f"Incremental reference must increase "
                f"object count: current={self.obj_nums}, "
                f"new={obj_nums}."
            )

        if self.curr_enc_embs is None:
            raise RuntimeError(
                "No current frame embeddings are available "
                "for incremental reference initialization."
            )

        self.add_reference_frame(
            img=None,
            mask=mask,
            obj_nums=obj_nums,
            frame_step=frame_step,
            img_embs=self.curr_enc_embs,
        )


    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------

    def update_long_term_memory(self, new_memories):
        if self.long_term_memories is None:
            self.long_term_memories = new_memories
            return

        token_num = new_memories[0][0].shape[0]

        updated = []

        for new_memory, old_memory in zip(
            new_memories,
            self.long_term_memories,
        ):
            layer = []

            for new_e, old_e in zip(
                new_memory,
                old_memory,
            ):
                if new_e is None or old_e is None:
                    layer.append(None)
                    continue

                if old_e.shape[0] >= (
                    self.max_len_long_term * token_num
                ):
                    old_e = old_e[
                        :(self.max_len_long_term - 1)
                        * token_num
                    ]

                layer.append(
                    torch.cat(
                        [new_e, old_e],
                        dim=0,
                    )
                )

            updated.append(layer)

        self.long_term_memories = updated

    # ------------------------------------------------------------------
    # Short-term memory
    # ------------------------------------------------------------------

    def update_short_term_memory(
        self,
        curr_mask,
        skip_long_term_update=False,
    ):
        if self.obj_nums is None:
            raise RuntimeError(
                "add_reference_frame() must be called before "
                "update_short_term_memory()"
            )

        if self.curr_lstt_output is None:
            raise RuntimeError(
                "No current LSTT output is available."
            )

        curr_one_hot_mask = one_hot_mask(
            curr_mask,
            self.max_obj_num,
        )

        curr_id_emb = self.assign_identity(
            curr_one_hot_mask
        )

        # IMPORTANT:
        # curr_lstt_output is the raw output of the most recent
        # LSTT_forward(). Do not mutate it here.
        lstt_curr_memories = (
            self.curr_lstt_output[1]
        )

        memories_2d = []

        for layer_idx, memories in enumerate(
            lstt_curr_memories
        ):
            (
                curr_k,
                curr_v,
                curr_id_k,
                curr_id_v,
            ) = memories

            fused_id_k, fused_id_v = (
                self.DeAOT.LSTT
                .layers[layer_idx]
                .fuse_key_value_id(
                    curr_id_k,
                    curr_id_v,
                    curr_id_emb,
                )
            )

            memories_2d.append([
                seq_to_2d(
                    curr_k,
                    self.enc_size_2d,
                ),
                seq_to_2d(
                    curr_v,
                    self.enc_size_2d,
                ),
                (
                    seq_to_2d(
                        fused_id_k,
                        self.enc_size_2d,
                    )
                    if fused_id_k is not None
                    else None
                ),
                seq_to_2d(
                    fused_id_v,
                    self.enc_size_2d,
                ),
            ])

        self.short_term_memories_list.append(
            memories_2d
        )

        self.short_term_memories_list = (
            self.short_term_memories_list[
                -self.short_term_mem_skip:
            ]
        )

        self.short_term_memories = (
            self.short_term_memories_list[-1]
        )

        if (
            self.frame_step - self.last_mem_step
            >= self.long_term_mem_gap
        ):
            if not skip_long_term_update:
                self.update_long_term_memory(
                    lstt_curr_memories
                )
                self.last_mem_step = (
                    self.frame_step
                )

    def update_memory(
        self,
        curr_mask,
        skip_long_term_update=False,
    ):
        self.update_short_term_memory(
            curr_mask,
            skip_long_term_update=skip_long_term_update,
        )

    # ------------------------------------------------------------------
    # Propagation / decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def match_propogate_one_frame(self, img):
        if self.obj_nums is None:
            raise RuntimeError(
                "add_reference_frame() must be called "
                "before track()"
            )

        self.frame_step += 1

        curr_enc_embs, _ = (
            self.encode_one_img_mask(
                img,
                None,
            )
        )

        self.curr_enc_embs = curr_enc_embs

        # Produces a fresh raw curr_lstt_output.
        self.curr_lstt_output = (
            self.DeAOT.LSTT_forward(
                curr_enc_embs,
                self.long_term_memories,
                self.short_term_memories,
                size_2d=self.enc_size_2d,
            )
        )

    def decode_current_logits(
        self,
        output_size=None,
    ):
        pred_id_logits = (
            self.DeAOT.decode_id_logits(
                self.curr_lstt_output[0],
                self.curr_enc_embs,
            )
        )

        pred_id_logits[
            :,
            self.obj_nums + 1:,
        ] = -1e10

        if output_size is not None:
            pred_id_logits = F.interpolate(
                pred_id_logits,
                size=output_size,
                mode="bilinear",
                align_corners=True,
            )

        self.pred_id_logits = pred_id_logits

        return pred_id_logits

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def restart_engine(self):
        self.batch_size = 1
        self.frame_step = 0
        self.last_mem_step = -1

        self.obj_nums = None

        self.enc_size_2d = None
        self.enc_hw = None
        self.input_size_2d = None

        self.long_term_memories = None

        self.short_term_memories_list = []
        self.short_term_memories = None

        self.curr_enc_embs = None
        self.curr_id_embs = None
        self.curr_one_hot_mask = None
        self.curr_lstt_output = None

        self.pred_id_logits = None

    def update_size(
        self,
        input_size,
        enc_size,
    ):
        self.input_size_2d = input_size
        self.enc_size_2d = enc_size
        self.enc_hw = (
            enc_size[0] * enc_size[1]
        )