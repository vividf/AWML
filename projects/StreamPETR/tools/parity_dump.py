"""Dump a StreamPETR parity reference from the AWML (mm) stack.

Runs one real validation frame through the AWML model in float32 eval mode
(no DN, no GridMask, no dropout, flash-attention replaced by an equivalent
fp32 SDPA) and saves inputs, ground truth, intermediates, outputs, and every
loss term. The companion checker in autoware-ml
(``autoware_ml/tools/streampetr_parity_check.py``) replays the same tensors
through the native model with converted weights and compares.

Run inside the AWML docker:

    python projects/StreamPETR/tools/parity_dump.py \
        --config projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_7_traffic_barrier_j6gen2_partialignore.py \
        --checkpoint work_dirs/streampetr_2_7/epoch_20.pth \
        --output work_dirs/parity/streampetr_parity_reference.pt \
        [--sample-index 0] [--inspect]
"""

import argparse
import copy
import sys
import types

import torch
import torch.nn.functional as F


def _install_flash_attn_stub():
    """Provide inert flash_attn modules when the package is absent.

    The parity dump replaces ``FlashMHA.forward`` with an fp32 SDPA before
    any forward pass, so the flash kernels themselves are never invoked;
    only the import statements in ``stream_petr`` need to succeed.
    """
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        return

    def _stub(*args, **kwargs):
        raise RuntimeError(
            "flash_attn stub was called; the parity dump should have replaced "
            "FlashMHA.forward before any forward pass."
        )

    from importlib.machinery import ModuleSpec

    def _make_module(name):
        module = types.ModuleType(name)
        # transformers probes flash_attn.__spec__; a bare ModuleType has None.
        module.__spec__ = ModuleSpec(name, loader=None)
        return module

    root = _make_module("flash_attn")
    root.__version__ = "0.0.0-parity-stub"
    bert_padding = _make_module("flash_attn.bert_padding")
    bert_padding.unpad_input = _stub
    bert_padding.pad_input = _stub
    interface = _make_module("flash_attn.flash_attn_interface")
    interface.flash_attn_varlen_kvpacked_func = _stub
    root.bert_padding = bert_padding
    root.flash_attn_interface = interface
    sys.modules["flash_attn"] = root
    sys.modules["flash_attn.bert_padding"] = bert_padding
    sys.modules["flash_attn.flash_attn_interface"] = interface


_install_flash_attn_stub()

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint


def _fp32_sdpa_forward(self, q, k, v, key_padding_mask=None):
    """Replace FlashMHA.forward with an exact fp32 SDPA using the same weights."""
    assert key_padding_mask is None
    num_heads = self.num_heads
    q_proj, k_proj, v_proj = F._in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)

    def split_heads(x):
        batch, seq, _ = x.shape
        return x.view(batch, seq, num_heads, -1).transpose(1, 2)

    attended = F.scaled_dot_product_attention(split_heads(q_proj), split_heads(k_proj), split_heads(v_proj))
    batch, _, seq, _ = attended.shape
    merged = attended.transpose(1, 2).reshape(batch, seq, -1)
    return self.out_proj(merged), None


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        # float64 arrays (2D annotations) shrink to float32; integer label
        # tensors must keep their dtype for downstream indexing.
        return value.float() if value.is_floating_point() else value
    if isinstance(value, (list, tuple)):
        return [_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--inspect", action="store_true", help="Print the batch structure and exit.")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope("mmdet3d")
    if cfg.get("custom_imports"):
        from mmengine.utils import import_modules_from_strings

        import_modules_from_strings(**cfg.custom_imports)

    # Deterministic single-frame batch from the val split (test_mode: no
    # camera shuffle, no augmentation). num_workers=0 keeps it in-process.
    dataloader_cfg = copy.deepcopy(cfg.val_dataloader)
    dataloader_cfg.num_workers = 0
    dataloader_cfg.persistent_workers = False
    dataloader = Runner.build_dataloader(dataloader_cfg)
    dataset = dataloader.dataset
    sample = dataset[args.sample_index]
    batch = dataloader.collate_fn([sample])

    if args.inspect:

        def describe(value, indent=0):
            pad = " " * indent
            if isinstance(value, torch.Tensor):
                print(f"{pad}Tensor{tuple(value.shape)} {value.dtype}")
            elif isinstance(value, dict):
                for key, item in value.items():
                    print(f"{pad}{key}:")
                    describe(item, indent + 2)
            elif isinstance(value, (list, tuple)):
                print(f"{pad}{type(value).__name__}[{len(value)}]")
                for item in value[:2]:
                    describe(item, indent + 2)
            else:
                print(f"{pad}{type(value).__name__}: {value}")

        describe(batch)
        return

    from mmdet3d.registry import MODELS

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = model.cuda().float()
    # Petr3D.train() does not return self, so never chain .eval().
    model.eval()
    model.use_grid_mask = False
    model.pts_bbox_head.with_dn = False

    # Exact-math fp32 attention: flash kernels always cast to fp16 internally.
    from projects.StreamPETR.stream_petr.models.utils.attention import FlashMHA

    FlashMHA.forward = _fp32_sdpa_forward

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.cuda()
        if isinstance(value, (list, tuple)):
            return [move(item) for item in value]
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if hasattr(value, "to") and hasattr(value, "tensor"):  # LiDARInstance3DBoxes
            return value.to("cuda")
        return value

    batch = move(batch)

    # Frame 0 of the window (T=1 everywhere in the target recipe).
    # pseudo_collate leaves tensors as per-sample lists; Petr3D.forward's
    # stack_tensors stacks them into (B, T, ...) — do the same here.
    data = {}
    for key in [
        "img",
        "lidar2img",
        "intrinsics",
        "extrinsics",
        "timestamp",
        "img_timestamp",
        "ego_pose",
        "ego_pose_inv",
        "prev_exists",
    ]:
        value = batch[key]
        data[key] = torch.stack(value, dim=0) if isinstance(value, list) else value
    # pseudo_collate transposes per-sample time lists: index [0] selects
    # frame 0 of the (T=1) window, exactly like obtain_history_memory does.
    img_metas = batch["img_metas"]
    gt_bboxes_3d = batch["gt_bboxes_3d"]
    gt_labels_3d = batch["gt_labels_3d"]
    gt_bboxes_2d = batch["gt_bboxes"]  # frame -> camera -> per-sample tensors
    gt_labels_2d = batch["gt_bboxes_labels"]
    centers_2d = batch["centers_2d"]
    depths_2d = batch["depths"]

    model.pts_bbox_head.reset_memory()

    captured = {}
    original_position = model.pts_bbox_head.position_embeding

    def capture_position(*pos_args, **pos_kwargs):
        pos_embed, cone = original_position(*pos_args, **pos_kwargs)
        captured["pos_embed"] = pos_embed
        captured["cone"] = cone
        return pos_embed, cone

    model.pts_bbox_head.position_embeding = capture_position

    with torch.no_grad():
        # Match forward_train's data plumbing for one frame.
        rec_img = data["img"][:, -1:]
        img_feats = model.extract_feat(rec_img, 1)
        data_t = {key: value[:, -1] for key, value in data.items() if key != "img"}
        data_t["img_feats"] = img_feats[:, -1]

        metas_t = img_metas[0]
        location = model.prepare_location([x[0] for x in metas_t["pad_shape"]], **data_t)
        # The mm head mutates `location` in place (pixel-scaling inside
        # position_embeding), so keep a pristine copy for the dump.
        location_snapshot = location.clone()
        outs_roi = model.img_roi_head(location, **data_t)
        # mm's clip_sigmoid uses in-place sigmoid_ inside the roi loss, which
        # would silently turn these logits into probabilities; snapshot first.
        outs_roi_snapshot = {key: value.clone() for key, value in outs_roi.items() if isinstance(value, torch.Tensor)}
        outs = model.pts_bbox_head(location, metas_t, None, gt_bboxes_3d[0], gt_labels_3d[0], **data_t)

        losses_3d = model.pts_bbox_head.loss(gt_bboxes_3d[0], gt_labels_3d[0], outs, img_metas=metas_t)
        losses_2d = model.img_roi_head.loss(
            gt_bboxes_2d[0], gt_labels_2d[0], centers_2d[0], depths_2d[0], outs_roi, metas_t
        )

    gt_tensors = [boxes.tensor for boxes in gt_bboxes_3d[0]]
    reference = {
        "inputs": {
            "img": data["img"][:, -1],
            "intrinsics": data_t["intrinsics"],
            "extrinsics": data_t["extrinsics"],
            "lidar2img": data_t["lidar2img"],
            "timestamp": data_t["timestamp"],
            "prev_exists": data_t["prev_exists"],
            "ego_pose": data_t["ego_pose"],
            "ego_pose_inv": data_t["ego_pose_inv"],
            "pad_shape": [x[0] for x in metas_t["pad_shape"]],
        },
        "gt": {
            "gt_boxes": gt_tensors,
            "gt_labels": gt_labels_3d[0],
            "gt_bboxes_2d": gt_bboxes_2d[0],
            "gt_labels_2d": gt_labels_2d[0],
            "centers_2d": centers_2d[0],
            "depths_2d": depths_2d[0],
            "traffic_cone_barrier_status": metas_t.get("traffic_cone_barrier_status", [True]),
        },
        "intermediates": {
            "img_feats": data_t["img_feats"],
            "pos_embed": captured.get("pos_embed"),
            "cone": captured.get("cone"),
            "location": location_snapshot,
        },
        "outputs": {
            "all_cls_scores": outs["all_cls_scores"],
            "all_bbox_preds": outs["all_bbox_preds"],
            "roi_enc_cls_scores": outs_roi_snapshot["enc_cls_scores"],
            "roi_enc_bbox_preds": outs_roi_snapshot["enc_bbox_preds"],
            "roi_pred_centers2d": outs_roi_snapshot["pred_centers2d"],
            "roi_centerness": outs_roi_snapshot["centerness"],
        },
        "losses": {**losses_3d, **losses_2d},
        "meta": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "sample_index": args.sample_index,
            "with_dn": False,
        },
    }

    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(_to_cpu(reference), args.output)
    print(f"Wrote parity reference to {args.output}")
    for name, value in reference["losses"].items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: {value.item():.6f}")


if __name__ == "__main__":
    main()
