import types
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from .ops import Voxelization


def _ensure_float_lidar_bev(x: Tensor) -> Tensor:
    """BEV tensor for ``pts_backbone`` / pytorch_quantization must be float.

    ``pts_middle_encoder`` can return a quantized-per-channel/tensor or integer
    dense map; ``TensorQuantizer`` then raises
    "Quantize only works on Float Tensor, got Int".
    """
    if not isinstance(x, torch.Tensor):
        return x
    if getattr(x, "is_quantized", False):
        return x.dequantize()
    if x.is_floating_point():
        return x
    try:
        if x.dtype in (torch.qint8, torch.quint8, torch.qint32):
            return x.dequantize()
    except (AttributeError, RuntimeError, TypeError):
        pass
    return x.float()


def register_pts_middle_encoder_float_input_hook(encoder: Optional[torch.nn.Module]) -> None:
    """Ensure voxel feature tensor is FP32 before sparse encoder forward.

    Some traced/optimized sparse paths can run ``quantize_per_tensor`` on the
    forward argument before any inlined ``to(float32)`` node, so integer voxel
    features raise ``RuntimeError: ... got Int``.

    ``forward_pre_hook`` is not reliably invoked for this path, so we wrap
    instance ``forward``: the wrapper runs first and then calls the original
    ``forward`` with FP32 ``feats``.
    """
    if encoder is None:
        return
    if getattr(encoder, "_awml_float_voxel_feat_patched", False):
        return

    orig_forward = encoder.forward

    def forward_with_float_voxel_feats(self, feats, coords, batch_size):
        if isinstance(feats, torch.Tensor):
            if not feats.is_floating_point():
                feats = feats.float()
            else:
                feats = feats.to(dtype=torch.float32)
            feats = feats.contiguous()
        return orig_forward(feats, coords, batch_size)

    encoder.forward = types.MethodType(forward_with_float_voxel_feats, encoder)
    encoder._awml_float_voxel_feat_patched = True


def _ensure_float_for_pts_pipeline(x: Tensor) -> Tensor:
    """Float BEV before dense calib/QDQ."""
    if not isinstance(x, torch.Tensor):
        return x
    return _ensure_float_lidar_bev(x)


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        voxelize_cfg: Optional[dict] = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Initialize BEVFusion model.

        Args:
            data_preprocessor (dict): Data preprocessor config.
            voxelize_cfg (dict): Voxelization config.
            pts_voxel_encoder (dict): Point voxel encoder config.
            pts_middle_encoder (dict): Point middle encoder config.
            fusion_layer (dict): Fusion layer config.
            img_backbone (dict): Image backbone config.
            img_neck (dict): Image neck config.
            pts_backbone (dict): Point backbone config.
            pts_neck (dict): Point neck config.
            bbox_head (dict): Bbox head config.
            init_cfg (dict): Initialization config.
            seg_head (dict): Segmentation head config.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if voxelize_cfg is not None:
            self.voxelize_reduce = voxelize_cfg.pop("voxelize_reduce")
            self.pts_voxel_layer = Voxelization(**voxelize_cfg)
            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
            self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        else:
            self.voxelize_reduce = False
            self.pts_voxel_layer = None
            self.pts_voxel_encoder = None
            self.pts_middle_encoder = None

        self.img_backbone = MODELS.build(img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(view_transform) if view_transform is not None else None

        self.fusion_layer = MODELS.build(fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone) if pts_backbone is not None else None
        self.pts_neck = MODELS.build(pts_neck) if pts_neck is not None else None

        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()
        register_pts_middle_encoder_float_input_hook(self.pts_middle_encoder)

    def _align_lidar_bev_to_head_grid(self, feats):
        """Resize pts_backbone+neck BEV maps to match ``bbox_head.bev_pos`` resolution.

        ``BEVFusionHead`` builds ``bev_pos`` from ``test_cfg['grid_size'] // out_size_factor``
        (e.g. 1440//8 → 180). Heatmap/top-k indices assume ``H*W == len(bev_pos)``.

        If the sparse tower exposes **full voxel resolution** (e.g. 1440×1440) into SECOND
        (``dense()`` / ``spatial_shape`` bugs), FPN output can be hundreds of pixels
        per side while ``bev_pos`` stays 180×180 → ``gather`` uses out-of-range indices
        (CUDA scatter/gather assert) and later ops may report missing backends.

        Pooling here only runs when ``H,W`` differ from the head grid; normal training paths
        keep the correct encoder stride and are unchanged.
        """
        head = getattr(self, "bbox_head", None)
        if head is None or not hasattr(head, "test_cfg") or head.test_cfg is None:
            return feats
        try:
            grid = head.test_cfg["grid_size"]
            osf = int(head.test_cfg["out_size_factor"])
            gh = int(grid[0] // osf)
            gw = int(grid[1] // osf)
        except Exception:
            return feats

        def _pool(t: Tensor) -> Tensor:
            if t.dim() != 4:
                return t
            _, _, h, w = t.shape
            if int(h) == gh and int(w) == gw:
                return t
            return F.adaptive_avg_pool2d(t, (gh, gw))

        if isinstance(feats, Tensor):
            return _pool(feats)
        if isinstance(feats, (list, tuple)):
            return type(feats)(_pool(t) if isinstance(t, Tensor) else t for t in feats)
        return feats

    def _forward(
        self, batch_inputs_dict: Tensor, batch_data_samples: OptSampleList = [], using_image_features=False, **kwargs
    ):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """

        # NOTE(knzo25): this is used during onnx export
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas, using_image_features)

        if self.with_bbox_head:
            outputs = self.bbox_head(feats, batch_input_metas)

        return outputs[0][0]

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(value for key, value in log_vars if "loss" in key)
        log_vars.insert(0, ["loss", loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head."""
        return hasattr(self, "seg_head") and self.seg_head is not None

    def get_image_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        assert BN == B * N, (BN, B * N)
        x = x.view(B, N, C, H, W)
        return x

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        camera_intrinsics_inverse=None,
        img_aug_matrix_inverse=None,
        lidar_aug_matrix_inverse=None,
        geom_feats=None,
        using_image_features=False,
    ) -> torch.Tensor:

        if not using_image_features:
            x = self.get_image_backbone_features(x)

        with torch.amp.autocast("cuda", enabled=False):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
                camera_intrinsics_inverse,
                img_aug_matrix_inverse,
                lidar_aug_matrix_inverse,
                geom_feats,
            )
        return x

    def extract_pts_feat(self, feats, coords, sizes, points=None) -> torch.Tensor:
        if points is not None:
            # NOTE(knzo25): training and normal inference
            with torch.amp.autocast("cuda", enabled=False):
                points = [point.float() for point in points]
                feats, coords, sizes = self.voxelize(points)
                # Use Python int for ``batch_size``; 0-dim int tensor can confuse traced graphs.
                batch_size = max(int((coords[-1, 0] + 1).item()), 1)
        else:
            # NOTE(knzo25): onnx inference. Voxelization happens outside the graph
            with torch.amp.autocast("cuda", enabled=False):
                assert self.voxelize_reduce
                if self.voxelize_reduce:
                    # Avoid 0/0 → NaN in TRT / ONNX if any voxel reports num_points==0 (rare but possible).
                    sz = sizes.type_as(feats).view(-1, 1).clamp(min=1.0)
                    feats = feats.sum(dim=1, keepdim=False) / sz

                # spconv INT8 / torch.quantize_per_tensor requires float activations; voxel grids may be integer
                feats = feats.to(dtype=torch.float32)
                # spconv + torch.jit ONNX trace: avoid aliasing traced tensors; enforce int32 indices
                feats = feats.contiguous()
                coords = coords.contiguous().to(dtype=torch.int32)
                if torch.jit.is_tracing():
                    feats = feats.clone()
                    coords = coords.clone()
                # batch index column must match batch_size passed to SparseConvTensor (hardcoding 1 is wrong if coors hold larger batch ids)
                batch_size = int(coords[:, 0].max().item()) + 1
                batch_size = max(batch_size, 1)
        # Sparse quantize nodes require float features; integer voxel grids can raise:
        # "Quantize only works on Float Tensor, got Int" inside pts_middle_encoder.
        feats = feats.to(dtype=torch.float32)
        x = self.pts_middle_encoder(feats, coords, batch_size)
        # INT8 sparse tower → dense BEV may be qint*/int; dense Q/DQ and calibrators need float.
        # Do not gate on tracing state flags; they can stay True in later runs.
        x = _ensure_float_for_pts_pipeline(x)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                sz = sizes.type_as(feats).view(-1, 1).clamp(min=1.0)
                feats = feats.sum(dim=1, keepdim=False) / sz
                feats = feats.contiguous()

        if isinstance(feats, torch.Tensor) and not feats.is_floating_point():
            feats = feats.float()

        return feats, coords, sizes

    def predict(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
        using_image_features=False,
        **kwargs,
    ) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas, using_image_features)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        using_image_features,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get("imgs", None)
        points = batch_inputs_dict.get("points", None)
        features = []

        is_onnx_inference = False
        if imgs is not None and "lidar2img" not in batch_inputs_dict:
            # NOTE(knzo25): normal training and testing
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta["lidar2img"])
                camera_intrinsics.append(meta["cam2img"])
                camera2lidar.append(meta["cam2lidar"])
                img_aug_matrix.append(meta.get("img_aug_matrix", np.eye(4)))
                lidar_aug_matrix.append(meta.get("lidar_aug_matrix", np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(
                imgs,
                deepcopy(points),
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
                using_image_features=using_image_features,
            )
            features.append(img_feature)
        elif imgs is not None:
            # NOTE(knzo25): onnx inference
            is_onnx_inference = True
            lidar2image = batch_inputs_dict["lidar2img"]
            camera_intrinsics = batch_inputs_dict["cam2img"]
            camera2lidar = batch_inputs_dict["cam2lidar"]
            img_aug_matrix = batch_inputs_dict["img_aug_matrix"]
            lidar_aug_matrix = batch_inputs_dict["lidar_aug_matrix"]
            geom_feats = batch_inputs_dict["geom_feats"]

            img_feature = self.extract_img_feat(
                imgs,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
                geom_feats=geom_feats,
                using_image_features=using_image_features,
            )
            features.append(img_feature)

        if self.pts_middle_encoder is not None:
            pts_feature = self.extract_pts_feat(
                batch_inputs_dict.get("voxels", {}).get("voxels", None),
                batch_inputs_dict.get("voxels", {}).get("coors", None),
                batch_inputs_dict.get("voxels", {}).get("num_points_per_voxel", None),
                points=points if not is_onnx_inference else None,
            )
            features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        if self.pts_backbone is not None:
            x = _ensure_float_for_pts_pipeline(x)
            x = self.pts_backbone(x)

        if self.pts_neck is not None:
            x = self.pts_neck(x)

        x = self._align_lidar_bev_to_head_grid(x)

        return x

    def loss(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
        using_image_features: bool = False,
        **kwargs,
    ) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas, using_image_features)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        return losses
