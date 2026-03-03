import os
import torch
import numpy as np

from typing import List

from flux_mv.Model.concerto.model import load
from flux_mv.Model.concerto.transform import Compose


class Detector(object):
    def __init__(
        self,
        model_file_path: str,
        device: str='cuda:0',
    ) -> None:
        self.device = device

        assert os.path.exists(model_file_path)

        self.ptv3_encoder = load(model_file_path)
        self.ptv3_encoder.to(device)

        self.ptv3_transform = Compose([
            dict(type="Update", keys_dict={"index_valid_keys": ["coord", "color", "normal", "batch"]}),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse", "batch"),
                feat_keys=("coord", "color", "normal"),
            ),
        ])
        return

    def encodePoints(
        self,
        points: List[np.ndarray], # Nix3
    ) -> torch.Tensor:
        coords = np.concatenate(points, axis=0)

        batch_indices = [
            torch.full((t.shape[0],), i, dtype=torch.long) for i, t in enumerate(points)
        ]
        batch = torch.cat(batch_indices, dim=0).cuda()

        point = {
            "coord": coords,
            "color": np.zeros_like(coords),
            "normal": np.zeros_like(coords),
            "batch": batch,
        }

        point = self.ptv3_transform(point)

        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].to(self.device, non_blocking=True)

        point = self.ptv3_encoder(point)

        feature = point.feat
        batch_ids = point.batch

        features_list = []
        for i in range(len(points)):
            mask = batch_ids == i
            features_list.append(feature[mask])

        return features_list
