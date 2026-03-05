import os
import torch
import numpy as np

from typing import List, Union, Dict

from point_cept.Model.utonia.structure import Point
from point_cept.Model.utonia.model import load
from point_cept.Model.utonia.transform import Compose


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
        self.ptv3_encoder.eval()

        self.ptv3_transform = Compose([
            dict(type="Update", keys_dict={"index_valid_keys": ["coord", "color", "normal", "batch"]}),
            dict(type="NormalizeCoord"),
            #dict(type="RandomScale", scale=[scale, scale]),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse", "batch"),
                feat_keys=("coord", "color", "normal"),
            ),
        ])
        return

    def detect(self, point: Dict) -> Point:
        point = self.ptv3_transform(point)

        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].to(self.device, non_blocking=True)

        point = self.ptv3_encoder(point)
        return point

    def encodePoints(
        self,
        points: Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]], # BxNx3 or Nix3
    ) -> Point:
        # 判断输入类型，标准化为 List[np.ndarray] 格式
        if isinstance(points, (torch.Tensor, np.ndarray)):  # e.g. torch.Tensor[B, N, 3] or np.ndarray[B, N, 3]
            if points.ndim == 3:
                # BxNx3
                B = points.shape[0]
                point_list = []
                for b in range(B):
                    p = points[b]
                    if isinstance(p, torch.Tensor):
                        point_list.append(p.cpu().numpy())
                    else:
                        point_list.append(p)
                coords = np.concatenate(point_list, axis=0)
                batch = torch.cat([
                    torch.full((point_list[bi].shape[0],), bi, dtype=torch.long) for bi in range(B)
                ], dim=0).to(self.device)
            elif points.ndim == 2:
                # 单个点云 Nx3
                if isinstance(points, torch.Tensor):
                    coords = points.cpu().numpy()
                else:
                    coords = points
                batch = torch.zeros((coords.shape[0],), dtype=torch.long).to(self.device)
            else:
                raise ValueError(f'Input tensor/array of shape {points.shape} is not supported.')
        elif isinstance(points, list):
            # List of torch.Tensor or np.ndarray, each shape Ni x 3
            point_list = []
            for p in points:
                if isinstance(p, torch.Tensor):
                    point_list.append(p.cpu().numpy())
                else:
                    point_list.append(p)
            coords = np.concatenate(point_list, axis=0)
            batch = torch.cat([
                torch.full((point_list[i].shape[0],), i, dtype=torch.long) for i in range(len(point_list))
            ], dim=0).to(self.device)
        else:
            raise TypeError("points must be torch.Tensor, np.ndarray, or List of them")

        point = {
            "coord": coords,
            "color": np.zeros_like(coords),
            "normal": np.zeros_like(coords),
            "batch": batch,
        }

        point = self.detect(point)

        return point
