import sys
sys.path.append('../camera-control/')
sys.path.append('../../../camera-control/')

import os
import torch
import open3d as o3d

from point_cept.Model.utonia.data import load
from point_cept.Method.pca import get_pca_color
from point_cept.Module.detector import Detector


if __name__ == "__main__":
    model_file_path = os.environ['HOME'] + '/chLi/Model/Utonia/utonia.pth'
    device = 'cuda:0'

    detector = Detector(
        model_file_path=model_file_path,
        device=device,
    )

    point = load("sample3_object")
    points = point['coord'][:, [0, 2, 1]]

    point = detector.encodePoints([points])[0]

    # upcast point feature
    # Point is a structure contains all the information during forward
    # Use range(4) to upcast features from all levels for quantitative evaluation
    for _ in range(2):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent

    batched_coord = point.coord.clone()
    batched_color = point.color.clone()
    pca_color = get_pca_color(point.feat, brightness=1.2, center=True)

    data_folder_path = '/home/lichanghao/chLi/Dataset/pixel_align/test/'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(batched_coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
    o3d.io.write_point_cloud(data_folder_path + 'pca.ply', pcd, write_ascii=True)
