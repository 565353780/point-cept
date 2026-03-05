import sys
sys.path.append('../camera-control/')
sys.path.append('../../../camera-control/')

import os
import numpy as np
import open3d as o3d

from camera_control.Method.io import loadMeshFile
from camera_control.Method.pcd import toPcd

from point_cept.Model.perceiver_resampler import PerceiverResampler
from point_cept.Module.detector import Detector


def demo():
    model_file_path = os.environ['HOME'] + '/chLi/Model/Utonia/utonia.pth'
    device = 'cuda:0'

    detector = Detector(
        model_file_path=model_file_path,
        device=device,
    )

    points = [
        np.random.rand(600, 3),
        np.random.rand(200, 3),
        np.random.rand(300, 3),
        np.random.rand(400, 3),
    ]

    data_folder_path = '/home/lichanghao/chLi/Dataset/pixel_align/0228c5cdba8393cd4d947ac2e915f769f684c73b87e6939c129611ba665cafcb/'
    mesh_file_path = data_folder_path + 'gt_normalized.ply'

    mesh = loadMeshFile(mesh_file_path)

    points = [mesh.vertices]

    point = detector.encodePoints(points)

    print('batch:', point.batch.shape)

    feature_list = []
    grid_coord_list = []
    for i in range(len(points)):
        mask = point.batch == i
        feature_list.append(point.feat[mask])
        grid_coord_list.append(point.grid_coord[mask])

    for i in range(len(feature_list)):
        print(f'feat {i}:', feature_list[i].shape)
        print(f'grid_coord {i}:', grid_coord_list[i].shape, grid_coord_list[i].min(), grid_coord_list[i].max())

        grid_coord_pcd = toPcd(grid_coord_list[i])

        o3d.io.write_point_cloud(data_folder_path + f'grid_coord_{i:06d}.ply', grid_coord_pcd, write_ascii=True)

    # trainable
    perceiver_resampler = PerceiverResampler(
        dim_in=576,
        num_latents=512,
        dim_out=1024,
    ).to(device)

    pt_tokens = perceiver_resampler(feature_list)

    print('token:', pt_tokens.shape)
    return True
