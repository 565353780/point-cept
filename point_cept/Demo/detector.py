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

    feats, coords, grid_coords = detector.encodePoints(points)
    for i in range(len(feats)):
        print(f'feat {i}:', feats[i].shape)
        print(f'coord {i}:', coords[i].shape, coords[i].min(), coords[i].max())
        print(f'grid_coord {i}:', grid_coords[i].shape, grid_coords[i].min(), grid_coords[i].max())

        coord_pcd = toPcd(coords[i])
        grid_coord_pcd = toPcd(grid_coords[i])

        o3d.io.write_point_cloud(data_folder_path + f'coord_{i:06d}.ply', coord_pcd, write_ascii=True)
        o3d.io.write_point_cloud(data_folder_path + f'grid_coord_{i:06d}.ply', grid_coord_pcd, write_ascii=True)

    # trainable
    perceiver_resampler = PerceiverResampler(
        dim_in=576,
        num_latents=512,
        dim_out=1024,
    ).to(device)

    pt_tokens = perceiver_resampler(feats)

    print('token:', pt_tokens.shape)
    return True
