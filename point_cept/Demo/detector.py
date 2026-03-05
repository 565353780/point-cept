import os
import numpy as np

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

    mesh_file_path = ''

    feats, coords, grid_coords = detector.encodePoints(points)
    for i in range(len(feats)):
        print(f'feat {i}:', feats[i].shape)
        print(f'coord {i}:', coords[i].shape, coords[i].min(), coords[i].max())
        print(f'grid_coord {i}:', grid_coords[i].shape, grid_coords[i].min(), grid_coords[i].max())

    # trainable
    perceiver_resampler = PerceiverResampler(
        dim_in=576,
        num_latents=512,
        dim_out=1024,
    ).to(device)

    pt_tokens = perceiver_resampler(feats)

    print('token:', pt_tokens.shape)
    return True
