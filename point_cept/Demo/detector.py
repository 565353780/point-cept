import os
import numpy as np

from point_cept.Module.detector import Detector


def demo():
    model_file_path = os.environ['HOME'] + '/chLi/Model/Concerto/concerto_large.pth'
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

    feats = detector.encodePoints(points)
    for i in range(len(feats)):
        print(f'feat {i}:', feats[i].shape)
    return True
