import os
import torch

from point_cept.Model.sonata.data import load as load_data
from point_cept.Model.sonata.model import load as load_model


def test():
    model_file_path = "/home/chli/chLi/Model/Sonata/pretrain-sonata-v1m1-0-base.pth"
    if not os.path.exists(model_file_path):
        print("[ERROR][test::sonata]")
        print("\t model file not exist!")
        print("\t model_file_path:", model_file_path)
        return False

    points = [
        torch.rand(600, 3).cuda(),
        torch.rand(200, 3).cuda(),
        torch.rand(300, 3).cuda(),
        torch.rand(400, 3).cuda(),
    ]

    coords = torch.cat(points, dim=0)

    batch_indices = [
        torch.full((t.shape[0],), i, dtype=torch.long) for i, t in enumerate(points)
    ]
    batch = torch.cat(batch_indices, dim=0).cuda()

    data = {
        "coord": coords,
        "feat": coords,
        "batch": batch,
        "grid_size": 0.01,
    }

    data = load_data("sample1")

    sonata = load_model(model_file_path).cuda()

    point = sonata(data)

    feature = point.feat.reshape(points.shape[0], points.shape[1], -1)

    print("point feature shape: ", feature.shape)
    return True
