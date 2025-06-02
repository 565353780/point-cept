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

    points = torch.rand((2, 400, 3), dtype=torch.float32, device="cuda")

    data = {
        "coord": points.reshape(-1, 3),
        "batch": torch.cat(
            [
                torch.ones(points.shape[1], dtype=torch.long, device="cuda") * i
                for i in range(points.shape[0])
            ]
        ),
    }

    data = load_data("sample1")

    sonata = load_model(model_file_path).cuda()

    point = sonata(data)

    feature = point.feat.reshape(points.shape[0], points.shape[1], -1)

    print("point feature shape: ", feature.shape)
    return True
