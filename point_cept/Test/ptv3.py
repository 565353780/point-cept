import torch

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)


def test():
    points = torch.rand((2, 400, 3), dtype=torch.float32, device="cuda")

    data = {
        "coord": points.reshape(-1, 3),
        "feat": points.reshape(-1, 3),
        "batch": torch.cat(
            [
                torch.ones(points.shape[1], dtype=torch.long, device="cuda") * i
                for i in range(points.shape[0])
            ]
        ),
        "grid_size": 0.01,
    }

    ptv3 = PointTransformerV3(3).cuda()

    point = ptv3(data)

    feature = point.feat.reshape(points.shape[0], points.shape[1], -1)

    print("point feature shape: ", feature.shape)
    return True
