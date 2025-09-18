import torch

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)


def test():
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

    ptv3 = PointTransformerV3(3).cuda()

    point = ptv3(data)

    feature = point.feat

    print("coords shape: ", coords.shape)
    print("point feature shape: ", feature.shape)
    return True
