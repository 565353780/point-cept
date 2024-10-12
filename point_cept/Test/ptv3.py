import torch

from pointcept.models.point_transformer_v3 import PointTransformerV3



def test():
    points = torch.rand((400, 3), dtype=torch.float32, device='cuda')

    data = {
        'coord': points,
        'feat': points,
        'batch': torch.zeros(points.shape[0], dtype=torch.long, device='cuda'),
        'grid_size': 0.01,
    }

    ptv3 = PointTransformerV3(3).cuda()

    point = ptv3(data)

    print('point.feat.shape: ', point.feat.shape)
    return True
