import numpy as np
import scipy.spatial
import skimage.draw
import torch
from torchvision import io
import face_alignment

import matplotlib.pyplot as plt


def interpolate_from_landmarks(image, landmarks, vertex_indices=None, weights=None, mask=None):
    H, W = image.shape[-2:]
    step = 4

    rect = landmarks.new_tensor([[0, 0], [H, 0], [0, W], [H, W]])
    vertices = torch.cat([landmarks, rect], dim=0)
    vertices_cpu = vertices.cpu().detach()

    if vertex_indices is None:
        delaunay = scipy.spatial.Delaunay(vertices_cpu)
        triangles = delaunay.simplices
        
        facet_map = np.full([H, W], -1, dtype=np.int32)
        for index, triangle in enumerate(triangles):
            points = vertices_cpu[triangle]

            rr, cc = skimage.draw.polygon(points[:,0], points[:,1], [H - 1, W - 1])
            facet_map[rr, cc] = index

        facet_map = torch.from_numpy(facet_map).long().to(image.device)
        triangles = torch.from_numpy(triangles).long().to(image.device)
        grid0, grid1 = torch.meshgrid(torch.arange(H), torch.arange(W))
        grids = torch.stack([grid0, grid1], dim=-1).to(image.device)
        valid = facet_map >= 0
        if mask is not None:
            valid = valid & mask
        facet_map = facet_map[valid]
        grids = grids[valid]
        N = len(facet_map)

        # N -> N x 1 x 3
        facet_map = facet_map[..., None, None].expand(-1, 1, 3)
        # F x 3 -> N x F x 3
        expanded = triangles[None, ...].expand(N, -1, -1)
        # N x 1 x 3
        vertex_indices = torch.gather(expanded, dim=1, index=facet_map)
        # N x 1 x 3 -> N x 3
        vertex_indices = vertex_indices.squeeze(1)
    else:
        assert mask is None

    N = len(vertex_indices)
    # N x 3 -> N x 3 x 2
    expanded = vertex_indices[..., None].expand(-1, -1, 2)
    # V x 2 -> N x V x 2 
    vertices = vertices[None, ...].expand(N, -1, -1)
    # N x 3 x 2
    vertices = torch.gather(vertices, dim=1, index=expanded)
        
    if weights is None:
        with torch.no_grad():
            # https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates/63203#63203
            v0 = vertices[:, 1, :] - vertices[:, 0, :]
            v1 = vertices[:, 2, :] - vertices[:, 0, :]
            v2 = grids - vertices[:, 0, :]
            den = v0[:, 1] * v1[:, 0] - v1[:, 1] * v0[:, 0]
            v = (v2[:, 1] * v1[:, 0] - v1[:, 1] * v2[:, 0]) / den
            w = (v0[:, 1] * v2[:, 0] - v2[:, 1] * v0[:, 0]) / den
            u = 1. - v - w

        weights = torch.stack([u, v, w], dim=-1)

    interpolated = (vertices * weights.unsqueeze(-1)).sum(dim=1)

    if False:
        if not hasattr(interpolate_from_landmarks, 'triangles'):
            interpolate_from_landmarks.triangles = triangles
        if not hasattr(interpolate_from_landmarks, 'save_index'):
            interpolate_from_landmarks.save_index = 0

        f = plt.figure(figsize=(3, 3))
        plt.imshow(image.permute(1, 2, 0).cpu().detach())
        for index, triangle in enumerate(interpolate_from_landmarks.triangles):
            points = vertices_cpu[triangle]
            points = points - 0.5
            plt.plot(points[:, 1], points[:, 0], c='g')

        # mask = facet_map[:, 0, 0] == 100
        # vertices = vertices[mask].cpu().detach()
        # interpolated = interpolated[mask].cpu().detach()
        np.random.seed(12)
        selected = np.random.choice(len(interpolated), 100)
        selected = interpolated[selected, :].cpu().detach()
        selected = selected - 0.5

        # plt.figure(figsize=(16, 12))
        # plt.imshow(image.permute(1, 2, 0).cpu().detach())
        plt.scatter(x=selected[:, 1], y=selected[:, 0], c='r')
        # plt.scatter(x=vertices[0, :, 1], y=vertices[0, :, 0], c='g')

        f.gca().set_axis_off()
        f.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        f.gca().xaxis.set_major_locator(plt.NullLocator())
        f.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.show()
        f.savefig(f'/opt_/cephfs_workspace/gpudisk/libo427/workspace/attractive/figs/blend_{interpolate_from_landmarks.save_index}.pdf',
                bbox_inches='tight', pad_inches=0)
        interpolate_from_landmarks.save_index += 1

    # N x 2, N x 3, N x 3
    return interpolated, vertex_indices, weights

if __name__ == '__main__':
    image = io.read_image('research/jackiechan.png')
    im = image.permute(1, 2, 0).numpy()
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    landmarks = fa.get_landmarks(im)[0]
    landmarks = torch.from_numpy(landmarks).flip(dims=(-1,))
    
    interpolated, vertex_indices, weights = interpolate_from_landmarks(image, landmarks)

    import pdb; pdb.set_trace()


