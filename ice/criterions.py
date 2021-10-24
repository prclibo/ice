import matplotlib.pyplot as plt

import kornia
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from wrapper import StyleGanWrapper, FaceSegmenter, KeyPointDetector
from landmark_interpolation import interpolate_from_landmarks

def masked_mean(x, mask):
    B, C, H, W = x.shape
    assert mask.shape == (B, 1, H, W) and mask.dtype == torch.bool

    x = x * mask
    x = x.flatten(2, -1).sum(dim=-1)
    m_sum = mask.flatten(2, -1).sum(dim=-1)
    x = x / m_sum
    return x

    

class MaskedMSE(StyleGanWrapper):
    def __init__(self, generator, from_, segmenter, resize=None, return_complementary=True):
        super(MaskedMSE, self).__init__(generator)

        self.segmenter = segmenter
        self.segmenter.eval()
        self.gen_size = (self.generator.img_resolution,) * 2
        self.cached_mask = None
        self.from_ = from_
        self.resize = None if resize is None else transforms.Resize(resize)
        self.return_complementary = return_complementary

    def reset(self):
        self.cached_mask = None

    def get_mask(self, x):
        if self.cached_mask is None:
            with torch.no_grad():
                self.cached_mask = self.segmenter(x)

        return self.cached_mask

    def crop_to_bbox(self, x, mask):
        mask = mask.flatten(0, 1).any(dim=0)
        H, W = x.shape[-2:]
        grid0, grid1 = torch.meshgrid(torch.arange(H), torch.arange(W))
        grids = torch.stack([grid0, grid1], dim=0).to(x.device)
        # XXX broadcast in masked_selected does not preserve outer dimension.
        pixels = grids.masked_select(mask).reshape(2, -1)
        ul = pixels.min(dim=1)[0]
        br = pixels.max(dim=1)[0]
        size = br - ul + 1
        x = transforms.functional.crop(x, ul[0], ul[1], size[0], size[1])
        return x

    def forward(self, u):
        x = super(MaskedMSE, self).forward(u, from_=self.from_)

        mask = self.get_mask(x)

        if self.resize is not None:
            x = self.resize(x)
            mask = self.resize(mask)
        x_in = x * mask
        x_in = self.crop_to_bbox(x_in, mask)
        if self.return_complementary:
            inv_mask = ~mask
            x_out = x * (inv_mask)
            x_out = self.crop_to_bbox(x_out, inv_mask)
            return x_in, x_out
        else:
            return x_in

            
class LandmarkMSE(StyleGanWrapper):
    def __init__(self, generator, from_, detector, selected):
        super(LandmarkMSE, self).__init__(generator)

        self.detector = detector
        self.detector.eval()
        self.from_ = from_
        self.selected = selected
        self.not_selected = list(set(range(detector.num_landmarks)).difference(self.selected))

    def reset(self):
        pass

    def forward(self, u):
        x = super(LandmarkMSE, self).forward(u, from_=self.from_)
        landmarks = self.detector(x)

        return landmarks[:, self.selected], landmarks[:, self.not_selected]

class RelativeLandmarkMSE(LandmarkMSE):
    def __init__(self, generator, from_, detector, selected):
        super(RelativeLandmarkMSE, self).__init__(
                generator, from_, detector, selected)

    def reset(self):
        pass

    def forward(self, u):
        landmarks0, landmarks1 = super(RelativeLandmarkMSE, self).forward(u)
        # landmarks = torch.cat([landmarks0, landmarks1], dim=1)
        # centers = landmarks.mean(dim=1, keepdim=True)
        landmarks0 = landmarks0 - landmarks0.mean(dim=1, keepdim=True)
        landmarks1 = landmarks1 - landmarks1.mean(dim=1, keepdim=True)

        return landmarks0, landmarks1

class LandmarkInterpolationMSE(StyleGanWrapper):
    def __init__(self, generator, from_, detector, segmenter, resize=None):
        super(LandmarkInterpolationMSE, self).__init__(generator)
        self.detector = detector
        self.segmenter = segmenter
        detector.eval()
        not segmenter or segmenter.eval()
        self.from_ = from_
        self.resize = None if resize is None else transforms.Resize(resize)

    def reset(self):
        self.vertex_indices = None
        self.interp_weights = None
        pass

    def forward(self, u):
        x = super(LandmarkInterpolationMSE, self).forward(u, from_=self.from_)
        landmarks = self.detector(x)
        mask = None if not self.segmenter else self.segmenter(x)

        if self.resize is not None:
            scale = x.new_tensor(self.resize.size) / x.new_tensor(x.shape[-2:])
            landmarks *= scale
            x = self.resize(x)
            mask = self.resize(mask)

        pixels, vertex_indices, interp_weights = [], [], []
        if self.vertex_indices is None or self.interp_weights is None:
            for i in range(len(x)):
                ret = interpolate_from_landmarks(x[i], landmarks[i], mask=mask[i, 0])
                pixels.append(ret[0])
                vertex_indices.append(ret[1])
                interp_weights.append(ret[2])
            self.vertex_indices = vertex_indices
            self.interp_weights = interp_weights
        else:
            for i in range(len(x)):
                ret = interpolate_from_landmarks(x[i], landmarks[i],
                        vertex_indices=self.vertex_indices[i],
                        weights = self.interp_weights[i])
                pixels.append(ret[0])

        min_len = min(len(_) for _ in pixels)
        assert min_len > 10
        for i in range(len(pixels)):
            perm = torch.randperm(len(pixels[i]))
            pixels[i] = pixels[i][perm[:min_len]]

        pixels = torch.stack(pixels, dim=0).unsqueeze(-2)
        # pixels = pixels.detach()
        pixels = pixels / (u.new_tensor(x.shape[-2:]) - 1) * 2 - 1
        # grid_sample assume grid to be "xy" order
        pixels = pixels.flip(dims=(-1,))
        interpolated = F.grid_sample(x, pixels, align_corners=True)
        return interpolated

        # H, W = x.shape[-2:]
        # grid0, grid1 = torch.meshgrid(torch.arange(H), torch.arange(W))
        # pixels = torch.stack([grid1, grid0], dim=-1).unsqueeze(0).expand(len(u), -1, -1, -1)
        # pixels = pixels.to(u.device).float()

        # pixels = pixels / (u.new_tensor([H, W]) - 1) * 2 - 1
        # interpolated = F.grid_sample(x, pixels, align_corners=True)
        
class RegionColorMSE(StyleGanWrapper):
    def __init__(self, generator, from_, segmenter, resize=None, return_residual=True):
        super(RegionColorMSE, self).__init__(generator)
        self.segmenter = segmenter
        not segmenter or segmenter.eval()
        self.from_ = from_
        self.resize = None if resize is None else transforms.Resize(resize)
        self.return_residual = return_residual

    def forward(self, u):
        x = super(RegionColorMSE, self).forward(u, from_=self.from_)
        mask = None if not self.segmenter else self.segmenter(x)
        if self.resize is not None:
            x = self.resize(x)
            mask = self.resize(mask)

        mean_color = masked_mean(x, mask)[..., None, None]
        mean_color_map = mean_color * mask
        residual_map = x - mean_color_map
        if self.return_residual:
            return mean_color, residual_map
        else:
            return mean_color

        

class HighFrequencyMSE(StyleGanWrapper):
    def __init__(self, generator, from_, segmenter, resize=None):
        super(HighFrequencyMSE, self).__init__(generator)
        self.segmenter = segmenter
        not segmenter or segmenter.eval()
        self.from_ = from_
        self.resize = None if resize is None else transforms.Resize(resize)
        self.low_freq_resize = transforms.Resize((24, 24))

    def forward(self, u):
        x = super(HighFrequencyMSE, self).forward(u, from_=self.from_)
        mask = None if not self.segmenter else self.segmenter(x)
        if self.resize is not None:
            x = self.resize(x)
            if mask is not None:
                mask = self.resize(mask)

        x_small = self.low_freq_resize(x)
        x_low = transforms.functional.resize(x_small, x.shape[-2:])
        x_low = x_low
        x_high = x - x_low
        if mask is not None:
            x_low = x_low * mask
            x_high = x_high * mask

        # plt.figure()
        # plt.imshow((x_high)[0].permute(1, 2, 0).cpu().detach())
        # plt.show()
        # plt.figure()
        # plt.imshow(x_small[0].permute(1, 2, 0).cpu().detach())
        # plt.show()
        # plt.figure()
        # plt.imshow(x_low[0].permute(1, 2, 0).cpu().detach())
        # plt.show()
        # import pdb; pdb.set_trace()

        return x_high, x_low

    def forward_interesting(self, u):
        x = super(HighFrequencyMSE, self).forward(u, from_=self.from_)
        mask = None if not self.segmenter else self.segmenter(x)
        if self.resize is not None:
            x = self.resize(x)
            mask = self.resize(mask)

        x_small = self.low_freq_resize(x)
        x_low = transforms.functional.resize(x_small, x.shape[-2:])

        # plt.figure()
        # plt.imshow((x - x_low)[0].permute(1, 2, 0).cpu().detach())
        # plt.show()
        # plt.figure()
        # plt.imshow(x_small[0].permute(1, 2, 0).cpu().detach())
        # plt.show()
        # plt.figure()
        # plt.imshow(x_low[0].permute(1, 2, 0).cpu().detach())
        # plt.show()
        # import pdb; pdb.set_trace()

        return x - x_low, x_small

class IDFeatureMSE(StyleGanWrapper):
    def __init__(self, generator, from_, extractor):
        super(IDFeatureMSE, self).__init__(generator)
        self.extractor = extractor 
        self.extractor.eval()
        self.from_ = from_

    def forward(self, u):
        x = super(IDFeatureMSE, self).forward(u, from_=self.from_)
        feature = self.extractor(x)
        return feature

        
