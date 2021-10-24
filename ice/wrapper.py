import matplotlib.pyplot as plt
import face_alignment
import kornia
import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch_utils import misc

import dnnlib
import legacy

from external.identity.iresnet import iresnet50, iresnet100
from external.landmark.mobilefacenet import MobileFaceNet
from external.parsing.model import BiSeNet

class StyleGanWrapper(nn.Module):
    def __init__(self, generator):
        super(StyleGanWrapper, self).__init__()
        if isinstance(generator, nn.Module):
            self.generator = generator
        elif isinstance(generator, str):
            with open(generator, 'rb') as f:
                official_gan = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
            self.generator = Generator(z_dim=official_gan.z_dim,
                    c_dim=official_gan.c_dim,
                    w_dim=official_gan.w_dim,
                    img_resolution=official_gan.img_resolution,
                    img_channels=official_gan.img_channels)
    
            self.generator.load_state_dict(official_gan.state_dict())
        else:
            raise NotImplementedError

        self.generator.eval()
        self.truncation_psi = 1
        self.noise_mode = 'const'
        gen_size = (self.generator.img_resolution, ) * 2

    def reset(self):
        pass

    def forward(self, u, from_='z', to='x'):
        self.generator.eval()

        if from_ == 'z':
            misc.assert_shape(u, [None, self.generator.z_dim])
            zs = u
        if to == 'z':
            return zs

        if from_ == 'w':
            misc.assert_shape(u, [None, self.generator.w_dim])
            w0s = u
            ws = w0s.unsqueeze(1).expand(-1, self.generator.num_ws, -1)
        elif from_ == 'w+':
            misc.assert_shape(u, [None, self.generator.num_ws, self.generator.w_dim])
            ws = u
        elif from_ in ['z']:
            ws = self.generator.mapping(zs, None, truncation_psi=self.truncation_psi)
            w0s = ws[:, 0, :]
        if to == 'w':
            return w0s
        if to == 'w+':
            return ws

        if from_ == 's':
            svec = u
            styles = self.generator.unpack_styles(svec)
        elif from_ in ['z', 'w', 'w+']:
            styles = self.generator.compute_styles(ws)
            svec = self.generator.pack_styles(styles)
        if to == 's':
            return svec

        x = self.generator.synthesis(styles, noise_mode=self.noise_mode)
        x = x * 0.5 + 0.5
        return x


class FaceSegmenter(nn.Module):
    def __init__(self, path, mask_labels, morphology=None, ks_ratio=0.1):
        super(FaceSegmenter, self).__init__()

        n_classes = 19
        self.segmenter = BiSeNet(n_classes=n_classes)
        with dnnlib.util.open_url(path, 'rb') as f:
            self.segmenter.load_state_dict(torch.load(f))
        self.segmenter.eval()
        self.sgt_size = (512, 512)
        self.normalize = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize(self.sgt_size)
        ])

        self.morphology = morphology
        ks_size = int(self.sgt_size[0] * ks_ratio)
        self.register_buffer('kernel',
                torch.ones(ks_size, ks_size, dtype=torch.int))
        self.mask_labels = mask_labels

    def forward(self, x):
        x_n = self.normalize(x)
        pred = self.segmenter(x_n)[0].argmax(dim=1, keepdim=True)
        target = x.new_tensor(self.mask_labels)[None, :, None, None]
        mask = (pred == target).any(dim=1, keepdim=True).int()

        mask = mask.cpu()
        self.kernel = self.kernel.cpu()
        if self.morphology == 'dilation':
            mask = kornia.morphology.dilation(mask, self.kernel, border_type='constant')
        elif self.morphology == 'erosion':
            mask = kornia.morphology.erosion(mask, self.kernel, border_type='constant')
        elif self.morphology == 'ring':
            dilated  = kornia.morphology.dilation(mask, self.kernel, border_type='constant')
            eroded = kornia.morphology.erosion(mask, self.kernel, border_type='constant')
            mask = dilated & (~eroded)
        else:
            assert self.morphology is None
        mask = mask.to(x.device)

        mask = transforms.functional.resize(mask, x.shape[-2:])
        mask = mask > 0
        return mask

class KeyPointDetector2(nn.Module):
    def __init__(self, path):
        super(KeyPointDetector2, self).__init__()
        self.det_size = (112, 112)
        self.num_landmarks = 68
        self.detector = MobileFaceNet(self.det_size, self.num_landmarks * 2)
        self.detector.load_state_dict(torch.load(path)['state_dict'])
        self.detector.eval()
        self.normalize = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize(self.det_size)
        ])

    def forward(self, x):
        x = self.normalize(x)
        B, C, H, W = x.shape
        landmarks = self.detector(x)[0]
        landmarks = landmarks.reshape(-1, self.num_landmarks, 2)
        landmarks = landmarks.flip(dims=(-1,)) * x.new_tensor([H, W])

        # landmarks = landmarks.cpu().detach()
        # for i in range(len(x)):
        #     plt.figure()
        #     plt.imshow(x[i].permute(1, 2, 0).cpu().detach())
        #     plt.scatter(landmarks[i, :, 1], landmarks[i, :, 0])
        #     plt.show()
        # import pdb; pdb.set_trace()
        
        return landmarks

class KeyPointHeatMapper(nn.Module):
    def __init__(self):
        super(KeyPointHeatMapper, self).__init__()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        self.detector = self.fa.face_alignment_net
        self.detector.eval()
        self.det_size = (256, 256)
        self.num_landmarks = 68
        self.resize = transforms.Compose([ transforms.Resize(self.det_size) ])

    def forward(self, x):
        H0, W0 = x.shape[-2:]
        x = self.resize(x)
        out = self.detector(x)

        return out 

class KeyPointDetector(nn.Module):
    def __init__(self):
        super(KeyPointDetector, self).__init__()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        self.detector = self.fa.face_alignment_net
        self.detector.eval()
        self.det_size = (256, 256)
        self.num_landmarks = 68
        self.resize = transforms.Compose([ transforms.Resize(self.det_size) ])

    def forward(self, x):
        H0, W0 = x.shape[-2:]
        x = self.resize(x)
        # out = self.detect(x)

        # landmarks = self.fa.get_landmarks(x[0].permute(1, 2, 0))
        # plt.imshow(x[0].permute(1, 2, 0).cpu().detach())
        # plt.show()
        # import pdb; pdb.set_trace()

        preds = self.detector(x)
        preds /= preds.sum([2, 3], keepdim=True)
        B, C, H, W = preds.shape
        grid0, grid1 = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid0 = grid0[None, None, ...].to(x.device)
        grid1 = grid1[None, None, ...].to(x.device)
        p0 = (grid0 * preds).sum(dim=[2, 3])
        p1 = (grid1 * preds).sum(dim=[2, 3])
        p0 = (p0 + 0.) / H * H0
        p1 = (p1 + 0.) / W * W0
        landmarks = torch.stack([p0, p1], dim=-1)
        
        return landmarks

    def detect0(self, x):
        fa = self.fa
        x = self.resize(x)
        resolution = 256
        for i in range(len(x)):
            x_perm = x[i].permute(1, 2, 0) * 255
            with torch.no_grad():
                faces = fa.face_detector.detect_from_image(x_perm)
            assert len(faces) > 0
            d = faces[0]
            center = torch.tensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / fa.face_detector.reference_scale
            ul = face_alignment.utils.transform([1, 1], center, scale, resolution, True)
            br = face_alignment.utils.transform([resolution, resolution], center, scale, resolution, True)
            print('my', center, scale, resolution, ul, br)
            size = br - ul
            cropped = resized_crop(x[i],
                    ul[0], ul[1], size[0], size[1], [resolution, resolution])
            out = fa.face_alignment_net(cropped.unsqueeze(0)).detach()

            inp = face_alignment.utils.crop(x_perm.detach().cpu().numpy(), center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()

            inp = inp.to(x.device)
            inp.div_(255.0).unsqueeze_(0)

            out1 = fa.face_alignment_net(inp).detach()
            out2 = fa.face_alignment_net(x[i].unsqueeze(0)).detach()

            preds = out
            preds /= preds.sum([2, 3], keepdim=True)
            B, C, H, W = preds.shape
            grid0, grid1 = torch.meshgrid(torch.arange(H), torch.arange(W))
            grid0 = grid0[None, None, ...].to(device)
            grid1 = grid1[None, None, ...].to(device)
            p0 = (grid0 * preds).sum(dim=[2, 3]).cpu().detach()
            p1 = (grid1 * preds).sum(dim=[2, 3]).cpu().detach()
            plt.imshow(preds[0, 0].cpu().detach())
            plt.scatter(x=p1, y=p0, c='r')
            plt.show()

            preds = out1
            preds /= preds.sum([2, 3], keepdim=True)
            B, C, H, W = preds.shape
            grid0, grid1 = torch.meshgrid(torch.arange(H), torch.arange(W))
            grid0 = grid0[None, None, ...].to(device)
            grid1 = grid1[None, None, ...].to(device)
            p0 = (grid0 * preds).sum(dim=[2, 3]).cpu().detach()
            p1 = (grid1 * preds).sum(dim=[2, 3]).cpu().detach()
            plt.imshow(preds[0, 0].cpu().detach())
            plt.scatter(x=p1, y=p0, c='r')
            plt.show()

            preds = out2
            preds /= preds.sum([2, 3], keepdim=True)
            B, C, H, W = preds.shape
            grid0, grid1 = torch.meshgrid(torch.arange(H), torch.arange(W))
            grid0 = grid0[None, None, ...].to(device)
            grid1 = grid1[None, None, ...].to(device)
            p0 = (grid0 * preds).sum(dim=[2, 3]).cpu().detach()
            p1 = (grid1 * preds).sum(dim=[2, 3]).cpu().detach()
            plt.imshow(preds[0, 0].cpu().detach())
            plt.scatter(x=p1, y=p0, c='r')
            plt.show()
            import pdb; pdb.set_trace()


class IDFeatureExtractor(nn.Module):
    def __init__(self, model_pth="backbone100.pth"):
        super(IDFeatureExtractor, self).__init__()
        self.model = iresnet100()
        with dnnlib.util.open_url(path, 'rb') as f:
            self.model.load_state_dict(torch.load(f))

    def forward(self, x):
        x = (x - 0.5) / 0.5
        z = self.model(F.interpolate(x, size=112, mode='bilinear'))
        z = F.normalize(z)
        return z
