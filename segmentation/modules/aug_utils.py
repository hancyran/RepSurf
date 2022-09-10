"""
Author: Haoxi Ran
Date: 06/30/2022
"""

import numpy as np


def transform_point_cloud_coord(args):
    transform_list = []
    aug_args = args.aug_args
    if args.aug_scale:
        transform_list.append(
            RandomScale(aug_args['scale_factor'], aug_args['scale_ani'], aug_args['scale_prob']))
    if args.aug_rotate:
        if args.aug_rotate == 'pert':
            transform_list.append(
                RandomRotatePerturb(aug_args['pert_factor'], 3 * aug_args['pert_factor'], aug_args['pert_prob']))
        elif args.aug_rotate == 'pert_z':
            transform_list.append(
                RandomRotatePerturbAligned(aug_args['pert_factor'], 3 * aug_args['pert_factor'], aug_args['pert_prob']))
        elif args.aug_rotate == 'rot':
            transform_list.append(
                RandomRotate(prob=aug_args['rot_prob']))
        elif args.aug_rotate == 'rot_z':
            transform_list.append(
                RandomRotateAligned(prob=aug_args['rot_prob']))
    if args.aug_jitter:
        transform_list.append(
            RandomJitter(aug_args['jitter_factor'], 5 * aug_args['jitter_factor'], aug_args['jitter_prob'], args.lidar))
    if args.aug_flip:
        transform_list.append(RandomFlip())
    if args.aug_shift:
        transform_list.append(RandomShift(aug_args['shifts'], aug_args['shift_prob']))
    return Compose(transform_list) if len(transform_list) > 0 else None


def transform_point_cloud_rgb(args):
    transform_list = []
    aug_args = args.aug_args
    if args.color_contrast:
        transform_list.append(ChromaticAutoContrast())
    if args.color_shift:
        transform_list.append(ChromaticTranslation())
    if args.color_jitter:
        transform_list.append(ChromaticJitter())
    if args.hs_shift:
        transform_list.append(HueSaturationTranslation())
    if args.color_drop:
        transform_list.append(RandomDropColor())
    return Compose(transform_list) if len(transform_list) > 0 else None


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coord, feat, label, mask=None):
        for t in self.transforms:
            coord, feat, label = t(coord, feat, label, mask)
        return coord, feat, label


class RandomRotate(object):
    def __init__(self, rot=(np.pi/24, np.pi/24, np.pi/4), prob=1.):
        self.rot = rot
        self.prob = prob

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.prob:
            angle_x = np.random.uniform(-self.rot[0], self.rot[0])
            angle_y = np.random.uniform(-self.rot[1], self.rot[1])
            angle_z = np.random.uniform(-self.rot[2], self.rot[2])
            cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
            cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
            R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
            R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            R = np.dot(R_z, np.dot(R_y, R_x))
            coord = np.dot(coord, np.transpose(R))
        return coord, feat, label


class RandomRotateAligned(object):
    def __init__(self, rot=np.pi, prob=1.):
        self.rot = rot
        self.prob = prob

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.prob:
            angle_z = np.random.uniform(-self.rot, self.rot)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            coord = np.dot(coord, R)
        return coord, feat, label


class RandomRotatePerturb(object):
    def __init__(self, sigma=0.03, clip=0.09, prob=1.):
        self.sigma = sigma
        self.clip = clip
        self.prob = prob

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.prob:
            angle_x = np.clip(np.random.normal() * self.sigma, -self.clip, self.clip)
            angle_y = np.clip(np.random.normal() * self.sigma, -self.clip, self.clip)
            angle_z = np.clip(np.random.normal() * self.sigma, -self.clip, self.clip)
            cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
            cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
            R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
            R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            R = np.dot(R_z, np.dot(R_y, R_x))
            coord = np.dot(coord, np.transpose(R))
        return coord, feat, label


class RandomRotatePerturbAligned(object):
    def __init__(self, sigma=0.03, clip=0.09, prob=1.):
        self.sigma = sigma
        self.clip = clip
        self.prob = prob

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.prob:
            angle_z = np.clip(np.random.normal() * self.sigma, -self.clip, self.clip)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            coord = np.dot(coord, R)
        return coord, feat, label


class RandomScale(object):
    def __init__(self, scale=0.1, anisotropic=False, prob=1.):
        self.scale = scale
        self.anisotropic = anisotropic
        self.prob = prob

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.prob:
            scale = np.random.uniform(1 - self.scale, 1 + self.scale, 3 if self.anisotropic else 1)
            coord *= scale
        return coord, feat, label


class RandomShift(object):
    def __init__(self, shift=(0.2, 0.2, 0), p=0.95):
        self.shift = shift
        self.p = p

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.p:
            shift_x = np.random.uniform(-self.shift[0], self.shift[0])
            shift_y = np.random.uniform(-self.shift[1], self.shift[1])
            shift_z = np.random.uniform(-self.shift[2], self.shift[2])
            coord += [shift_x, shift_y, shift_z]
        return coord, feat, label


class RandomFlip(object):
    def __init__(self, p=1.):
        self.p = p

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.p:
            if np.random.rand() < 0.5:
                coord[:, 0] = -coord[:, 0]
            if np.random.rand() < 0.5:
                coord[:, 1] = -coord[:, 1]
        return coord, feat, label


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05, p=1., is_lidar=False):
        self.sigma = sigma
        self.clip = clip
        self.p = p
        self.is_lidar = is_lidar

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.p:
            assert (self.clip > 0)
            jitter = np.clip(self.sigma * np.random.randn(coord.shape[0], 3), -1 * self.clip, self.clip)
            if self.is_lidar:
                jitter[:, 2] *= 0.1  # re-scale z-axis jitter
            coord += jitter
        return coord, feat, label


class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.p:
            tmp_feat = feat[mask] if mask is not None else feat
            lo = np.min(tmp_feat, 0, keepdims=True)
            hi = np.max(tmp_feat, 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (tmp_feat[:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            tmp_feat[:, :3] = (1 - blend_factor) * tmp_feat[:, :3] + blend_factor * contrast_feat
            if mask is not None:
                feat[mask] = tmp_feat
            else:
                feat = tmp_feat
        return coord, feat, label


class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, feat.shape[1]) - 0.5) * 255 * 2 * self.ratio
            feat[:, :3] = np.clip(tr + feat[:, :3], 0, 255)
            if mask is not None:
                feat[:, :3][~mask] = 0.
        return coord, feat, label


class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.p:
            noise = np.random.randn(*feat.shape)
            noise *= self.std * 255
            feat[:, :3] = np.clip(noise + feat[:, :3], 0, 255)
            if mask is not None:
                feat[:, :3][~mask] = 0.
        return coord, feat, label


class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2, p=1.):
        self.hue_max = hue_max
        self.saturation_max = saturation_max
        self.p = p

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.p:
            # Assume feat[:, :3] is rgb
            tmp_feat = feat[mask] if mask is not None else feat
            hsv = HueSaturationTranslation.rgb_to_hsv(tmp_feat[:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            tmp_feat[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
            if mask is not None:
                feat[mask] = tmp_feat
            else:
                feat = tmp_feat
        return coord, feat, label


class RandomDropColor(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, coord, feat, label, mask=None):
        if np.random.rand() < self.p:
            feat[:, :3] = 0
        return coord, feat, label
