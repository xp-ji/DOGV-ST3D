import torchvision
torchvision.set_image_backend('accimage')
import random
random.seed(2)
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch

class NumpyToGroupPIL(object):
    def __init__(self, channel=3):
        self.channel = channel

    def __call__(self, img_array):
        if self.channel == 3:
            img_array = img_array.astype(np.uint8)
            img_group = [Image.fromarray(img_array[i], 'RGB') for i in range(img_array.shape[0])]
        else:
            img_array = img_array.astype(np.uint8)
            img_group = [Image.fromarray(img_array[i, 0], mode='L') for i in range(img_array.shape[0])] #I;16
        return img_group


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    # torchvision.transforms.RandomHorizontalFlip(p=0.5)
    '''Randomly horizontally flips the given PIL.Image with a probability of 0.5
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_group, is_flow=False):
        if random.random() < self.p:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        # rep_std = self.std * (tensor.size()[0]//len(self.std))
        #
        # # TODO: make efficient
        if isinstance(tensor, list):
            tensor_list = [t.sub(self.mean).div(self.std) for t in tensor]
            print(tensor_list[0].shape)
            print(len(tensor_list))
            raise RuntimeError
            return tensor_list

        else:
            if isinstance(self.mean, list):
                for t, m, s in zip(tensor, self.mean, self.std):
                    t.sub_(m).div_(s)
            else:
                tensor = tensor.sub(self.mean).div(self.std)
            return tensor


class GroupScale(object):
    ''' Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    '''

    def __init__(self, size, interpolation=Image.NEAREST): # BILINEAR
        self.size = size
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        image_w, image_h = img_group[0].size
        if image_w == self.size[0]:
            return img_group
        else:
            return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.NEAREST # BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]

        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if self.max_distort is not None:
                    if abs(i - j) <= self.max_distort:
                        pairs.append((w, h))
                else:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)

        w_offset = random.randint(0, image_w - crop_pair[0])
        h_offset = random.randint(0, image_h - crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset


class GroupStack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if isinstance(img_group[0], np.ndarray):
            return np.stack([x for x in img_group], axis=0)

        elif img_group[0].mode == 'L': #'I;16':
            return np.stack([np.expand_dims(x, -1) for x in img_group], axis=0)

        elif img_group[0].mode == 'RGB':
            return np.stack([np.array(x) for x in img_group], axis=0)

        else:
            raise ValueError

class ToTorchFormatTensor(object):
    ''' Converts a PIL.Image (RGB) or numpy.ndarray (F x H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] '''
    def __init__(self, div_val=1.0):
        self.div_val = div_val

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 5:
                img = torch.from_numpy(pic.astype(np.float)).permute(0, 4, 1, 2, 3).contiguous()
            elif pic.ndim == 4:
                # img = torch.from_numpy(pic.astype(np.float)).permute(0, 3, 1, 2).contiguous()
                img = torch.from_numpy(pic.astype(np.float)).permute(3, 0, 1, 2).contiguous()

            else:
                img = torch.from_numpy(pic.astype(np.float))

            return img.float().div(self.div_val)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()

            return img.float().div(self.div_val)


class IdentityTransform(object):

    def __call__(self, data):
        return data

class GroupGradient(object):
    def __init__(self, is_dogv=False):
        self.is_dogv = is_dogv

    def __call__(self, img_list):

        seqs_len = len(img_list) // 3
        input_list = []
        for si in range(seqs_len):
            img_b = np.asarray(img_list[si * 3]).astype(np.float)
            img_c = np.asarray(img_list[si * 3 + 1]).astype(np.float)
            img_n = np.asarray(img_list[si * 3 + 2]).astype(np.float)
            if self.is_dogv:
                [dy, dx] = np.gradient(img_c)
                dt = (img_n - img_b) / 2.0
                d_norm = np.sqrt(np.square(dx) + np.square(dy) + np.square(dt) + 1e-8)
                d_val = np.stack([dt/d_norm, dy/d_norm, dx/d_norm], axis=-1)
                input_list.append(d_val)
            else:
                img_cat = np.stack([img_b, img_c, img_n], axis=-1)
                input_list.append(img_cat)
        return input_list

class NumpyRandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img_array):
        if random.random() < self.p:
            img_array = np.flip(img_array, axis=3)
        return img_array


class ComposePILList(object):
    def __init__(self, to_compose=True):
        self.to_compose = to_compose

    def __call__(self, PIL_img_list):
        if self.to_compose:
            PIL_img_list_composed = []
            for img_list in PIL_img_list:
                for img in img_list:
                    PIL_img_list_composed.append(img)
            return PIL_img_list_composed
        else:
            raise RuntimeError

def dynamic_temporal_sparse_sampling(length_ori, temporal_param):
    length_ori -= 2
    length_scales = math.ceil(random.choice(temporal_param['scales']) * length_ori)
    frms_list = list(range(0, length_ori))
    segments = temporal_param['segments']
    sampling_type = temporal_param['sampling_type']

    frms_dynamic_list = random.sample(frms_list, length_scales)

    if length_scales < segments:
        if length_ori < segments:
            tile_time = segments // length_ori + 1
            frms_list = np.tile(frms_list, tile_time)
            frms_list = frms_list.tolist()
        frms_dynamic_list = frms_list

    frms_dynamic_list.sort()
    segments_list = partition(frms_dynamic_list, segments)

    frame_idx = []
    if sampling_type == 'fixed_middle':
        for si in segments_list:
            sidx = len(si) // 2
            frame_idx.append(si[sidx])
            frame_idx.append(si[sidx] + 1)
            frame_idx.append(si[sidx] + 2)
    else:
        frame_idx = []
        for si in segments_list:
            pre_idx = random.choice(si)
            frame_idx.append(pre_idx)
            frame_idx.append(pre_idx + 1)
            frame_idx.append(pre_idx + 2)

    return frame_idx

def partition(ls, n):
    split_array = np.array_split(ls, n)
    split_list = [list(array) for array in split_array]
    return split_list

def group_data_transforms(short_param, intype):
    spatial_transform = None
    spilt_method = NumpyToGroupPIL(1)

    if 'group_scale' in short_param.keys():
        scale_method = GroupScale(short_param['group_scale']['size'])
    else:
        scale_method = None

    if 'group_multi_scale_crop' in short_param.keys():
        crop_method = GroupMultiScaleCrop(short_param['group_multi_scale_crop']['size'],
                                          short_param['group_multi_scale_crop']['scales'])

    elif 'group_center_crop' in short_param.keys():
        crop_method = GroupCenterCrop(short_param['group_center_crop']['size'])

    else:
        crop_method = None

    if 'random_horizontal_flip' in short_param.keys():
        flip_method = GroupRandomHorizontalFlip(short_param['random_horizontal_flip'])
    else:
        flip_method = None
    
    is_dogv = True if intype == 'DOGV' else False

    feature_method = GroupGradient(is_dogv)
    stack_method = GroupStack()

    # Do not do normalization for DOGV
    if is_dogv:
        norm_value = 1.0
        mean = None
        std = None
    else:
        norm_value = short_param['normalization']
        mean = short_param['standardization']['mean']
        std = short_param['standardization']['std']

    if mean is not None and std is not None:
        standard_method = GroupNormalize(mean, std)

    else:
        standard_method = None

    totensor_method = ToTorchFormatTensor(norm_value)

    method_set = [spilt_method, scale_method, crop_method, flip_method, feature_method, stack_method, totensor_method, standard_method]
    method_set = [mi for mi in method_set if mi is not None]

    spatial_transform = torchvision.transforms.Compose(method_set)

    return spatial_transform

