import random
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, ColorJitter, ToTensor
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import pad
from torchvision import transforms
import numpy as np
import numbers

class RandomOverlayD:
    def __init__(self, key_foreground, key_background):
        self.key_foreground = key_foreground
        self.key_background = key_background

    def __call__(self, data):
        foreground_image, background_image = data[self.key_foreground], data[self.key_background]
        max_offset_x, max_offset_y = background_image.size[0] - foreground_image.size[0], background_image.size[1] - foreground_image.size[1]
        background_image.paste(foreground_image, (int(max_offset_x * random.random()), int(max_offset_y * random.random())), foreground_image)
        return background_image

class ResizeD:
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None, keys=[]):
        self.transform = Resize(size, interpolation, max_size, antialias)
        self.keys = keys
    
    def __call__(self, data):
        for key in self.keys:
            data[key] = self.transform(data[key])
        return data

class RandomResizeD:
    def __init__(self, size_range_longer_edge, interpolation=InterpolationMode.BILINEAR, antialias=None, keys=[]):
        self.size_range_longer_edge = size_range_longer_edge
        self.interpolation = interpolation
        self.antialias = antialias
        self.keys = keys
    
    def __call__(self, data):
        for key in self.keys:
            max_size = int(self.size_range_longer_edge[0] + random.random() * (self.size_range_longer_edge[1] - self.size_range_longer_edge[0]))
            ratio = max_size / max(data[key].size)
            size = (int(data[key].size[1] * ratio), int(data[key].size[0] * ratio))
            data[key] = resize(data[key], size, interpolation=self.interpolation, antialias=self.antialias)
        return data

class RandomRotateD:
    def __init__(self, degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None, keys=[]):
        self.keys = keys
        self.transform = RandomRotation(degrees, interpolation, expand, center, fill, resample)
    
    def __call__(self, data):
        for key in self.keys:
            data[key] = self.transform(data[key])
        return data

class TransformD:
    def __init__(self, keys, transform):
        self.keys = keys
        self.transform = transform
    def __call__(self, data):
        for key in self.keys:
            data[key] = self.transform(data[key])
        return data

class ColorJitterMaskedD:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, keys=[]):
        self.transform = ColorJitter(brightness, contrast, saturation, hue)
        self.keys = keys
    
    def __call__(self, data):
        for key in self.keys:
            #TODO this is a hacky way to preserve transparency... consider optimizing
            jittered = data[key].copy()
            jittered = self.transform(jittered)
            data[key].paste(jittered, (0, 0), data[key])
        return data

class ToTensorD:
    def __init__(self, keys=[]):
        self.transform = ToTensor()
        self.keys = keys
    
    def __call__(self, data):
        for key in self.keys:
            data[key] = self.transform(data[key])
        return data

# ====================================================
# slightly modified https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/3

def get_padding(image, size):    
    w, h = image.size
    max_w, max_h = size
    h_padding = (max_w - w) / 2
    v_padding = (max_h - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class NewPad(object):
    def __init__(self, size, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return pad(img, get_padding(img, self.size), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)
# ====================================================
