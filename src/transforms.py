import random
from torchvision.transforms import RandomRotation, Resize, InterpolationMode, ColorJitter, ToTensor
from torchvision.transforms.functional import resize

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