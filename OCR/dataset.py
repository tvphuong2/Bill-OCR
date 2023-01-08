import os
import re
import sys
import six
import math
import torch

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms

def Batch_Dataloader(args, data_dir, character):
  align = Align(imgH = args.imgH, imgW = args.imgW)
  dataset = makeDataset(root=data_dir, args= args, character= character)

  return torch.utils.data.DataLoader(
              dataset, batch_size=args.batch_size,
              shuffle=True,
              num_workers=int(args.workers),
              collate_fn= align, pin_memory=True)


class makeDataset(Dataset):

  def __init__(self, root, args, character):
    self.root = root
    self.args = args
    self.character = character
    self.images = []

    datalist = os.listdir(root)
    print(f'Training dir: {root}')
    if len(datalist) > 180000:
      datalist = datalist[:180000]
    for d in datalist:
      with open(f'{root}/{d}', 'rb') as f:
        image = f.read()
        buf = six.BytesIO()
        buf.write(image)
        buf.seek(0)
        image = Image.open(buf).convert('L')
        label = d.replace(".jpg","")
        self.images.append((image, label))

    self.nSamples = len(self.images)


  def __len__(self):
      return self.nSamples

  def __getitem__(self, index):
      (image, label) = self.images[index]

      label = label.replace('*', "/").replace('+', "-")
      out_of_char = f'[^{self.character}]'
      label = re.sub(out_of_char, '-', label)

      return (image, label)


class Normalize(object):
  def __init__(self, max_size):
      self.toTensor = transforms.ToTensor()
      self.max_size = max_size
      self.max_width_half = math.floor(max_size[2] / 2)

  def __call__(self, img):
      img = self.toTensor(img)
      img.sub_(0.5).div_(0.5)
      c, h, w = img.size()
      Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
      Pad_img[:, :, :w] = img
      if self.max_size[2] != w:
          Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

      return Pad_img

class Align(object):
  def __init__(self, imgH=64, imgW=300):
    self.imgH = imgH
    self.imgW = imgW

  def __call__(self, batch):
      batch = filter(lambda x: x is not None, batch)
      images, labels = zip(*batch)

      resized_max_w = self.imgW

      transform = Normalize((1, self.imgH, resized_max_w))

      resized_images = []
      for image in images:
          w, h = image.size
          ratio = w / float(h)
          if math.ceil(self.imgH * ratio) > self.imgW:
              resized_w = self.imgW
          else:
              resized_w = math.ceil(self.imgH * ratio)

          resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
          resized_images.append(transform(resized_image))
          
          # resized_image.save(f'/content/output/out0/{d}.jpg')

      image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

      return image_tensors, labels