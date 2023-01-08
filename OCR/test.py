import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from dataset import Batch_Dataloader
from text2id import Converter
from model import Model
from validation import validation
import torchvision.transforms as transforms

from PIL import Image, ImageDraw

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CHAR = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzĐđĂăÂâÊêÔôƠơƯưÀàẰằẦầÈèỀềÌìÒòỒồỜờÙùỪừỲỳẢảẲẳẨẩẺẻỂểỈỉỎỏỔổỞởỦủỬửỶỷÃãẴẵẪẫẼẽỄễĨĩÕõỖỗỠỡŨũỮữỸỹÁáẮắẤấÉéẾếÍíÓóỐốỚớÚúỨứÝýẠạẶặẬậẸẹỆệỊịỌọỘộỢợỤụỰựỴỵ /()-,.0123456789%:"

class Averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def predict(args):
  predict_dataloader = Batch_Dataloader(args, args.predict_dir, CHAR)

  converter = Converter(CHAR)
  args.num_class = len(CHAR) + 1

  model = Model(args)
  model = torch.nn.DataParallel(model).to(device)

  criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)

  filtered_parameters = []
  for p in filter(lambda p: p.requires_grad, model.parameters()):
    filtered_parameters.append(p)

  checkpoint = torch.load(args.saved_model)
  model.load_state_dict(checkpoint['model_state_dict'])
  
  loss_avg = Averager()
  start_time = time.time()

  model.eval()
  with torch.no_grad():  
    preds_val, labels_val, acc, loss = validation(model, criterion, predict_dataloader, converter, args)
  
  elapsed_time = time.time() - start_time

  print(f'PREDICT \t\t\t loss: {loss} \t acc: {acc} \t time: {elapsed_time}')
  print('------------------------------------------')
  for i in range(len(preds_val)):
    print(f'|{preds_val[i]}|\t\t{labels_val[i]}')
  print('-----------------------------------------------------------------')

  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--predict_dir')
  parser.add_argument('--random_seed', type=int, default=1111)
  parser.add_argument('--workers', type=int, default=2)
  parser.add_argument('--batch_size', type=int, default=192)
  parser.add_argument('--text_max_length', type=int, default=45)
  parser.add_argument('--saved_model')
  parser.add_argument('--imgH', type=int, default=64)
  parser.add_argument('--imgW', type=int, default=736)
  parser.add_argument('--input_channel', type=int, default=1)
  parser.add_argument('--output_channel', type=int, default=512)
  parser.add_argument('--hidden_size', type=int, default=256)

  args = parser.parse_args()

  random.seed(args.random_seed)
  np.random.seed(args.random_seed)
  torch.manual_seed(args.random_seed)
  torch.cuda.manual_seed(args.random_seed)

  cudnn.benchmark = True
  cudnn.deterministic = True

  predict(args)