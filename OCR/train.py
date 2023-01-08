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

def train(args):
  train_dataloader = Batch_Dataloader(args, args.train_dir, CHAR)
  valid_dataloader = Batch_Dataloader(args, args.valid_dir, CHAR)

  converter = Converter(CHAR)
  args.num_class = len(CHAR) + 1

  model = Model(args)

  for name, param in model.named_parameters():
    try:
        if 'bias' in name:
            init.constant_(param, 0.0)
        elif 'weight' in name:
            init.kaiming_normal_(param)
    except Exception as e:  # for batchnorm.
        if 'weight' in name:
            param.data.fill_(1)
        continue

  model = torch.nn.DataParallel(model).to(device)
  model.train()
  print("Model:")
  print(model)

  criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
  # criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

  filtered_parameters = []
  for p in filter(lambda p: p.requires_grad, model.parameters()):
    filtered_parameters.append(p)

  # optimizer = optim.Adam(filtered_parameters, lr=args.lr)
  optimizer = optim.Adadelta(filtered_parameters, lr=args.lr, rho=0.95, eps=1e-8)

  iteration = 0
  if os.path.exists(args.saved_model):
    print("Load model")
    checkpoint = torch.load(args.saved_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iter']
  else:
    print("New model")
  
  loss_avg = Averager()
  start_time = time.time()

  while(True):
    for image_tensors, labels in train_dataloader:
      image = image_tensors.to(device)
      batch_size = image.size(0)

      text, length = converter.encode(labels, args.text_max_length)

      preds = model(image) # 192x45x206
      
      preds_size = torch.IntTensor([preds.size(1)] * batch_size)
      
      preds = preds.log_softmax(2).permute(1, 0, 2) # 45x192x206
      # print(preds[0])
      cost = criterion(preds, text, preds_size, length)

      model.zero_grad()
      cost.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      loss_avg.add(cost)

      if (iteration) % args.valInterval == 0:
        elapsed_time = time.time() - start_time

        model.eval()
        with torch.no_grad():  
          preds_val, labels_val, acc, loss = validation(model, criterion, valid_dataloader, converter, args)
        model.train()

        print(f'TRAIN [{iteration}]\t\t loss: {loss_avg.val()} \t time: {elapsed_time}')
        print(f'VALID \t\t\t loss: {loss} \t acc: {acc}')
        print('------------------------------------------')
        for i in range(5):
          print(f'|{preds_val[i]}|\t\t{labels_val[i]}')
        print('-----------------------------------------------------------------')

        loss_avg.reset()
      iteration += 1
      if (iteration) % 500 == 0:
        torch.save({
              'iter': iteration,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, args.saved_model)
        print("Model saved")

      if iteration == args.num_iter:
        print('end the training')
        sys.exit()

  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_dir',        type=str)
  parser.add_argument('--valid_dir',        type=str)
  parser.add_argument('--saved_model',      type=str)
  parser.add_argument('--random_seed',      type=int,   default=1111)
  parser.add_argument('--workers',          type=int,   default=2)
  parser.add_argument('--batch_size',       type=int,   default=192)
  parser.add_argument('--text_max_length',  type=int,   default=45)
  parser.add_argument('--num_iter',         type=int,   default=300000)
  parser.add_argument('--imgH',             type=int,   default=64)
  parser.add_argument('--imgW',             type=int,   default=736)
  parser.add_argument('--input_channel',    type=int,   default=1)
  parser.add_argument('--output_channel',   type=int,   default=512)
  parser.add_argument('--hidden_size',      type=int,   default=256)
  parser.add_argument('--lr',               type=float, default=0.001)
  parser.add_argument('--grad_clip',        type=float, default=1)
  parser.add_argument('--valInterval',      type=int,   default=500)

  args = parser.parse_args()

  random.seed(args.random_seed)
  np.random.seed(args.random_seed)
  torch.manual_seed(args.random_seed)
  torch.cuda.manual_seed(args.random_seed)

  cudnn.benchmark = True
  cudnn.deterministic = True

  train(args)



