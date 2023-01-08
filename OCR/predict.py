import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np

from dataset import Batch_Dataloader, Align
from text2id import Converter
from model import Model
import torch.nn.functional as F

import cv2

from PIL import Image, ImageDraw
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
import statistics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CHAR = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzĐđĂăÂâÊêÔôƠơƯưÀàẰằẦầÈèỀềÌìÒòỒồỜờÙùỪừỲỳẢảẲẳẨẩẺẻỂểỈỉỎỏỔổỞởỦủỬửỶỷÃãẴẵẪẫẼẽỄễĨĩÕõỖỗỠỡŨũỮữỸỹÁáẮắẤấÉéẾếÍíÓóỐốỚớÚúỨứÝýẠạẶặẬậẸẹỆệỊịỌọỘộỢợỤụỰựỴỵ /()-,.0123456789%:"

def make_box(image_path):
    image = read_image(image_path)

    refine_net = load_refinenet_model(cuda=True)
    craft_net = load_craftnet_model(cuda=True)
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.5,
        link_threshold=0.4,
        low_text=0.4,
        cuda=True,
        long_size=1280
    )
    boxes = prediction_result["boxes"]

    images = []
    for box in boxes:
      rect = cv2.minAreaRect(box)
      box = cv2.boxPoints(rect)
      box = np.int0(box)

      center = rect[0]
      size = rect[1]
      angle = rect[2]
      center, size = tuple(map(int, center)), tuple(map(int, size))
      rows, cols = image.shape[0], image.shape[1]
      M = cv2.getRotationMatrix2D(center, angle, 1)
      img_rot = cv2.warpAffine(image, M, (cols, rows))
      out = cv2.getRectSubPix(img_rot, size, center)
      # out = cv2.copyMakeBorder(out,3,3,3,3,cv2.BORDER_CONSTANT,value=(255,255,255))
      out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
      if len(out) > len(out[0]):
        out = Image.fromarray(out).rotate(-90, expand=1)
      else:
        out = Image.fromarray(out)
      out 
      images.append([out, ""])
    
    # images[10][0].save("/content/demo/demo1/test2.jpg")
    # sys.exit()
    return images, boxes

def crop(img, f, margin):
    (img_w, img_h) = img.size
    images = []
    for pos in f:
      x = pos[0][0] - 5
      y = pos[0][1] - 5
      xx = pos[2][0] + 5
      yy = pos[2][1] + 5
      if x < 0 or y < 0 or xx > img_w or yy > img_h:
        x = pos[0][0]
        y = pos[0][1]
        xx = pos[2][0]
        yy = pos[2][1]
      images.append([img.crop((x, y, xx, yy)), ""])
    return images

def drawlines(img, lines, h):
  (img_w, img_h) = img.size
  img_m = img_w/2
  draw = ImageDraw.Draw(img) 
  for i in range(len(lines)):
    y1 = -lines[i]*img_m + i*h
    y2 = lines[i]*img_m + i*h
    if y1 < 0 or y1 > img_h or y2 < 0 or y2 > img_h:
      continue
    draw.line((0,y1, img_w,y2))
  img.save("/content/test/out2.jpg")

def make_line(img, boxes, height, smooth = 15, m = 5):
    #make lines
    (img_w, img_h) = img.size
    img_m = img_w/2
    lines = [0]*int(img_h/height)
    for b in boxes:
      x12 = b[0][0] - b[1][0]
      y12 = b[0][1] - b[1][1]
      x23 = b[1][0] - img_m
      y3 =  b[1][1] - y12/x12 * x23
      if y3> 0 and y3<img_h and int(y3/height) < len(lines):
        lines[int(y3/height)] = y12/x12
    
    #smooting
    last = lines[0]
    for i in range(len(lines)):
      if lines[i] != 0:
        last = lines[i]
      else:
        lines[i] = last
    lines = np.array(lines)

    for i in range(smooth):
      for j in range(m,len(lines)-m):    
        lines[j] = sum(lines[j-m:j+m])/len(lines[j-m:j+m])
    
    drawlines(img, lines, height)
    return lines

def predict(args):
    images, boxes = make_box(args.predict_dir)
    boxes_h = []
    for b in boxes:
      boxes_h.append(b[3][1] - b[0][1])
    args.line_height = statistics.mean(boxes_h)*0.78
    print(args.line_height)

    # crop image
    img = Image.open(args.predict_dir).convert('L')

    # align image
    align = Align(imgH = args.imgH, imgW = args.imgW)
    image_tensors, _ = align(images)

    # predict
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
    
    start_time = time.time()

    model.eval()
    with torch.no_grad():  
      image = image_tensors.to(device)
      batch_size = image.size(0)

      length_for_pred = torch.IntTensor([args.text_max_length] * batch_size).to(device)

      preds = model(image) # 192x29x204
      preds_size = torch.IntTensor([preds.size(1)] * batch_size)
      

      _, preds_index = preds.max(2)
      preds_str = converter.decode(preds_index.data, preds_size.data)

    # create horizontal lines in the direction of the bill
    lines = make_line(img, boxes, args.line_height)
    (img_w, img_h) = img.size
    img_m = img_w/2

    # create rows from lines
    rows = []
    for l in lines:
      rows.append([])

    for (b_i, b) in enumerate(boxes):
      m_y = (b[2][1] + b[0][1])/2
      m_x = (b[2][0] + b[0][0])/2
      for (i, line) in enumerate(lines):
        if m_x*line + i * args.line_height > m_y:
          rows[i].append([preds_str[b_i],b[0][0]])
          break
    
    for i in range(len(rows)):
      rows[i] = sorted(rows[i], key=lambda x:x[1])
    
    for r in rows:
      print(r)

    # find money rows
    money_row = []
    for i in range(len(rows)):
      if len(rows[i]) > 0:
        text = rows[i][-1][0].replace("D", "0").replace("Ọ", "0").replace("O", "0").replace("o", "0").replace(".", "").replace(",", "").replace(" ", "").replace("đ", "").replace("-", "")

        if text.isnumeric():
          money_row.append(i)

    # for i in range(1, len(money_row)-1):
    #   s = money_row[i] - money_row[i-1]
    #   e = money_row[i + 1] - money_row[i]
    #   if s > 3*e:
    #     money_row[i-1] = 0
    #   if s*3 < e:
    #     money_row[i+1] = 99

    # money_row = [i for i in money_row if i != 0 and i != 99]

    money_col = []
    for m in money_row:
      money_col.append(rows[m][-1][0].replace("D", "0").replace("Ọ", "0").replace("O", "0").replace("o", "0").replace(".", "").replace(",", "").replace(" ", "").replace("đ", "").replace("-", ""))

    # merge rows
    for i in range(len(rows)):
      dis = []
      for j in money_row:
        dis.append([j, abs(j-i)])
      dis = sorted(dis, key=lambda x:x[1])

      if dis[0][1] != 0 and dis[0][1] < abs(dis[0][0] - dis[1][0]):
        if dis[0][0] < i:
          rows[dis[0][0]] = rows[dis[0][0]] + rows[i] 
        else:
          rows[dis[0][0]] = rows[i] + rows[dis[0][0]]

    #show result
    for i in money_row:
      new_row = []
      for j in range(len(rows[i])):
        result = rows[i][j][0].replace("D", "0").replace("Ọ", "0").replace("O", "0").replace("o", "0").replace(".", "").replace(",", "").replace(" ", "").replace("đ", "").replace("-", "")
        result = ''.join([i for i in result if not i.isdigit()])

        if len(result) > 2:
          new_row.append(rows[i][j][0])
      rows[i] = new_row


    for i in range(len(money_row)):
        print(f'{rows[money_row[i]]} : {money_col[i]}')

  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--predict_dir')
  parser.add_argument('--saved_model')
  parser.add_argument('--line_height',      type=int, default=15)
  parser.add_argument('--image_margin',     type=int, default=3)
  parser.add_argument('--random_seed',      type=int, default=1111)
  parser.add_argument('--workers',          type=int, default=2)
  parser.add_argument('--batch_size',       type=int, default=192)
  parser.add_argument('--text_max_length',  type=int, default=45)
  parser.add_argument('--imgH',             type=int, default=64)
  parser.add_argument('--imgW',             type=int, default=736)
  parser.add_argument('--input_channel',    type=int, default=1)
  parser.add_argument('--output_channel',   type=int, default=512)
  parser.add_argument('--hidden_size',      type=int, default=256)

  args = parser.parse_args()

  random.seed(args.random_seed)
  np.random.seed(args.random_seed)
  torch.manual_seed(args.random_seed)
  torch.cuda.manual_seed(args.random_seed)

  cudnn.benchmark = True
  cudnn.deterministic = True

  predict(args)