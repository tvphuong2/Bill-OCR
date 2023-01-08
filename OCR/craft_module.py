from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
from scipy.stats import norm
import numpy as np

image_path = '/content/drive/MyDrive/AI Projects/OCR Project/Image/Test/Raw/7194137_438cb0adf0860f5c3b6331e13810eeab.jpg'

def crop(img, f, margin):
  (img_w, img_h) = img.size
  images = []
  for pos in f:
    x = pos[0][0] - 5
    y = pos[0][1] - 5
    xx = pos[2][0] + 5
    yy = pos[2][1] + 5
    if x < 0 or y < 0 or xx > img_w or yy > img_h:
      continue
    images.append(img.crop((x, y, xx, yy)))
  images[8].save("/content/test/out.jpg")

def drawlines(img, lines):
  (img_w, img_h) = img.size
  img_m = img_w/2
  draw = ImageDraw.Draw(img) 
  for i in range(len(lines)):
    y1 = -lines[i]*img_m + i*10
    y2 = lines[i]*img_m + i*10
    if y1 < 0 or y1 > img_h or y2 < 0 or y2 > img_h:
      continue
    draw.line((0,y1, img_w,y2), fill=0)
  img.save("/content/test/out2.jpg")

def get_line(point, lines, img_m):
  (x,y) = point
  for (i, line) in enumerate(lines):
    yy = i*10 - y
    xx = img_m - x
    if xx*line > yy:
      return i



  



def export(image, prediction_result, output_dir):
  exported_file_paths = export_detected_regions(
      image=image,
      regions=prediction_result["boxes"],
      output_dir=output_dir,
      rectify=True
  )

  # export heatmap, detection points, box visualization
  export_extra_results(
      image=image,
      regions=prediction_result["boxes"],
      heatmaps=prediction_result["heatmaps"],
      output_dir=output_dir
  )

if __name__ == '__main__':
  image = read_image(image_path)

  refine_net = load_refinenet_model(cuda=False)
  craft_net = load_craftnet_model(cuda=False)
  prediction_result = get_prediction(
      image=image,
      craft_net=craft_net,
      refine_net=refine_net,
      text_threshold=0.5,
      link_threshold=0.4,
      low_text=0.4,
      cuda=False,
      long_size=1280
  )

  img = Image.open(image_path)
  boxes = prediction_result["boxes"]

  # crop(img, boxes, 5)
  # export(image, prediction_result, "/content/drive/MyDrive/AI Projects/OCR Project/Image/Test/Cropped/TrungLam")

  #make lines
  (img_w, img_h) = img.size
  img_m = img_w/2
  lines = [0]*int(img_h/10)
  for b in boxes:
    x12 = b[0][0] - b[1][0]
    y12 = b[0][1] - b[1][1]
    x23 = b[1][0] - img_m
    y3 =  b[1][1] - y12/x12 * x23
    if y3> 0 and y3<img_h:
      lines[int(y3/10)] = y12/x12
  
  #smooting
  last = lines[0]
  for i in range(len(lines)):
    if lines[i] != 0:
      last = lines[i]
    else:
      lines[i] = last
  lines = np.array(lines)

  m = 5
  for i in range(5):
    for j in range(m,len(lines)-m):    
      lines[j] = sum(lines[j-m:j+m])/len(lines[j-m:j+m])

  #draw lines
  # drawlines(img, lines)

  draw = ImageDraw.Draw(img) 
  for b in boxes:
    m_y = (b[2][1] + b[0][1])/2
    m_x = (b[2][0] + b[0][0])/2
    for (i, line) in enumerate(lines):
      yy = i*10 - m_y
      xx = img_m - m_x

      if m_x*line + i * 10 > m_y:
        y1 = -line*img_m + i*10
        y2 = line*img_m + i*10
        if y1 < 0 or y1 > img_h or y2 < 0 or y2 > img_h:
          continue
        draw.line((0,y1, img_w,y2), fill=128)
        break
  
  img.save("/content/test/out2.jpg")
        








  
