import cv2
import math
import os
import random as rnd
import numpy as np

from PIL import Image, ImageDraw, ImageFilter

def make_background(height, width, fit):
    image = np.ones((height, width)) * 255

    cv2.randn(image, rnd.randint(120,255), rnd.randint(3,12))

    color = rnd.randint(0,3)
    size = rnd.randint(1,2)
    check = False
    #gạch trên
    mode = rnd.randint(0,20)
    if mode == 1 and not fit:
      pos = rnd.randint(0,4)
      image[pos:pos + size,:] = color
      check = True
    
    #gạch dưới
    mode = rnd.randint(0,20)
    if mode == 1 and not fit:
      pos= rnd.randint(0,8)
      image[-pos-size:-pos,:] = color
      check = True

    #gạch trái
    mode = rnd.randint(0,20)
    if mode == 1 and not fit:
      pos = rnd.randint(0,4)
      image[:,pos:pos + size] = color
      check = True

    #gạch phải
    mode = rnd.randint(0,20)
    if mode == 1 and not fit:
      pos = rnd.randint(0,4)
      image[:,-pos-size:-pos] = color
      check = True

    #chấm
    mode = rnd.randint(0,20)
    if not check and not fit and mode == 1:
      size = rnd.randint(2,4)
      if rnd.randint(0,1) == 0:
        pos = rnd.randint(0,8)
      else:
        pos = rnd.randint(52,63-size)
      for i in range(-pos-size, -pos):
        for j in range(0, len(image[0])):
          if (j // size) % 2 == 0:
            image[i][j] = color

    return Image.fromarray(image).convert("RGBA")