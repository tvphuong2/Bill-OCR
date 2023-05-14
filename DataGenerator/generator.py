import argparse
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random as rnd
import string

from string_generator import create_strings_from_dict
from image_generator import image_generator

def margins(margin):
  margins = margin.split(",")
  if len(margins) == 1:
      return [int(margins[0])] * 4
  return [int(m) for m in margins]

def main(args):

#create string
  lang_dict = []
  if os.path.isfile(args.dict_dir):
    with open(args.dict_dir, "r", encoding="utf8", errors="ignore") as d:
      lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
  else:
      sys.exit("Cannot open dict")
  
  strings = create_strings_from_dict(args.count, lang_dict,  args.text_scale)
  string_count = len(strings)

# collect fonts
  fonts = [
    os.path.join(args.font_dir, p)
    for p in os.listdir(args.font_dir)
    if os.path.splitext(p)[1] == ".ttf"
  ]

#image generation
  for i in tqdm(range(string_count)):
    image_generator(args, strings[i], fonts)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--count", type=int, default="1000")
  parser.add_argument("--blur", type=int, default="1")
  parser.add_argument("--text_color", type=str, default="#242424,#000000")
  parser.add_argument("--size", type=int, default="64")
  parser.add_argument("--width", type=int, default="-1")
  parser.add_argument("--skewing_angle", type=int, default="1")
  parser.add_argument("--text_scale", type=float, default="0.7")
  parser.add_argument("--space_width", type=float, default="1.0")
  parser.add_argument("--fit_scale", type=float, default="0.1")
  parser.add_argument("--margins", type=margins, default=(5,5,5,5))
  parser.add_argument("--output_dir", type=str, default="/content/Blur2.0")
  parser.add_argument("--extension", type=str, default="jpg")
  parser.add_argument("--dict_dir", type=str, default="/content/drive/MyDrive/Project/DataGenerator/Dict/Viet74K.txt")
  parser.add_argument("--font_dir", type=str, default="/content/drive/MyDrive/Project/DataGenerator/trdg/fonts/vi")


  args = parser.parse_args()

  main(args)