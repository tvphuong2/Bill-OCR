import random as rnd
import numpy as np

from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter

def make_img_text(args, text, fonts, fit):
  character_spacing = rnd.randint(1,5)

  font = fonts[rnd.randrange(0, len(fonts))]
  image_font = ImageFont.truetype(font=font, size=args.size)
  italic = rnd.randint(0, 4)

#get text size in image
  space_width = int(image_font.getsize(" ")[0] * args.space_width)

  text_height = image_font.getsize(text)[1]
  text_width = image_font.getsize(text)[0]
  text_width += character_spacing * (len(text) - 1)
  if italic == 0:
    text_width += 15

#create image
  if text_width  == 0 or text_height == 0:
    return "",""
  txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
  txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))
  
  txt_img_draw = ImageDraw.Draw(txt_img)
  # txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
  # txt_mask_draw.fontmode = "1"

  colors = [ImageColor.getrgb(c) for c in args.text_color.split(",")]
  c1, c2 = colors[0], colors[-1]

  fill = (
    rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
    rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
    rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
  )

  txt_img_draw.text(
      (0,0),
      text,
      fill=fill,
      font=image_font,
    )
  if italic == 0:
    img = np.asarray(txt_img)
    img = img.copy()
    exp = np.array([[img[0][0]]*18]*len(img))
    img = np.concatenate((exp, img), axis=1)
    s = img.shape
    angl = rnd.randint(4,9)
    if rnd.randint(1,2) == 1:
      for i in range(s[0]):
        for j in range(s[1]):
          img[s[0] - 1 -i,j,:] = img[s[0] - 1 - i,(j+ i//angl)%s[1],:]
    else:
      for i in range(s[0]):
        for j in range(s[1]):
          img[s[0] - 1 -i,j,:] = img[s[0] - 1 - i,(j+ i//angl)%s[1],:]
    txt_img = Image.fromarray(img)

  # txt_mask_draw.text(
  #   (0, 0),
  #   text,
  #   fill=(0,0,3),
  #   font=image_font,
  # )

  if fit:
      return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
  else:
      return txt_img, txt_mask

