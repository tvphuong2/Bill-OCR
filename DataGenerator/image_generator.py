import os
import random as rnd

from PIL import Image, ImageFilter, ImageStat

from image_text_generator import make_img_text
from background_generator import make_background

def image_generator(args, text, fonts):
   
  fit = False
  if rnd.randint(1,100) < int(100*args.fit_scale):
    fit = True
    margin_top, margin_left, margin_bottom, margin_right = [0,0,0,0]
  else:
    margin_top, margin_left, margin_bottom, margin_right = args.margins
  horizontal_margin = margin_left + margin_right
  vertical_margin = margin_top + margin_bottom

#create image
  image, mask = make_img_text(args, text, fonts, fit)
  if image == "":
    return

  random_angle = rnd.randint(0 - args.skewing_angle, args.skewing_angle)

#rotate
  rotated_img = image.rotate(random_angle, expand=1)
  # rotated_mask = mask.rotate(random_angle, expand=1)

#resize
  new_width = int(
    rotated_img.size[0]
    * (float(args.size - vertical_margin) / float(rotated_img.size[1]))
  )
  resized_img = rotated_img.resize(
    (new_width, args.size - vertical_margin), Image.ANTIALIAS
  )
  # resized_mask = rotated_mask.resize((new_width, args.size - vertical_margin), Image.NEAREST)

#background
  background_width = args.width if args.width > 0 else new_width + horizontal_margin
  background_height = args.size

  background_img = make_background(background_height, background_width, fit)
  # background_mask = Image.new("RGB", (background_width, background_height), (0, 0, 0))

#alignment center
  new_text_width, _ = resized_img.size

  background_img.paste(
    resized_img,
    (int(background_width / 2 - new_text_width / 2), margin_top),
    resized_img,
  )
  # background_mask.paste(
  #   resized_mask,
  #   (int(background_width / 2 - new_text_width / 2), margin_top),
  # )


  background_img = background_img.convert("RGB")
  # background_mask = background_mask.convert("RGB") 

  gaussian_filter = ImageFilter.GaussianBlur(
      radius= rnd.randint(0, args.blur)
  )
  final_image = background_img.filter(gaussian_filter)
  # final_mask = background_mask.filter(gaussian_filter)

  name = text.replace("/", "*").strip()
  image_name = "{}.{}".format(name, args.extension)
  # mask_name = "{}_mask.png".format(name)
  # box_name = "{}_boxes.txt".format(name)
  # tess_box_name = "{}.box".format(name)

  final_image.save(os.path.join(args.output_dir, image_name))
