#https://github.com/ptran1203/pytorch-animeGAN
import torch

import cv2
import random
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import read_image, resize_image

from PIL import Image

os.chdir("/home/sangmin/cft/Pytorch-animeGAN") 
os.getcwd()

##### generating image 'hayao' #####
from inference import Transformer
transformer = Transformer('hayao')  

# -----------------------------------------------
# --------------- EXAMPLE -----------------------
#####loading file and generating image 
img = Image.open("/home/sangmin/cft/Pytorch-animeGAN/CFT_bg_ref/city/cft_bg_ref(5).png")
type(img)
img_to_Array = np.array(img)
# The fourth layer is the transparency value for image formats that support transparency, like PNG. 
# If you remove the 4th value it'll be a correct RGB image without transparency.
img_to_Array.shape
# therefore, u need to follow below stage.
img_RGB = img_to_Array[...,:3]
img_resize = resize_image(img_RGB)
#u can control below code for filtering degree
anime_img = (transformer.transform(img_resize) + 1) / 2
# output to image
plt.imshow(anime_img[0]) #array format (H,W,C)
plt.axis('off') # only graph ficture
plt.savefig("/home/sangmin/cft/Pytorch-animeGAN/hayao/hayao_test.png", bbox_inches='tight')
# ------------------------------------------------
# ------------------------------------------------

# -----------------------------------------------
# --------------- PROJECT STAGE -----------------
#### getting pathway into img_list
img_list = []
for(path, dir, files) in os.walk("/home/sangmin/cft/Pytorch-animeGAN/CFT_bg_ref"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.png':
            img_list.append("%s/%s" % (path,filename))

img_list[1] #check

##### generating loop for images #####
for i, name in enumerate(img_list):
    img = Image.open(name)
    imgArray = np.array(img) #(...,4)
    #resizing needed because of channel dim
    image = imgArray[...,:3]
    image = resize_image(image)

    #generating stage
    anime_img = (transformer.transform(image) + 1) / 2
    plt.imshow(anime_img[0]) #array format (H,W,C)
    #storing as jpg format into folder 'hayao'

    plt.axis('off')
    plt.savefig("/home/sangmin/cft/Pytorch-animeGAN/hayao/hayao_{}.png".format(i), bbox_inches='tight')
# ------------------------------------------------
# ------------------------------------------------