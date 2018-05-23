import PIL
from PIL import Image

from JSONextractor import *

path = "/home/luca/Desktop/ComV/Train_Eval_MASKRCNN" ##TODO: path to your working folder
jsonName = "labelbox.json"                ##TODO: your json file name

paths = [path + "/" + jsonName,
         path,
         path + "/" + "pngImages",
         path + "/" + "bmpImages"]


test = JSONextractor(paths)
test.extraction()
test.testing()
