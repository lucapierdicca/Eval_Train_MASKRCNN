import PIL
from PIL import Image

from JSONextractor import *

path = "/home/luca/Desktop/ComV/LABE2/" ##TODO: path to your working folder
jsonName = "E.json"                ##TODO: your json file name

paths = [path + "/" + jsonName,
         path,
         path + "/" + "pngImages",
         path + "/" + "bmpImages"]


test = JSONextractor(paths)
test.extraction()
test.testing()
