import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
from pprint import pprint
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import cv2
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import random




def coordToMatrix(coord, w, h):
    img_size = (w, h)
    poly = Image.new("RGB", img_size)
    pdraw = ImageDraw.Draw(poly)
    pdraw.polygon(coord,
                  fill=(255,255,255), outline=(255,255,255))
    poly = poly.transpose(Image.FLIP_LEFT_RIGHT)
    poly = poly.rotate(180)
    #pix = np.array(poly.getdata()).reshape(w, h)
    return poly

def find_centroid(im):
    width, height = im.size
    XX, YY, count = 0, 0, 0
    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if im.getpixel((x, y)) == (255,255,255):
                XX += x
                YY += y
                count += 1
    return XX/count, YY/count

def compute_area(im):
    width, height = im.size
    area = 0
    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if im.getpixel((x, y)) == (255,255,255):
                area += 1
    return area

def find_max_coord(x, y):
    x_max = 0
    x_min = 10000000
    y_max = 0
    y_min = 10000000
    for indice in range(len(x)):
        if x[indice] < x_min:
            x_min = x[indice]
        if y[indice] < y_min:
            y_min = y[indice]
        if x[indice] > x_max:
            x_max = x[indice]
        if y[indice] > x_max:
            x_max = y[indice]
    return [x_max, x_min, y_max, y_min]

def cade_internamente(max, centroide):
    if centroide[0]< max[0] and centroide[0] > max[1]:
        if centroide[1]< max[2] and centroide[1] > max[3]:
            return attributes
    return False

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def find_person(fig):
    indexList = [0,1,2,3,4,5,6,7,8]
    results = model.detect([fig], verbose=0)
    r = results[0]
    ax = get_ax(1)
    r = results[0]
    # ['BG', 'screwdriver', 'belt', 'guard', 'mesh', 'spanner', 'boh1', 'boh2'], r['scores']
    visualize.display_instances(fig, r['rois'], r['masks'], r['class_ids'], class_names , r['scores'], ax=ax, title="Predictions")

    ids = r['class_ids']
    ret = []
    for i in indexList:
        ret.append(np.count_nonzero(ids == i))
    return ids

def centreAnalisi(fig, w, h):

    results = model.detect([fig], verbose=0)
    r = results[0]
    ids = r['class_ids']
    maschere = r["masks"]
    numMasks = 0
    try:
        numMasks = maschere.shape[2]
    except Exception as e:
        print(e)
        print('EEEE')
        return 0

    maskRet = [np.zeros((h, w), dtype=(np.uint8,3)) for i in range(numMasks)]

    for i in range(numMasks):
        for h in range(maschere.shape[0]):
            for w in range(maschere.shape[1]):
                if maschere[h,w,i]:
                    maskRet[i][h,w] = (255, 255, 255)


    centroidi_ret= []
    aree = []
    for i in range(len(maskRet)):
        image = Image.fromarray(maskRet[i], 'RGB')
        aree.append(compute_area(image))
        ret = find_centroid(image)
        centroidi_ret.append(ret)
    return centroidi_ret, ids, aree

def test():
    file_names = next(os.walk(IMAGE_DIR))[2]
    for f in file_names:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, f))
        number = find_person(image)
        print(number)

class VisiopeDataset(utils.Dataset):

    def load_visiope(self, sampling, return_coco=False):

        assert sampling in ["train", "val"]

        #Dataset main paths initialization
        self.path = PNG_IMAGES_PATH 
        self.jsonName = JSON_PATH

        #.JSON storing in main memory
        b = json.load(open(self.jsonName))
        self.b = list(b)


        #Random splitting train-val
        all_images_ids = range(len(self.b))

        np.random.seed(0)
        train_images_ids = sorted(np.random.choice(len(self.b), 
                                                   replace=False, 
                                                   size=int(len(self.b)*0.9)).tolist())
        val_images_ids = sorted(list(set(all_images_ids)-set(train_images_ids)))



        #Classes rebuilding
        classes = []
        image_ids = [] 

        if sampling == 'train':
            selected_subset = train_images_ids
        else:
            selected_subset = val_images_ids
            
        for xx in selected_subset:
            name = xx
            if self.b[xx]['Label'] == "Skip":
                continue
            else:
                image_ids.append(xx)
            for x in self.b[xx]['Label'].keys():
                name = x
                if name not in classes:
                    classes.append(name)

        classes = sorted(classes)


        #Add classes
        for i in range(len(classes)):
            self.add_class("visiope", i+1, classes[i]) #cerca add_class

        # Add images
        for i in image_ids:
            self.add_image("visiope", 
                            image_id=i, 
                            path=PNG_IMAGES_PATH+"/image"+str(i)+".jpeg", 
                            labels={lbl:len(val) if lbl != 'Straight razor' else 1 for lbl,val in self.b[i]['Label'].items()}) 

        if return_coco:
            return self.b


    def get_dataset_distribution(self):
        COCOcatid_to_label={i['id']:i['name'] for i in self.class_info if i['source'] == 'coco'}

        stat_dict = {}
        for i in self.image_info:
            if i['source'] == 'visiope':
                for k,v in i['labels'].items():
                    if k in stat_dict:
                        stat_dict[k] += v
                    else:
                        stat_dict[k] = v
            else:
                for j in i['annotations']:
                    if COCOcatid_to_label[j['category_id']] in stat_dict:
                        stat_dict[COCOcatid_to_label[j['category_id']]] += 1
                    else:
                        stat_dict[COCOcatid_to_label[j['category_id']]] = 1     

        tot = 0
        for v in stat_dict.values():
            tot+=v

        for k,v in stat_dict.items():
            stat_dict[k] = (v, str(round(v/tot*100))+"%")

        return stat_dict









# Model
#----------------------------------------------------------------
DEFAULT_LOGS_DIR = "./logs_VF"

PNG_IMAGES_PATH = "./pngImages_mod" 
BMP_IMAGES_PATH = "./bmpImages_mod"
JSON_PATH       = "labelbox_mod.json"

FINETUNED_MODEL_PATH = DEFAULT_LOGS_DIR+'/'+'mask_rcnn_visiope_0080.h5'
#FINETUNED_MODEL_PATH = 'mask_rcnn_coco.h5'



class InferenceConfig(Config):
    NAME = "visiope"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1+15+18
    DETECTION_MIN_CONFIDENCE = 0.6

config = InferenceConfig()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=DEFAULT_LOGS_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(FINETUNED_MODEL_PATH, by_name=True)






# Load the validation dataset
#----------------------------------------------------------
print("\n======VALIDATION SET======")
dataset_val = VisiopeDataset()

dataset_val.load_visiope(sampling="val")
print("\nN tot_visiope images: %d" % len(dataset_val.b))

n_val_visiope_imgs = len(dataset_val.image_info)
print("N val_visiope images: %d" % n_val_visiope_imgs)

dataset_val.prepare()

dataset_val_distro = dataset_val.get_dataset_distribution()
print("N val_classes: %d" % len(dataset_val_distro))
pprint(dataset_val_distro)

#print(dataset_val.image_info)
#print(dataset_val.class_info)
#print(dataset_val.class_names)






# Evaluation
#------------------------------------------------------
b = dataset_val.b

success = 0
total = 1

classes = {'Electric Razor':1, 
            'Eye':2, 
            'Finger':3, 
            'Grabbing hands':4, 
            'Leg':5, 
            'Lenses':6, 
            'Makeup tool':7, 
            'Mouthwasher':8, 
            'Razor Pit':9, 
            'Shaving cream':10, 
            'Soap':11, 
            'Soapy hands':12, 
            'Straight Razor':13, 
            'Tap':14, 
            'Wash Basin':15
            }

for json_elem in dataset_val.image_info:
    if b[json_elem['id']]['Label'] == 'Skip':
        continue
    # read image for sizes
    img_path = json_elem['path']
    print(img_path)
    try:
        im = Image.open(img_path)
    except Exception as e:
        print('AAAA')
        print(e)

    w, h = im.size
    seg = b[json_elem['id']]["Label"]
    maskMat = []
    idClassi = []

    idss = []
    centroidi_lista = []
    aree = []
    max_coord = []
    for i in seg.keys():
        name = str(i)
        class_id = classes[name]

        for j in range(len(seg[name])):
            idClassi.append(classes.get(name))
            x_coord = []
            y_coord = []
            for k in range(len(seg[name][j])):
                y_coord.append(seg[name][j][k]['y'])
                x_coord.append(seg[name][j][k]['x'])
            coord = []
            for ind in range(len(x_coord)):
                coord.append(x_coord[ind])
                coord.append(y_coord[ind])
            immagine = coordToMatrix(coord, w, h)
            centroidi_lista.append(find_centroid(immagine))
            aree.append(compute_area(immagine))
            idss.append(class_id)
            max_coord.append(find_max_coord(x_coord, y_coord))


    centroidi_lista_mask, idss_mask, aree_mask = centreAnalisi(np.array(im), w, h)
    #print(centroidi_lista_mask, idss_mask, aree_mask)
    for indice in range(len(idss)):
        total += 1
        for indice_mask in range(len(idss_mask)):
            if (aree[indice] *0.5)< aree_mask[indice_mask] and aree_mask[indice_mask] < (aree[indice] *1.5):
                if cade_internamente(max_coord[indice], centroidi_lista_mask[indice_mask]):
                    if idss_mask[indice_mask] == idss[indice]:
                        success += 1



print("Numero di successi: %d" % success)
print("Numero totale label: %d" % total)

acc = float(success) / float(total)*100
print("Percentuale di successo: {} %".format(acc))
