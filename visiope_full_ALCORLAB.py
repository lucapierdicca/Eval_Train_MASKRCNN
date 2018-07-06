"""
Mask R-CNN
Configurations and data loading code for MS COCO.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco
    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True
    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5
    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last
    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug

from PIL import Image
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from PIL import Image, ImageDraw


DEFAULT_LOGS_DIR = "./logs_VF"


VISIOPE_PNG_IMAGES_PATH = "./pngImages_mod"
VISIOPE_BMP_MASKS_PATH = "./bmpImages_mod"
VISIOPE_JSON_PATH = "labelbox_mod.json"


COCO_IMAGES_PATH = "."
COCO_SUBSET = "train"
COCO_YEAR = "2017"

COCO_MODEL_PATH = "mask_rcnn_coco.h5"

############################################################
#  Configurations
############################################################

class VisiopeConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "visiope"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1  ##TODO: your pc or alcorlab's pc

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 15 + 18 ###TODO your assignment

    STEPS_PER_EPOCH = 1200


STAGE_1_EPOCHS = 40#40
STAGE_2_EPOCHS = 63#120
STAGE_3_EPOCHS = 80#160


############################################################
#  Dataset
############################################################

class VisiopeDataset(utils.Dataset):

    def load_visiope(self, sampling, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        assert sampling in ["train", "val"]

        self.path = VISIOPE_PNG_IMAGES_PATH  ##TODO: add the path to the dataset folder
        self.jsonName = VISIOPE_JSON_PATH  ##TODO: add json file name


        b = json.load(open(self.jsonName))
        #b = [img for img in b if 'Masks' in img and 'image_problems' not in img['Label']]
        self.b = list(b)

        all_images_ids = range(len(self.b))

        np.random.seed(0)
        train_images_ids = sorted(np.random.choice(len(self.b),
                                                   replace=False,
                                                   size=int(len(self.b)*0.9)).tolist())
        val_images_ids = sorted(list(set(all_images_ids)-set(train_images_ids)))

        self.train_images_ids = train_images_ids


        classes = []
        image_ids = []  # riempire con gli id di tutte le immagini non skippate

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
            self.add_class("visiope", i+1, classes[i])

        # Add images
        for i in image_ids:
            self.add_image("visiope",
                            image_id=i,
                            path=VISIOPE_PNG_IMAGES_PATH+"/image"+str(i)+".jpeg",
                            labels={lbl:len(val) if lbl != 'Straight razor' else 1 for lbl,val in self.b[i]['Label'].items()}) #cerca add_image

        if return_coco:
            return self.b


    def get_dataset_distribution(self):
        COCOcatid_to_label={i['id']:i['name'] for i in self.class_info if i['source'] == 'coco'}
        d = {}
        num = 0

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
            stat_dict[k] = (v, str(round(v/tot*100))+"%", v-int(tot/33), int(tot/33))

        return stat_dict



    def load_coco(self, sampling=None, class_ids=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        import random
        assert sampling in ["train", "val"]


        coco = COCO("{}/annotations/instances_{}{}.json".format(COCO_IMAGES_PATH, COCO_SUBSET, COCO_YEAR))
        image_dir = "{}/{}{}".format(COCO_IMAGES_PATH, COCO_SUBSET, COCO_YEAR)


        # Load all classes or a subset?
        if not class_ids:
            # All cla sses
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            coco_nimgs_per_class = int(len(self.train_images_ids)*0.55/18)
            aux = coco_nimgs_per_class

            print("COCO N CLASSES: %d" % len(class_ids))
            print("COCO N IMAGES PER CLASS: %d" % coco_nimgs_per_class)
            print("COCO N TOTAL IMAGES: %d" % (len(class_ids)*coco_nimgs_per_class))
            
            error_list = []#[51,62,84] # bowl - chair - book
            imgid_to_catgs = {}
            
            for id in class_ids:
                
                random_list = []

                if id in error_list:
                    coco_nimgs_per_class = 1

                current_len = len(list(coco.getImgIds(catIds=[id])))-1
                if current_len < coco_nimgs_per_class:
                    coco_nimgs_per_class = current_len
                
                while len(random_list)<coco_nimgs_per_class:
                    start = random.randint(0, current_len)
                    if start not in random_list:
                        random_list.append(start)

                for i in random_list:
                    d = [list(coco.getImgIds(catIds=[id]))[i]]
                    
                    if d[0] not in imgid_to_catgs:
                        imgid_to_catgs[d[0]] = [id]
                    else:
                        imgid_to_catgs[d[0]].append(id)

                    image_ids.extend(d)

                coco_nimgs_per_class = aux
            
            '''print('\n')
            image_ids_original = list(image_ids)
            print('Original', len(image_ids_original))
            # Remove duplicates
            image_ids = list(set(image_ids))
            print('No duplicates', len(image_ids))

            from pprint import pprint
            print('')
            print(len(list(imgid_to_catgs.keys())))
            pprint(imgid_to_catgs)'''



        else:
            # All images
            image_ids = list(coco.imgs.keys())

        np.random.seed(0)
        indices_train_images_ids = sorted(np.random.choice(len(image_ids),
                                                   replace=False,
                                                   size=int(len(image_ids)*0.9)).tolist())
        train_images_ids = sorted([image_ids[i] for i in indices_train_images_ids])

        val_images_ids = sorted(list(set(image_ids)-set(train_images_ids)))


        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        if sampling=="train":
            image_ids = train_images_ids
        else:
            image_ids = val_images_ids
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=imgid_to_catgs[i], iscrowd=None)))


        if return_coco:
            return coco





    def load_mask_visiope(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        path = VISIOPE_BMP_MASKS_PATH  ##TODO: add the path to the dataset folder
        #self.jsonName = "labelbox.json"  ##TODO: add json file name
        self.nomeBase = "image"

        ret1 = []
        ret2 = []

        #jsonPath = self.jsonName
        #b = json.load(open(jsonPath))
        #b = [img for img in b if 'Masks' in img and 'image_problems' not in img['Label']]

        classes = []
        image_ids = []  # riempire con gli id di tutte le immagini non skippate
        masks_per_img = []
        for xx in range(len(self.b)):
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

        labels = [0]*len(classes)

        #dato che ho diviso in (train & val) simple random sampling
        #il mapping indice_imgID salta e
        #sono costretto  prendere l'imgID interna all'(oggetto) dataset self
        immNum = self.image_info[image_id]['id']
        #immNum = image_id

        if len(self.b[immNum]['Label']) == 1:
            for x in self.b[immNum]['Label'].keys():
                name = x
            nameApp = name
            #TODO forse bisgna aggiungere uno 0 alla fine in questa stringa sotto
            name = path + "/image" + str(immNum) + name + '0' + ".bmp"
            aux = self.bmpToBinary(name)
            ret1.append(aux)
            ret2.append(classes.index(nameApp))
        else:
            for x in self.b[immNum]['Label'].keys():
                name = x
                nameApp = name
                name = path + "/image" + str(immNum) + name + str(labels[classes.index(name)]) + ".bmp"
                labels[classes.index(x)] += 1
                aux = self.bmpToBinary(name)
                ret1.append(aux)
                ret2.append(classes.index(nameApp))

        mask = np.stack(ret1, axis=2).astype(np.bool)
        class_ids = np.array(ret2, dtype=np.int32)+1

        return mask, class_ids

    #reads a .bmp mask from HDD (black&white img)
    #returns a boolean list of lists (H x W)
    def bmpToBinary(self, path):
        img = Image.open(path)
        w, h = img.size
        pixels = list(img.getdata())
        #print(len(pixels))
        aux = []
        for i in range(h):
            boolean_pixel_list = [False if rgb_tuple[0]==0 else True for rgb_tuple in pixels[w*i:w*(i+1)]]
            aux.append(boolean_pixel_list)
        return aux


    def load_mask_coco(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)


    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle


    def load_mask(self, image_id):
        if self.image_info[image_id]['source'] == 'visiope':
            mask, class_ids = self.load_mask_visiope(image_id)
        else:
            mask, class_ids = self.load_mask_coco(image_id)

        return mask, class_ids


    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.path + "/image" + (str(image_id)) + ".png"
        return info

############################################################
#  COCO Evaluation
############################################################


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    print(image_ids)

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        print(type(r))
        print(r)
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on VISIOPE_FULL')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model chkpt: ", args.model)
    print("Datasets: VISIOPE %s-%s \n\tCOCO %s" % (VISIOPE_PNG_IMAGES_PATH, VISIOPE_BMP_MASKS_PATH, COCO_IMAGES_PATH+"/"+COCO_SUBSET+COCO_YEAR))
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VisiopeConfig()
    else:
        class InferenceConfig(VisiopeConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            #DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    #config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH

        print("Loading weights ", model_path)
        model.load_weights(model_path,
                            by_name=True,
                            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
    elif args.model.lower() == "last":
        model_path = model.find_last()[1]
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)
    else:
        weights_path = args.model.lower()



    # Train or evaluate
    if args.command == "train":

        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        from pprint import pprint


        #sofa, phon
        selected_COCO_class_ids = [27,31,47,51,62,65,67,70,72,73,78,79,81,82,84,85,87,90]


        print("\n======TRAINING SET======")
        dataset_train = VisiopeDataset()

        dataset_train.load_visiope(sampling="train")
        print("\nN tot_visiope images: %d" % len(dataset_train.b))

        n_train_visiope_imgs = len(dataset_train.image_info)
        print("N train_visiope images: %d" % n_train_visiope_imgs)


        dataset_train.load_coco(sampling="train", class_ids=selected_COCO_class_ids)
        n_train_COCO_imgs = len(dataset_train.image_info)-n_train_visiope_imgs
        print("N train_COCO images: %d" % n_train_COCO_imgs)

        dataset_train.prepare()
        print("N tot train images (train_visiope + train_COCO): %d\n" % len(dataset_train.image_info))

        dataset_train_distro = dataset_train.get_dataset_distribution()
        print("N train_classes: %d" % len(dataset_train_distro))
        pprint(dataset_train_distro)



        print('')
        print('')


        print("\n======VALIDATION SET======")
        dataset_val = VisiopeDataset()

        dataset_val.load_visiope(sampling="val")
        print("\nN tot_visiope images: %d" % len(dataset_val.b))

        n_val_visiope_imgs = len(dataset_val.image_info)
        print("N val_visiope images: %d" % n_val_visiope_imgs)


        dataset_val.load_coco(sampling="val", class_ids=selected_COCO_class_ids)
        n_val_COCO_imgs = len(dataset_val.image_info)-n_val_visiope_imgs
        print("N val_COCO images: %d" % n_val_COCO_imgs)

        dataset_val.prepare()
        print("N tot val images (val_visiope + val_COCO): %d\n" % len(dataset_val.image_info))


        dataset_val_distro = dataset_val.get_dataset_distribution()
        print("N val_classes: %d" % len(dataset_val_distro))
        pprint(dataset_val_distro)


        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        
        #*******************************************************************
        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=STAGE_1_EPOCHS,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=STAGE_2_EPOCHS,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=STAGE_3_EPOCHS,
                    layers='all',
                    augmentation=augmentation)
        print('AAAAAAAAAAAAAAAAAAAAAAAa')
    

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = VisiopeDataset()
        coco = dataset_val.load_visiope(args.dataset, "val", return_coco=True)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
    