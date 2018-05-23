import json
import urllib.request
import os

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys


class JSONextractor():

    def __init__(self, paths):

        self.path = paths[0]
        self.directoryPath = paths[1]
        self.pngPath = paths[2]
        self.bmpPath = paths[3]
        self.keyDict = {}
        self.numberOfLabels = 0
        self.class_names =[]
        self.numObject = []

        self.nomeBase = "image"
        self.class_counters = []
        self.b = json.load(open(self.path))

        os.chdir(self.directoryPath)

        if not os.path.exists(self.pngPath):
            os.makedirs(self.pngPath)

        if not os.path.exists(self.bmpPath):
            os.makedirs(self.bmpPath)

        self.b = [img for img in self.b if 'Masks' in img and 'image_problems' not in img['Label']]
        #self.b = self.b[1000:1020]

        for xx in range(len(self.b)):
            name = ''
            if self.b[xx]['Label'] == "Skip":
                continue
            for x in self.b[xx]['Label'].keys():
                name = x
                if name not in self.class_names:
                    self.class_names.append(name)
                    self.numObject.append(1)
                    self.class_counters.append(0)
                else:
                    self.numObject[self.class_names.index(name)] += 1
        self.initStampa(self.class_names, self.numObject)



    def initStampa(self, ogg, num):
        print("there are ", len(ogg), " objects.")
        count = 0
        for x in range(len(ogg)):
            print (ogg[x], " appears ", num[x], " times.")
            count += num[x]

        print ("There are ", count, " objects labeled in total.")



    def extraction(self):
        self.numberOfLabels = len(self.b)
        name = ''

        for immNum in range(len(self.b)):
            if self.b[immNum]['Label'] == "Skip":
                continue
            name = self.nomeBase + str(immNum)
            imm = self.b[immNum]['Labeled Data']
            os.chdir(self.pngPath)
            urllib.request.urlretrieve(imm, name + ".png")

            self.converti(name)

            #img size di img appena scaricata
            #I have to use it for mask(s) creation 
            im = Image.open(name + ".png")
            width, height = im.size
            im.close()

            for label_name in self.b[immNum]['Label'].keys():
                mask_name = self.nomeBase + str(immNum) + label_name
                polygon_list = self.b[immNum]['Label'][label_name]
                self.createMask(polygon_list, width, height, mask_name)



    def createMask(self, polygon_list, width, height, mask_name):
        vector = []
        for polygon in polygon_list:
            vector.append([(vertex['x'], height-vertex['y']) for vertex in polygon])
        
        if('Straight Razor' not in mask_name):
            for i, polygon in enumerate(vector):
                img_size = (width, height)
                mask = Image.new('RGB', img_size)
                pdraw = ImageDraw.Draw(mask)
                pdraw.polygon(polygon, fill=(255, 255, 255, 127), outline=(255, 255, 255, 255))

                os.chdir(self.pngPath)
                mask.save(mask_name + str(i) + '.png')
                os.chdir(self.bmpPath)
                mask.save(mask_name + str(i) + '.bmp') 
        else:
            img_size = (width, height)
            mask = Image.new('RGB', img_size)
            pdraw = ImageDraw.Draw(mask)
            
            for polygon in vector:
                pdraw.polygon(polygon, fill=(255, 255, 255, 127), outline=(255, 255, 255, 255))

            os.chdir(self.pngPath)
            mask.save(mask_name + '0' + '.png')
            os.chdir(self.bmpPath)
            mask.save(mask_name + '0' + '.bmp') 



    def stampa(self):
        print(self.keyDict)

    def converti(self, name):
        path = self.pngPath + "/" + name + ".png"
        try:
            img = Image.open(path)
            file_out = self.bmpPath + "/" + name + ".bmp"
            img.save(file_out)
            img.close()
        except OSError:
            print("Error: "+path)
            #os.remove(path) 


    def testing(self):
        jsonPath = self.path
        classes = []
        image_ids = []  # riempire con gli id di tutte le immagini non skippate
        for xx in range(len(self.b)):
            if self.b[xx]['Label'] == "Skip":
                continue
            else:
                image_ids.append(xx)
            for x in self.b[xx]['Label'].keys():
                name = x
                if name not in classes:
                    classes.append(name)
        print("classes --> ", classes)
        print("images_ids --> ", image_ids)

