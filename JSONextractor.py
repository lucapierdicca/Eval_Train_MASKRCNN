import json
import urllib.request
import os

from PIL import Image
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
        self.classes =[]
        self.numObject = []

        self.nomeBase = "image"
        self.labels = []

        os.chdir(self.directoryPath)

        if not os.path.exists(self.pngPath):
            os.makedirs(self.pngPath)

        if not os.path.exists(self.bmpPath):
            os.makedirs(self.bmpPath)

        b = json.load(open(self.path))
        for xx in range(len(b)):
            name = ''
            if b[xx]['Label'] == "Skip":
                continue
            for x in b[xx]['Label'].keys():
                name = x
                if name not in self.classes:
                    self.classes.append(name)
                    self.numObject.append(1)
                    self.labels.append(0)
                else:
                    self.numObject[self.classes.index(name)] += 1
        self.initStampa(self.classes, self.numObject)



    def initStampa(self, ogg, num):
        print("there are ", len(ogg), " objects.")
        count = 0
        for x in range(len(ogg)):
            print (ogg[x], " appears ", num[x], " times.")
            count += num[x]

        print ("There are ", count, " objects labeled in total.")


    def extraction(self):

        b = json.load(open(self.path))
        self.numberOfLabels = len(b)
        name = ''
        for immNum in range(len(b)):
            if b[immNum]['Label'] == "Skip":
                continue
            name = self.nomeBase + str(immNum)
            imm = b[immNum]['Labeled Data']
            os.chdir(self.pngPath)
            urllib.request.urlretrieve(imm, name + ".png")
            self.converti(name)

            if "Mask" in b[immNum].keys():
                print("single mask")
                for x in b[immNum]['Label'].keys():
                    name = x
                nameApp = name
                name = self.nomeBase + str(immNum) + name
                imm = b[immNum]['Mask'][nameApp]
                os.chdir(self.pngPath)
                urllib.request.urlretrieve(imm, name + ".png")
                os.chdir(self.bmpPath)
                self.converti(name)

            else:
                if "Masks" in b[immNum]:
                    for x in range(len(self.labels)):
                        self.labels[x] = 0

                    if len(b[immNum]['Label']) == 1:
                        for x in b[immNum]['Label'].keys():
                            name = x
                        nameApp = name
                        name = self.nomeBase + str(immNum) + name
                        imm = b[immNum]['Masks'][nameApp]
                        os.chdir(self.pngPath)
                        urllib.request.urlretrieve(imm, name + ".png")
                        os.chdir(self.bmpPath)
                        self.converti(name)
                    else:
                        for x in b[immNum]['Label'].keys():
                            name = x
                            name = self.nomeBase + str(immNum) + name + str(self.labels[self.classes.index(name)])
                            self.labels[self.classes.index(x)] += 1
                            imm = b[immNum]['Masks'][x]
                            os.chdir(self.pngPath)
                            urllib.request.urlretrieve(imm, name + ".png")
                            os.chdir(self.bmpPath)
                            self.converti(name)



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
            os.remove(path) 


    def testing(self):
        jsonPath = self.path
        b = json.load(open(jsonPath))
        classes = []
        image_ids = []  # riempire con gli id di tutte le immagini non skippate
        for xx in range(len(b)):
            if b[xx]['Label'] == "Skip":
                continue
            else:
                image_ids.append(xx)
            for x in b[xx]['Label'].keys():
                name = x
                if name not in classes:
                    classes.append(name)
        print("classes --> ", classes)
        print("images_ids --> ", image_ids)

