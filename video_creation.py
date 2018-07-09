import visiope_full_ALCORLAB
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pickle
import colorsys
from skimage.measure import find_contours
from matplotlib import patches,  lines
import os
import shutil
import random
import numpy as np



def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    
    return ax

	
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors



def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the imagepSp7zYRYjHE.mp4
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = False

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[class_ids[i]-1]

        print("%d:%s:(%f,%f,%f)" % (class_ids[i], class_names[class_ids[i]], color[0], color[1], color[2]))


        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = patches.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()



def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image














# LOAD DATASET (we need it to obtain the mapping class_id:class_name)
#---------------------------------------------------------------------------------
selected_COCO_class_ids = [27,31,47,51,62,65,67,70,72,73,78,79,81,82,84,85,87,90]


print("\n======VALIDATION SET======")
dataset_val = visiope_full_ALCORLAB.VisiopeDataset()

dataset_val.load_visiope(sampling="val")
print("\nN tot_visiope images: %d" % len(dataset_val.b))

n_val_visiope_imgs = len(dataset_val.image_info)
print("N val_visiope images: %d" % n_val_visiope_imgs)


dataset_val.load_coco(sampling="val", class_ids=selected_COCO_class_ids)
n_val_COCO_imgs = len(dataset_val.image_info)-n_val_visiope_imgs
print("N val_COCO images: %d" % n_val_COCO_imgs)

dataset_val.prepare()
print("N tot val images (val_visiope + val_COCO): %d\n" % len(dataset_val.image_info))

print(dataset_val.class_names)

classid_to_classname = {index:name for index, name in enumerate(dataset_val.class_names)}

class_detected_distro = {name:0 for name in dataset_val.class_names}







# VIDEO
#-----------------------------------------------------------------------
video_name = "pSp7zYRYjHE.mp4"
detection_name = "pSp7zYRYjHE_90.pickle"
early_stop = 0




video_path="./VIDEO/"+video_name
detection_path = "./VIDEO/"+detection_name
detection = pickle.load(open(detection_path, 'rb'))


# Video capture
vcapture = cv2.VideoCapture(video_path)
width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vcapture.get(cv2.CAP_PROP_FPS)
n_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
print("Tot frames: %d" % n_frames)


count = 0
success = True
detection_list = []



colors = random_colors(18+15)

video_name = video_name[:video_name.find('.')]

os.mkdir("./VIDEO/imgs"+video_name)

while success:
    print("frame: %d / %d" % (count, n_frames))
    # Read next image
    success, image = vcapture.read()
    if success:
        # OpenCV returns images as BGR, convert to RGB
        image = image[..., ::-1]
        # Detect objects
        cv2.imwrite("./VIDEO/imgs"+video_name+"/image%d.png" % count, image) 
        # Save it on the HDD
        
        count += 1

        if count == early_stop:
            break

file_name = "./VIDEO/detected_"+video_name+'__'+detection_name[detection_name.find('_'):detection_name.find('.')]+".mp4"





# NO POSTPROCESSING
#-------------------------------------
os.mkdir("./VIDEO/temp")

if early_stop==0:
    img_range=n_frames
else:
    img_range=early_stop

c = [0]*33
a = []

for i in range(img_range):
    r = detection[i]


    img = cv2.imread("./VIDEO/imgs"+video_name+"/"+"image"+str(i)+".png")

    display_instances(img, 
                        r['rois'], 
                        r['masks'], 
                        r['class_ids'], 
                        dataset_val.class_names, 
                        scores=r['scores'], 
                        colors=colors, 
                        show_bbox=True)

    # for j in list(set(list(r['class_ids']))):
    #     c[j-1] = j
    #     a.append(c)
    #     c=[0]*33

# f = open('graph_video_data.txt', 'a')
# for i in a:
#     we = ",".join(str(j) for j in i)
#     print(we)
#     f.write(we+'\n')

# f.close()

    plt.savefig("./VIDEO/temp/image"+str(i)+".png", bbox_inches='tight')
    plt.close()

#pickle.dump(class_detected_distro, open('class_detected_distro_40.pickle', 'wb'))



img = cv2.imread("./VIDEO/temp/image0.png")
height,width,layers=img.shape
print(height, width)

vwriter = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'MJPG'), 15.0, (width, height))

for i in range(img_range):
    img = cv2.imread("./VIDEO/temp/image"+str(i)+".png")
    print(img.shape)
    print("./VIDEO/temp/image"+str(i)+".png")
    vwriter.write(img)

vwriter.release()

shutil.rmtree("./VIDEO/temp")
print("Saved to ", file_name)







'''

# POSTPROCESSING
#-------------------------------------
os.mkdir("./VIDEO/temp")
3buffer = []

if early_stop==0:
    img_range=n_frames
else:
    img_range=early_stop

for i in range(img_range):
    r = detection[i]
    
    3buffer.append(r)

    if len(3buffer) >= 3:
        for j in 3buffer:
            class_detected_distro[classid_to_classname[j]] += 1

        3buffer.pop(0)


    else:
        img = cv2.imread("./VIDEO/imgs"+video_name+"/"+"image"+str(i)+".png")

        display_instances(img, 
                            r['rois'], 
                            r['masks'], 
                            r['class_ids'], 
                            dataset_val.class_names, 
                            scores=r['scores'], 
                            colors=colors, visiope20180707T0933
                            show_bbox=True)



        plt.savefig("./VIDEO/temp/image"+str(i)+".png", bbox_inches='tight')
        plt.close()



pickle.dump(class_detected_distro, open('class_detected_distro_40.pickle', 'wb'))






img = cv2.imread("./VIDEO/temp/image0.png")
height,width,layers=img.shape
print(height, width)

vwriter = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'MJPG'), 15.0, (width, height))

for i in range(img_range):
    img = cv2.imread("./VIDEO/temp/image"+str(i)+".png")
    print(img.shape)
    print("./VIDEO/temp/image"+str(i)+".png")
    vwriter.write(img)

vwriter.release()

shutil.rmtree("./VIDEO/temp")
print("Saved to ", file_name)

'''