import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import visiope_full
import cv2
import pickle
import colorsys



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
    figsize: (optional) the size of the image
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
        color = colors[i]

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
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()



def detection_to_video(model, dataset, video_path=None):
        
    import cv2
    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    n_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Tot frames: %d" % n_frames)

    
    detection_list_name = "det_list_"+video_path
    
    count = 0
    success = True
    detection_list = []
    early_stop = 100
	
    while success:
        print("frame: %d / %d" % (count, n_frames))
        # Read next image
        success, image = vcapture.read()
        if success:
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects
            r = model.detect([image], verbose=0)[0]
            print("ROIS: %d" % len(r['rois']))
            detection_list.append(r)
			# Create a plot made of frame+masks+bboxes
            display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, scores=r['scores'])
            # Save it on the HDD
            plt.savefig("image" + str(count),facecolor=None, edgecolor=None)
            plt.close()
            
            count += 1

            if count == early_stop:
                break
          
    pickle.dump(detection_list, open(detection_list_name, 'wb'))



    img_range = 0
    if early_stop == 0:
        img_range=n_frames
    else:
        img_range = early_stop

    file_name = "detected_"+video_path
    
    img = cv2.imread('image0.png')
    height,width,layers=img.shape
    
    vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    
    for i in range(img_range-1):
        img = cv2.imread('image'+str(i)+'.png')
        vwriter.write(img)

    vwriter.release()
    print("Saved to ", file_name)


    

# CONFIG
#--------------------------------------------------------------------
config = visiope_full.VisiopeConfig()

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #DETECTION_MIN_CONFIDENCE = 0

config = InferenceConfig()
#config.display()
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0 





# MODEL
#---------------------------------------------------------------------
# Create model in inference mode
MODEL_DIR= r"./logs_VF"

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)






# LOAD CHKPT
#----------------------------------------------------------------------------------
weights_path = r"./logs_VF/mask_rcnn_visiope_0080.h5"
print("Model weights path: ", weights_path)
model.load_weights(weights_path, by_name=True)







# LOAD DATASET (we need it to obtain the mapping class_id:class_name)
#---------------------------------------------------------------------------------
selected_COCO_class_ids = [27,31,47,51,62,65,67,70,72,73,78,79,81,82,84,85,87,90]


print("\n======VALIDATION SET======")
dataset_val = visiope_full.VisiopeDataset()

dataset_val.load_visiope(sampling="val")
print("\nN tot_visiope images: %d" % len(dataset_val.b))

n_val_visiope_imgs = len(dataset_val.image_info)
print("N val_visiope images: %d" % n_val_visiope_imgs)


dataset_val.load_coco(sampling="val", class_ids=selected_COCO_class_ids)
n_val_COCO_imgs = len(dataset_val.image_info)-n_val_visiope_imgs
print("N val_COCO images: %d" % n_val_COCO_imgs)

dataset_val.prepare()
print("N tot val images (val_visiope + val_COCO): %d\n" % len(dataset_val.image_info))







# VIDEO
#---------------------------------------------------------------------
video_path= "v.mp4"

detection_to_video(model, dataset_val, video_path=video_path)

