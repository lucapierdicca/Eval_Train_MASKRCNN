import cv2
import tensorflow as tf
from mrcnn import model as modellib




def detection_to_video(model, show_bbox=False, early_stop=0, video_path=None, chkpt=0):
        

    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    n_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Tot frames: %d" % n_frames)

    video_path = video_path[:video_path.find('.')]

    
    detection_list_name = video_path+'_'+chkpt+'.pickle'
    
    count = 0
    success = True
    detection_list = []
    early_stop = early_stop


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

            count += 1

            if count == early_stop:
                break
          
    pickle.dump(detection_list, open(detection_list_name, 'wb'))



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
MODEL_DIR= "./logs_VF"

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Train Mask R-CNN on VISIOPE_FULL')
parser.add_argument("--video", required=True,
                    metavar="<command>",
                    help="'train' or 'evaluate' on MS COCO")
parser.add_argument('--model', required=True,
                    metavar="/path/to/weights.h5",
                    help="Path to weights .h5 file or 'coco'")


args = parser.parse_args()



# LOAD CHKPT
#----------------------------------------------------------------------------------
weights_path = "./logs_VF/mask_rcnn_visiope_00"+args.model+".h5"
print("Model weights path: ", weights_path)
model.load_weights(weights_path, by_name=True)





# VIDEO
#---------------------------------------------------------------------

detection_to_video(model, show_bbox=False, early_stop=0, video_path=args.video, chkpt=args.model)

