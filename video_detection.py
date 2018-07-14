import cv2
#import tensorflow as tf
#from mrcnn import model as modellib
#import visiope_full_ALCORLAB
import pickle
import os



def dataset_stats():
    videos = []
    dataset_path = '../Train_Eval_ActivityRecoLSTM/Personal_Care'
    class_folder_list = os.listdir(dataset_path)
    class_folder_list = sorted([i for i in class_folder_list if i[0] == '_'])

    classlbl_to_id = {classlbl:id_ for id_,classlbl in enumerate(class_folder_list)}

    for classlbl in class_folder_list:
        for video in os.listdir(dataset_path+'/'+classlbl):
            curr_id = classlbl_to_id[classlbl]
            
            vcapture = cv2.VideoCapture(dataset_path+'/'+classlbl+'/'+video)
            n_frame = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vcapture.get(cv2.CAP_PROP_FPS)
            
            videos.append({'class_id':curr_id,
                            'n_frame':n_frame,
                            'size':str(width)+'x'+str(height),
                            'fps':fps})    

    agg = {}

    for i in videos:
        if i['class_id'] not in agg:
            agg[i['class_id']]=[(i['class_id'],i['fps'],i['n_frame'])]
        else:
            agg[i['class_id']].append((i['class_id'],i['fps'],i['n_frame']))

    return videos, agg
        
    


def detection_to_video(model, show_bbox=False, early_stop=0, video_path=None, chkpt=0):
        

    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    n_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Tot frames: %d" % n_frames)

    max_n_frames = 2500
    orig_n_frames = n_frames

    start_frame, end_frame, stride = 0, orig_n_frames, 1

    while (n_frames>max_n_frames) and (fps>=7):
        n_frames = n_frames//2
        fps = fps//2

    if (n_frames>max_n_frames):
        adv_n_frame = (max_n_frames-n_frames)//2
        stride = origin_n_frames//n_frames
        start_frame = stride*adv_n_frame
        end_frame = orig_n_frames-(stride*adv_n_frame)
    
    stride = origin_n_frames//n_frames


    video_path = video_path[:video_path.find('.')]

    
    detection_list_name = video_path+'_'+chkpt+'.pickle'
    
    count = 0
    success = True
    detection_list = []
    #early_stop = early_stop


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


def main():

    # CONFIG
    #--------------------------------------------------------------------
    config = visiope_full_ALCORLAB.VisiopeConfig()

    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.6

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
        description='Train Mask R-CNN on visiope_full_ALCORLAB')
    parser.add_argument("--video", required=True,
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")


    args = parser.parse_args()



    # LOAD CHKPT
    #----------------------------------------------------------------------------------
    weights_path = "./logs_VF/visiope20180707T0933/mask_rcnn_visiope_00"+args.model+".h5"
    print("Model weights path: ", weights_path)
    model.load_weights(weights_path, by_name=True)
    detection_to_video(model, show_bbox=False, early_stop=0, video_path=args.video, chkpt=args.model)



def tst():
        

    # Video capture
    #vcapture = cv2.VideoCapture(video_path)
    #width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    n_frames = 4000
    print("Tot frames: %d" % n_frames)

    # stride = fps
    # while n_frames > 300:
    #     stride = stride // 15
    #     n_frames = n_frames/stride

    if n_frames > 2500:
        stride = fps//15

    i=2
    new_n_frames = n_frames
    while (new_n_frames > 2500) and (i<=stride):
        new_n_frames = n_frames // i
        i+=1

    stride = n_frames//new_n_frames

    start_frame, end_frame = 0,n_frames
    if (new_n_frames - 2500)>0:
        frames_to_be_removed = new_n_frames - 2500
        frames_to_be_removed = frames_to_be_removed // 2

        start_frame = frames_to_be_removed*stride
        end_frame = n_frames - frames_to_be_removed*stride

    print((end_frame-start_frame)//stride)
    print(fps//stride)
    print((float(start_frame)/n_frames)*100)

def test2():

    # Video capture
    fps = 30
    n_frames = 4000 
    print("Tot frames: %d" % n_frames)

    max_n_frames = 2500
    orig_n_frames = n_frames

    start_frame, end_frame, stride = 0, orig_n_frames, 1

    while (n_frames>max_n_frames) and (fps>=7):
        n_frames = n_frames//2
        fps = fps//2

    if (n_frames>max_n_frames):
        adv_n_frame = (max_n_frames-n_frames)//2
        stride = orig_n_frames//n_frames
        start_frame = stride*adv_n_frame
        end_frame = orig_n_frames-(stride*adv_n_frame)
    
    stride = orig_n_frames//n_frames

    print(n_frames)
    print(fps)
    print(start_frame, end_frame, stride)

# VIDEO
#---------------------------------------------------------------------


tst()
test2()


