import cv2
import tensorflow as tf
from mrcnn import model as modellib
import visiope_full_ALCORLAB
import pickle
import os



def video_dataset_stats():
    videos = []
    dataset_path = '../Train_Eval_ActivityRecoLSTM/PersonalCare'
    video_folders = os.listdir(dataset_path)
    video_folders = sorted([i for i in video_folders if i[0] == '_'])

    classlbl_to_id = {classlbl:id_ for id_,classlbl in enumerate(video_folders)}

    for classlbl in video_folders:
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
      
def check_video_length(fps, n_frames):

    # Video capture
    fps = fps
    n_frames = n_frames 
    print("Tot frames: %d" % n_frames)

    max_n_frames = 2500
    orig_n_frames = n_frames
    orig_fps = fps

    start_frame, end_frame, stride = 0, orig_n_frames, 1

    while (n_frames>max_n_frames) and (fps>7):
        n_frames = n_frames//2
        fps = fps//2

    stride = orig_n_frames//n_frames      

    if (n_frames>max_n_frames):
        adv_n_frames = (n_frames-max_n_frames)//2
        start_frame = stride*adv_n_frames
        end_frame = orig_n_frames-(stride*adv_n_frames)
        n_frames = n_frames-adv_n_frames*2

    print(n_frames)
    print(fps)
    print(start_frame, end_frame, stride)
    print((float(start_frame)/orig_n_frames)*100)

    return start_frame, end_frame, stride, n_frames, fps

def tst(fps, n_frames):
        
    # Video capture
    #vcapture = cv2.VideoCapture(video_name)
    #width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps
    n_frames = n_frames
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
    print(start_frame, end_frame, stride)
    print((float(start_frame)/n_frames)*100)

def video_to_detection(model, video_relative, video_folder, video_name, class_id):
        

    # Video capture
    video_path = video_relative+'/'+video_folder+'/'+video_name

    print(video_path)

    vcapture = cv2.VideoCapture(video_path)
    orig_fps = vcapture.get(cv2.CAP_PROP_FPS)
    orig_n_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame, end_frame, stride, new_n_frames, new_fps = check_video_length(orig_fps, orig_n_frames)

    
    print("Orig frames: %d" % orig_n_frames)
    print("Redux frames: %d" % new_n_frames)
    print("Orig fps: %d" % orig_fps)
    print("Redux fps: %d" % new_fps)


    count = 0
    success = True

    video_info = {'video_name': video_name,
                  'class_id': class_id,
                  'original_nframes': orig_n_frames,
                  'reduced_nframes': new_n_frames,
                  'original_fps': orig_fps,
                  'reduced_fps': new_fps,
                  'frames_info':[]
                  }

    count=0
    curr_frame_index = start_frame
    while success and (i<new_n_frames):
        print("frame: %d / %d" % (count, new_n_frames))
        # Read next image
        vcapture.set(1,curr_frame_index)
        success, image = vcapture.read()
        if success:
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects
            r = model.detect([image], verbose=0)[0]
            
            print("Objs: %d" % len(r['rois']))

            if len(r['class_ids']) > 0:
                video_info['frames_info'].append({'obj_class_ids':r['class_ids'],
                                              'obj_rois':r['rois']})
            count += 1
            curr_frame_index+=stride
      
    video_info['final_nframes'] = len(video_info[frames_info])
    video.append(video_info)

    return video_info

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




    # LOAD CHKPT
    #----------------------------------------------------------------------------------
    weights_path = "./logs_VF/visiope20180707T0933/mask_rcnn_visiope_0090.h5"
    print("Model weights path: ", weights_path)
    model.load_weights(weights_path, by_name=True)





    # VIDEO DETECTION
    #--------------------------------------------------------------------------------
    video_relative = '../Train_Eval_ActivityRecoLSTM/PersonalCare_'

    video_folders = sorted(os.listdir(video_relative))
    classlbl_to_id = {classlbl:id_ for id_,classlbl in enumerate(video_folders)}
    
    dataset_video = []
    for video_folder in os.listdir(video_relative):
        for video_name in video_folder:
            video_info = video_to_detection(model, 
                                            video_relative, 
                                            video_folder, 
                                            video_name, 
                                            classlbl_to_id[video_folder])

    pickle.dump(dataset_video, open('../Train_Eval_ActivityRecoLSTM/dataset_video.pickle','rb'))




main()





