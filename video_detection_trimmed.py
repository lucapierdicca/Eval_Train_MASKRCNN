import cv2
import tensorflow as tf
from mrcnn import model as modellib
import visiope_full_ALCORLAB
import pickle
import os
import json



def video_to_detection(model, video_relative, video_folder, video_name, class_id, annotations):
        
    # Video capture
    video_path = video_relative+'/'+video_folder+'/'+video_name
    vcapture = cv2.VideoCapture(video_path)
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    orig_n_frames = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #get segments from video
    curr_ann = annotations['database'][video_name[:video_name.find('.')]]['annotations']
    
    dataset_segments = {}
    counter = 0
    
    n_tot_segment_frame=0
    for j in curr_ann:
        segment_time = int(j['segment'][1] - j['segment'][0])	
        
        segment_start_frame= int(j['segment'][0])*fps                
        segment_frame = fps*segment_time
        dataset_segments[counter] = {}
        dataset_segments[counter]['segment_start_frame'] = segment_start_frame
        dataset_segments[counter]['segment_n_frame'] = segment_frame
        n_tot_segment_frame = n_tot_segment_frame+segment_frame
        counter+=1
    
    
    print(video_path)
    print("n video frame: %d" % orig_n_frames)
    print("fps: %d" % fps)
    print("n segment: %d" % len(dataset_segments))
    print("n tot segment frame: %d" % n_tot_segment_frame)
    


    
    segments = {}
    for i in dataset_segments:
        print("segment: %d / %d" % (i,len(dataset_segments)))
        count = 0
        success = True
        
        n_frames = dataset_segments[i]['segment_n_frame']
        start_frame = dataset_segments[i]['segment_start_frame']

        segments[i] ={
                      'n_frames': n_frames,
                      'frames_info':[]
                     }
    
        count=0
        curr_frame_index = start_frame
        while success and (count<n_frames):
            print("frame: %d / %d" % (count, n_frames))
            print()
            # Read next image
            vcapture.set(1,curr_frame_index)
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                
                print("objs: %d" % len(r['rois']))
    
                if len(r['class_ids']) > 0:
                    segments[i]['frames_info'].append({'obj_class_ids':r['class_ids'],
                                                       'obj_rois':r['rois'],
                                                       'obj_masks':r['masks']})
                count += 1
                curr_frame_index+=1
    


    video_info = {'video_name': video_name,
                  'class_id': video_folder,
                  'fps': fps,
                  'original_nframes':orig_n_frames, 
                  'segments_nframes':n_tot_segment_frame,
                  'n_segments': len(dataset_segments)}               
    
    video_info['segments'] = segments

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
    video_relative = r'../Train_Eval_ActivityRecoLSTM/PersonalCare'

    #video_folders = sorted(os.listdir(video_relative))
    #classlbl_to_id = {classlbl:id_ for id_,classlbl in enumerate(video_folders)}
    
    #load the annotations
    annotations = json.load(open(r'../Train_Eval_ActivityRecoLSTM/activity_net.v1-3.min.json','r'))

    temp = [i for i in os.listdir(video_relative) if 'pickle' != i]
    for video_folder in temp:
        for video_name in os.listdir(video_relative+'/'+video_folder):
            if os.path.isfile(video_relative+'/'+video_folder+'/'+video_name[:video_name.find('.')]+'_trimmed.pickle') == False:
                video_info = video_to_detection(model,
                                                video_relative, 
                                                video_folder, 
                                                video_name, 
                                                video_folder,
                                                annotations)


                video_name = video_name[:video_name.find('.')]
                pickle.dump(video_info, open(video_relative+'/'+video_folder+'/'+video_name+'_trimmed.pickle','wb'))
                print(video_name+' dumped')




main()





