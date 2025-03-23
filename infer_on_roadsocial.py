import argparse
parser = argparse.ArgumentParser(description="Input arguments.")
parser.add_argument("--qas_dir", type=str, help="QAs dir")
parser.add_argument("--model_prefix", type=str, help="model prefix")
parser.add_argument("--model_size", type=str, help="model param size")
parser.add_argument("--gpu_id", type=str, default='-1', help="gpu id")
args = parser.parse_args()

##################### CONFIG PARAMS #####################
import os
if(args.gpu_id!='-1'):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
### Model params
model_prefix = args.model_prefix
MODEL_VERSION = args.model_size
### Videos_DIR and QAs_DIR
INPUT_QA_DATA_FOLDER = args.qas_dir
if(INPUT_QA_DATA_FOLDER[-1]=='/'):
    INPUT_QA_DATA_FOLDER = INPUT_QA_DATA_FOLDER[:-1]
dst_video_dir = INPUT_QA_DATA_FOLDER+'_videos/'
OUTPUT_QA_FOLDER = INPUT_QA_DATA_FOLDER + model_prefix+'Inferred/'
if(not os.path.exists(OUTPUT_QA_FOLDER)):
    os.system('cp -r '+ INPUT_QA_DATA_FOLDER + ' ' + OUTPUT_QA_FOLDER)
print(f"########## Script CONFIG: \nMODEL_PREFIX_VERSION - {model_prefix}{MODEL_VERSION}\nOUTPUT_DIR - {OUTPUT_QA_FOLDER}")   # BATCH_SIZE = 1

########## Libraries import
if('Qwen2-VL' in model_prefix):
    from models.q2vl import *
elif('llava-ov' == model_prefix):
    from models.llavaov import *
elif('llava-ov_ft' == model_prefix):
    from models.llavaov_ft import *
else:
    print("No Video LLM library imported for inference!!!")
    pass
import sys, warnings
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from glob import glob

# function to choose the question corresponding to a template ID
def suitable_prompt(template_qa_idx, qa_type_Q):
    if template_qa_idx == 'Viewpoint':
        camera_type = f'''{qa_type_Q}
        Consider camera types, including but not limited to: Handheld cameras, smartphone, camcorder, vehicle-mounted cameras, dashcam, bike-mounted,fixed position cameras, CCTV, security camera, aerial cameras, drone, helicopter-mounted, multi-camera setups such as those used in advertisements or film production, etc. Do not specify camera brands or model names. Focus on the type or category of the recording device.'''
        return camera_type
    elif template_qa_idx == 'RoadEvent_Type':
        road_event_description_type = f'''{qa_type_Q}
        Consider categories of road events, including but not limited to: safe driving practices, traffic Violations, dangerous driving, rash driving, accident, near-miss incident, road rage incidents, infrastructure issues, educational or demonstrative scenarios, post-crash situations, animal-related incidents, defensive driving, etc. Briefly describe the road event.'''
        return road_event_description_type
    elif template_qa_idx == 'Road_type':
        road_type = f'''{qa_type_Q}
        Consider categories of roads and areas, including but not limited to: urban area, city streets, rural roads, highways, expressways, residential areas, commercial area, industrial zones, intersection, T-junctions, roundabouts, elevated roads, flyovers, overpasses, bridges, tunnels, mountain or hilly roads, forest, coastal roads, etc. Do not specify names, addresses, or exact locations. Briefly describe the road or area.'''
        return road_type
    elif template_qa_idx == 'Time_of_day':
        time_of_day_type = f'''{qa_type_Q}
        Consider categories of day time, including but not limited to: morning, early morning, afternoon, late afternoon, evening, night, etc.'''
        return time_of_day_type
    else:
        q_type = f'''{qa_type_Q}'''
        return q_type

### Init VideoLLM
video_llm = VideoLLM(MODEL_VERSION)

###
# prompt_answer_key = 'promptQ_'+model_prefix+MODEL_VERSION
pred_answer_key = 'pred_'+model_prefix+MODEL_VERSION

###### VideoLLM inference
allqas_src = glob(OUTPUT_QA_FOLDER+'/*.json')
for qa_path in tqdm(allqas_src):
    qa_json = json.load(open(qa_path, 'r'))

    ### create video path
    # if(qa_json['RoadEventRelated']):
    #     video_path = 'v_'+str(qa_json['Tweet_ID'])+'_'
    # else:
    #     video_path = ''
    video_path = 'v_'+str(qa_json['Tweet_ID'])+'_'
    if(len(qa_json['TweetVideoURLs'])>1):
        for vdi in qa_json['TweetVideoURLs']:
            video_path += vdi.split('.mp4')[0].split('/')[-1]+'.mp4'+'_'
        video_path = video_path[:-1]
        video_path += '.avi'
    else:
        video_path += qa_json['TweetVideoURLs'][0].split('.mp4')[0].split('/')[-1]+'.mp4'
    video_path = dst_video_dir+video_path
    if(not os.path.exists(video_path) or os.stat(video_path).st_size==0):
        print("Video not found or video size Nil for removed tweet post:",video_path, qa_path)
        continue
    # print(video_path, qa_path)
    
    for each_qa_dict_itr in range(len(qa_json['QAs'])):
        if(pred_answer_key not in qa_json['QAs'][each_qa_dict_itr]):
            if(qa_json['QAs'][each_qa_dict_itr]['QA-subtask'] is None or qa_json['QAs'][each_qa_dict_itr]['QA-subtask']=='None'):
                template_qa_idx = -1
            else:
                template_qa_idx = qa_json['QAs'][each_qa_dict_itr]['QA-subtask']
            our_prompt_with_question = suitable_prompt(template_qa_idx, qa_json['QAs'][each_qa_dict_itr]['Q'])
    
            ######
            prompt_question, prompt_message = video_llm.get_prompt_question(our_prompt_with_question, video_path)
            # qa_json['QAs'][each_qa_dict_itr][prompt_answer_key] = json.dumps(prompt_message)
            ######
            qa_json['QAs'][each_qa_dict_itr][pred_answer_key] = video_llm.proccess_prompt_question(prompt_question, prompt_message)
            # print(qa_json['QAs'][each_qa_dict_itr][pred_answer_key])
            with open(qa_path,'w') as fp:
                json.dump(qa_json, fp, indent=4)
    # break
del video_llm
torch.cuda.empty_cache()
print("#### VideoLLM prediction complete. All predictions stored in this key:",pred_answer_key)
