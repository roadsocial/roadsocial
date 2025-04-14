import argparse
parser = argparse.ArgumentParser(description="Input arguments.")
parser.add_argument("--qas_dir", type=str, help="QAs dir")
parser.add_argument("--model_prefix", type=str, help="model prefix")
parser.add_argument("--model_size", type=str, help="model param size")
# parser.add_argument("gpu_id", type=str, help="gpu id")
args = parser.parse_args()

YOUR_API_KEY = "<YOUR_API_KEY_here>"

### QAs_DIR
OUTPUT_QA_FOLDER = args.qas_dir
if(OUTPUT_QA_FOLDER[-1]=='/'):
    OUTPUT_QA_FOLDER = OUTPUT_QA_FOLDER[:-1]
OUTPUT_QA_FOLDER += args.model_prefix+'Inferred/'
###
pred_answer_key = 'pred_'+args.model_prefix+args.model_size

##################### CONFIG PARAMS #####################
import os
# if(args.gpu_id!='-1'):
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# import torch
# from gemeval import *
from llms_for_eval.gpteval import *
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from glob import glob
from temporal_gnd import create_pred_and_gt_actionformer, ANETdetection

# def get_temporal_grounding_mAP(gt_answer, pred_answer):
#     ### mAP score eval
#     return 0.0

TG_dict = {}
def get_temporal_grounding_mAP(TG_dict):
    pred_actionformer_dict, gt_actionformer_dict = create_pred_and_gt_actionformer(TG_dict)
    json.dump(gt_actionformer_dict, open("gt_actionformer.json", 'w')) # this is just to debug
    json.dump(pred_actionformer_dict, open("pred_actionformer.json", 'w')) # this is just to debug
    '''
    # CORRECT FORMAT - GT in ActionFormer format
    {
        "version": "Thumos14-30fps",
        "database": {
            "1579042565822693381": {
                "subset": "Test",
                "duration": null,
                "fps": null,
                "annotations": [
                    {
                        "label": "RoadEvent",
                        "segment": [
                            0,
                            12
                        ],
                        "label_id": 1
                    }
                ]
            },
            "1750151226946465887": {
                "subset": "Test",
                "duration": null,
                "fps": null,
                "annotations": [
                    {
                        "label": "RoadEvent",
                        "segment": [
                            3,
                            12
                        ],
                        "label_id": 1
                    }
                ]
            }
        }
    }
    '''

    '''
    # CORRECT FORMAT - PRED in ActionFormer format
    {
        "database": {
            "1579042565822693381": [
                {
                    "segment": [
                        0,
                        0
                    ],
                    "label_id": 1,
                    "scores": 1.0
                }
            ],
            "1750151226946465887": [
                {
                    "segment": [
                        7,
                        10
                    ],
                    "label_id": 1,
                    "scores": 1.0
                }
            ]
        }
    }
    '''

    det_eval = ANETdetection(
        gt_actionformer_dict, # RETURN GT JSON LOCATION: ./data/thumos/annotations/thumos14.json
        "test", # RETURN STRING "test"
        tiou_thresholds = [0.3,0.4, 0.5, 0.6, 0.7] # RETURN A LIST: [0.3 0.4 0.5 0.6 0.7]
    )

    average_mAP = det_eval.evaluate(pred_actionformer_dict)
    return average_mAP

###### Predicted QAs evaluation
allqas_src = glob(OUTPUT_QA_FOLDER+'/*.json')
llm_eval = LLMEval(YOUR_API_KEY)
for qa_path in tqdm(allqas_src):
    qa_json = json.load(open(qa_path, 'r'))
    for each_qa_dict_itr in range(len(qa_json['QAs'])):
        # if(pred_answer_key in qa_json['QAs'][each_qa_dict_itr]):
        if(pred_answer_key in qa_json['QAs'][each_qa_dict_itr] and pred_answer_key+'_score' not in qa_json['QAs'][each_qa_dict_itr]):
            qa_eval_score, eval_explanation = llm_eval.forward([qa_json['QAs'][each_qa_dict_itr]['Q'],   qa_json['QAs'][each_qa_dict_itr]['A'],   qa_json['QAs'][each_qa_dict_itr][pred_answer_key]])
            # print(qa_eval_score, eval_explanation)
            if(eval_explanation!='' and qa_eval_score!=0):
                qa_json['QAs'][each_qa_dict_itr][pred_answer_key+'_score'] = qa_eval_score
                qa_json['QAs'][each_qa_dict_itr][pred_answer_key+'_explanation'] =  eval_explanation
                with open(qa_path,'w') as fp:
                    json.dump(qa_json, fp, indent=4)
del llm_eval
# torch.cuda.empty_cache()
print("#### LLM-eval of predictions complete.")

###### Evaluated QAs aggregation (Simplify this code: code/TweetDrive/LATEST_agg_gpt_eval_scores.ipynb)
all_axes_qs = pd.read_csv('utils/Axes_templateQ_mapping.csv')
unique_tasks = all_axes_qs['Task_names'].unique()
final_result = {}
final_result_counts = {}
for taski in unique_tasks:
    final_result[taski] = 0.0
    final_result_counts[taski] = 0
del final_result_counts['UNK']
del final_result['UNK']

predicted_scores_flag = False
generic = []
specific = []
for qa_path in tqdm(allqas_src):
    qa_json = json.load(open(qa_path, 'r'))
    for each_qa_dict_itr in range(len(qa_json['QAs'])):
        if(pred_answer_key+'_score' in qa_json['QAs'][each_qa_dict_itr]):
            predicted_scores_flag = True
            qa_eval_score, eval_explanation = (qa_json['QAs'][each_qa_dict_itr][pred_answer_key+'_score'], qa_json['QAs'][each_qa_dict_itr][pred_answer_key+'_explanation'])
            # print(qa_eval_score, eval_explanation)
            qa_task_key = qa_json['QAs'][each_qa_dict_itr]['QA-task']
            qa_subtask_key = qa_json['QAs'][each_qa_dict_itr]['QA-subtask']
            if(eval_explanation!='' and qa_eval_score!=0 and qa_task_key!='Grounding'):
                final_result[qa_task_key] += qa_eval_score
                final_result_counts[qa_task_key] += 1
                #### the last condition although used for main paper, should not be used for camera ready check (why it was kept earlier???)
                q_type = qa_json['QAs'][each_qa_dict_itr]['Q-type']
                if(qa_task_key not in ['Where','Viewpoint','Grounding','Adversarial','Incompatible']): # and qa_subtask_key!='RoadEvent_Type'):
                    if(q_type=='Generic'):
                        generic.append(qa_eval_score)
                    else:
                        specific.append(qa_eval_score)
                        
            # elif(qa_task_key=='Grounding'):
            #     ### get qa_eval_score from mAP score evaluation code
            #     gt_a, pred_a = (qa_json['QAs'][each_qa_dict_itr]['A'], qa_json['QAs'][each_qa_dict_itr][pred_answer_key])
            #     qa_eval_score = get_temporal_grounding_mAP(gt_a, pred_a)
            #     final_result[qa_task_key] += qa_eval_score
            #     final_result_counts[qa_task_key] += 1
            #     pass

            filename, gt_a, pred_a = (os.path.basename(qa_path), qa_json['QAs'][each_qa_dict_itr]['A'], qa_json['QAs'][each_qa_dict_itr][pred_answer_key])   
            if str(filename) not in TG_dict.keys():
                TG_dict[str(filename)] = {
                    "gt_a": gt_a,
                    "pred_a": pred_a
                }

json.dump(TG_dict, open("TG_all_qas.json, w")) # debug if it's correct format
'''
# CORRECT FORMAT
{
    "qa_1579042565822693381.json": {
        "Q": "Can you specify the approximate time interval where the key road event is observed in the video? (The time interval should be specified in the format: xx to yy seconds)",
        "A": "The key road event is observed between 0 to 12 seconds.",
        "pred_GPT-4.o-B": "I can't determine the exact time interval from the frames provided. If you can describe the key road event, I might be able to help further."
    },
    "qa_1750151226946465887.json": {
        "Q": "Can you specify approximate time intervals where the key road event is observed in the video? (The time intervals should be specified in the format: xx to yy seconds, mm to nn seconds, etc.)",
        "A": "The key road event is observed between 3 to 12 seconds and 19 to 29 seconds.",
        "pred_GPT-4.o-B": "The key road event, which appears to be a collision, can be observed approximately during the following time intervals:\n\n- 7 to 10 seconds\n- 22 to 25 seconds\n\nThese intervals capture the moments leading up to and during the incident."
    } ...
}
'''

# if(qa_task_key=='Grounding'):
all_qas_average_mAP = get_temporal_grounding_mAP(TG_dict)

######## Overall scores aggregation
all_score = []
fci_score = []
for qa_task_key in unique_tasks:
    if(qa_task_key not in final_result_counts):
        continue
    if(final_result_counts[qa_task_key]!=0):
        final_result[qa_task_key] = final_result[qa_task_key]/final_result_counts[qa_task_key]
        all_score.append( final_result[qa_task_key] )
        if(qa_task_key!='Incompatible'):
            fci_score.append( final_result[qa_task_key] )
final_result['ALL'] = float(np.mean(all_score))
final_result['RT'] = float(np.mean(fci_score))
final_result['Generic'] = float(np.mean(generic))
final_result['Specific'] = float(np.mean(specific))
final_result['Grounding'] = all_qas_average_mAP

final_result = pd.DataFrame(final_result,index=[0]).round(1)
final_result = final_result.loc[:, ['Where', 'Key Entities', 'Viewpoint', 'Description', 'Why', 'Consequence', 'Grounding', 'Advisory', 'Introspection', 'Counterfactual', 'Adversarial', 'Incompatible', 'ALL', 'RT', 'Generic', 'Specific']]

os.makedirs('output/',exist_ok=True)
agg_scores_csv_out = 'output/'+pred_answer_key.replace('pred_','')+'_on_roadsocial_tasks_aggregated_llmevalscores.csv'
final_result.to_csv(agg_scores_csv_out,index=None)

if(not predicted_scores_flag):
    print("ERROR in Score Aggregation: '"+pred_answer_key+"' keys are either not present in any of the output JSONs or their llm-eval scores: '"+pred_answer_key+"_score' keys are not present.")
else:
    print("#### LLM-eval prediction aggregations complete. Output saved in:",agg_scores_csv_out)
