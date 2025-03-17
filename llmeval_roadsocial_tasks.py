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

def get_temporal_grounding_mAP(gt_answer, pred_answer):
    ### mAP score eval
    return 0.0

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
                        
            elif(qa_task_key=='Grounding'):
                ### get qa_eval_score from mAP score evaluation code
                gt_a, pred_a = (qa_json['QAs'][each_qa_dict_itr]['A'], qa_json['QAs'][each_qa_dict_itr][pred_answer_key])
                qa_eval_score = get_temporal_grounding_mAP(gt_a, pred_a)
                final_result[qa_task_key] += qa_eval_score
                final_result_counts[qa_task_key] += 1
                pass

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

final_result = pd.DataFrame(final_result,index=[0]).round(1)
final_result = final_result.loc[:, ['Where', 'Key Entities', 'Viewpoint', 'Description', 'Why', 'Consequence', 'Grounding', 'Advisory', 'Introspection', 'Counterfactual', 'Adversarial', 'Incompatible', 'ALL', 'RT', 'Generic', 'Specific']]

os.makedirs('output/',exist_ok=True)
agg_scores_csv_out = 'output/'+pred_answer_key.replace('pred_','')+'_on_roadsocial_tasks_aggregated_llmevalscores.csv'
final_result.to_csv(agg_scores_csv_out,index=None)

if(not predicted_scores_flag):
    print("ERROR in Score Aggregation: '"+pred_answer_key+"' keys are either not present in any of the output JSONs or their llm-eval scores: '"+pred_answer_key+"_score' keys are not present.")
else:
    print("#### LLM-eval prediction aggregations complete. Output saved in:",agg_scores_csv_out)
