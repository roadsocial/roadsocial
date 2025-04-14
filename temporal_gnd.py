'''
Note: ActionFormer computes precision-recall curves and mAP calculations across the entire dataset at once rather than averaging individual results
When we compute mAP for each example individually and then average them as in the above function "get_temporal_grounding_mAP()", we're essentially treating each example as its own separate precision-recall curve. 
ActionFormer instead:
1. Sorts all predictions across the entire dataset by confidence score
2. Computes a single precision-recall curve for all examples together
3. Calculates mAP from this unified curve

This difference in methodology is why our individual calculations didn't match ActionFormer's results.
'''

import json 
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def time_to_seconds(time_str):
    """
    Convert a time string to seconds.
    Handles formats: hh:mm:ss, mm:ss, ss
    Handles decimal numbers by rounding to nearest integer
    """
    try:
        # If it's just a number (seconds)
        if ':' not in time_str:
            # Round to nearest integer for decimal numbers
            return round(float(time_str))
        
        # Split the time string
        parts = time_str.split(':')
        
        if len(parts) == 3:  # hh:mm:ss
            h, m, s = parts
            # Round the seconds part if it contains decimals
            return int(h) * 3600 + int(m) * 60 + round(float(s))
        elif len(parts) == 2:  # mm:ss
            m, s = parts
            # Round the seconds part if it contains decimals
            return int(m) * 60 + round(float(s))
        else:
            return 0
    except (ValueError, TypeError):
        return 0


def validate_time_order(time_list):
    """
    Validate that the second time is greater than the first time.
    If not, return [0, 0].
    """
    if len(time_list) != 2:
        return [0, 0]
    
    if time_list[0] >= time_list[1]:
        return [0, 0]
    
    return time_list


def extract_time_values(text):
    """
    Extract time values from text and return them as a list of integers (seconds).
    """
    import re
    
    # First, try to find time patterns (hh:mm:ss or mm:ss)
    time_pattern = r'\d+:\d+(?::\d+)?'
    time_values = re.findall(time_pattern, text)
    
    # If not enough time values, look for decimal numbers in the text
    if len(time_values) < 2:
        # Look for patterns like "X.Y" or "X" where X and Y are digits
        # This will match both integer and decimal numbers
        decimal_pattern = r'\b\d+(?:\.\d+)?\b'
        
        # Find all matches in the text
        numbers = re.findall(decimal_pattern, text)
        
        # Filter out numbers that are part of time values
        numbers = [num for num in numbers if not any(num in tv for tv in time_values)]
        
        # Convert found decimal numbers to integers (rounded)
        time_values.extend(numbers)
    
    # Convert all values to seconds
    result = []
    for i in range(2):
        if i < len(time_values):
            result.append(time_to_seconds(time_values[i]))
        else:
            result.append(0)
    
    # Validate time order
    return validate_time_order(result)
    
def create_pred_and_gt_actionformer(preds_and_gt):
    pred_actionformer_dict = {"database": {}}
    gt_actionformer_dict = {"database": {}, "version": "Thumos14-30fps"}
    for question in preds_and_gt:

        gt = preds_and_gt[question]['A']
        pred = preds_and_gt[question]['pred_GPT-4.o-B']
        gt_interval = extract_time_values(gt)
        pred_interval = extract_time_values(pred)

        pred_actionformer_dict["database"][question] = [{'segment': pred_interval, 'label_id': 1, 'scores': 1.0}]
        
        gt_actionformer_dict["database"][question] = {
            "subset": "Test",
            "duration": None,
            "fps": None,
            'annotations': 
                [{
                    "segment" : gt_interval, 
                    'label': "RoadEvent", 'label_id': 1
                }
                ]
            
            }

    return pred_actionformer_dict, gt_actionformer_dict

###################################################################################################

# Compute mAP using actionformer/metrics.py
def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate / very short annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event['label_id']
        if (e - s) >= tol:
            valid = True
        else:
            valid = False
        for p_event in valid_events:
            if ((abs(s-p_event['segment'][0]) <= tol)
                and (abs(e-p_event['segment'][1]) <= tol)
                and (l == p_event['label_id'])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events



# Modified from official EPIC-Kitchens action detection evaluation code
# see https://github.com/epic-kitchens/C2-Action-Detection/blob/master/EvaluationCode/evaluate_detection_json_ek100.py
import os
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List
from typing import Tuple
from typing import Dict

####################### BY ME - To save prediction json ############################

def convert_predictions_to_ground_truth(predictions_df: pd.DataFrame, fps: float = 30.0) -> Dict:
    """
    Convert predictions DataFrame to ground truth JSON format.
    
    Args:
        predictions_df: DataFrame with columns [video-id, t-start, t-end, label, score]
        fps: Frames per second (default: 30.0)
        
    Returns:
        Dictionary in ground truth JSON format
    """
    # Initialize the output structure
    output = {
        "version": "Thumos14-30fps",
        "database": {}
    }
    
    # Group predictions by video-id
    grouped = predictions_df.groupby('video-id')
    
    # Process each video
    for video_id, group in grouped:
        # Calculate video duration from max t-end
        duration = group['t-end'].max()
        
        # Initialize video entry
        video_entry = {
            "subset": "Test",
            "duration": float(duration),
            "fps": fps,
            "annotations": []
        }
        
        # Process each prediction for this video
        for _, row in group.iterrows():
            # Convert time segments to frame segments
            frame_start = row['t-start'] * fps
            frame_end = row['t-end'] * fps
            
            annotation = {
                "label": f"Label_{row['label']}", # You might want to map label IDs to actual names
                "segment": [
                    float(row['t-start']),
                    float(row['t-end'])
                ],
                "segment(frames)": [
                    float(frame_start),
                    float(frame_end)
                ],
                "label_id": int(row['label']),
                "score": float(row['score'])  # Additional field from predictions
            }
            
            video_entry["annotations"].append(annotation)
            
        # Add video entry to database
        output["database"][video_id] = video_entry
    
    return output

def save_ground_truth(data: Dict, output_path: str):
    """
    Save the ground truth format data to a JSON file.
    
    Args:
        data: Dictionary in ground truth format
        output_path: Path where to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


##############################################################################


def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate / very short annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event['label_id']
        if (e - s) >= tol:
            valid = True
        else:
            valid = False
        for p_event in valid_events:
            if ((abs(s-p_event['segment'][0]) <= tol)
                and (abs(e-p_event['segment'][1]) <= tol)
                and (l == p_event['label_id'])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events


def load_gt_seg_from_json(json_file, split=None, label='label_id', label_offset=0):
    # load json file
    # with open(json_file, "r", encoding="utf8") as f:
    #     json_db = json.load(f)
    # json_db = json_db['database']

    print(json_file)
    json_db = json_file['database'] # assuming json_file is a dictionary

    vids, starts, stops, labels = [], [], [], []
    for k, v in json_db.items():

        # filter based on split
        if (split is not None) and v['subset'].lower() != split:
            continue
        # remove duplicated instances
        ants = remove_duplicate_annotations(v['annotations'])
        # video id
        vids += [k] * len(ants)
        # for each event, grab the start/end time and label
        for event in ants:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]

    # move to pd dataframe
    gt_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels
    })

    return gt_base


def load_pred_seg_from_json(json_file, label='label_id', label_offset=0):
    # load json file
    # with open(json_file, "r", encoding="utf8") as f:
    #     json_db = json.load(f)
    # json_db = json_db['database']
    
    json_db = json_file['database'] # assuming json_file is a dictionary

    vids, starts, stops, labels, scores = [], [], [], [], []
    for k, v, in json_db.items():
        # video id
        vids += [k] * len(v)
        # for each event
        for event in v:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]
            scores += [float(event['scores'])]

    # move to pd dataframe
    pred_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels,
        'score': scores
    })

    return pred_base


class ANETdetection(object):
    """Adapted from https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_detection.py"""

    def __init__(
        self,
        ant_file, # acccepts dictionary
        split=None,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        top_k=(1, 5),
        label='label_id',
        label_offset=0,
        num_workers=8,
        dataset_name="MYDATASET",
    ):
        self.tiou_thresholds = tiou_thresholds
        self.top_k = top_k
        self.ap = None
        self.num_workers = num_workers
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = os.path.basename(ant_file).replace('.json', '')

        # Import ground truth and predictions
        self.split = split
        print("works")
        self.ground_truth = load_gt_seg_from_json(
            ant_file, split=self.split, label=label, label_offset=label_offset)
        print("worked")
        # remove labels that does not exists in gt
        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth['label'].unique()))}
        self.ground_truth['label']=self.ground_truth['label'].replace(self.activity_index)

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            res = prediction_by_label.get_group(cidx).reset_index(drop=True)
            return res
        except:
            print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self, preds):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def wrapper_compute_topkx_recall(self, preds):
        """Computes Top-kx recall for each class in the subset.
        """
        recall = np.zeros((len(self.tiou_thresholds), len(self.top_k), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_topkx_recall_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
                top_k=self.top_k,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            recall[...,cidx] = results[i]

        return recall

    def evaluate(self, preds, verbose=True): # preds is a dictionary
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        preds can be (1) a pd.DataFrame; or (2) a json file where the data will be loaded;
        or (3) a python dict item with numpy arrays as the values
        """

        # preds = "/ssd_scratch/deepti/tal_mAP_testing/dummy_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/DolPhin7B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/DolPhin7B_Dashcam_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/roadsocial/tg_eval/TG_converted_to_seconds_for_all_models_thumosFormat/GPT-4.o-B_all_TG_QAs_gt_preds_PRED_thumosFormat_TRY.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/internlm-7B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/InternVL2-Llama3-76B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/llava-ov-finetuned-chck14000-test7B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/llava-ov7B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/llava-ov72B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/LLaVA-Video-7B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/LLaVA-Video-72B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/MiniCPM_8B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/Qwen2-VL-7B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/Qwen2-VL-72B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/tarsier_7B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/tarsier_34B_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/VITA_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/LongVU_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/gemini-1.5-pro-_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"
        # preds="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/Aria_all_TG_QAs_gt_preds_noCoT_PRED_thumosFormat.json"

        # print("PREDS DICT ================ ", preds) # DICT with KEYS: 'video-id', 't-start', 't-end', 'label', 'score'

        # if isinstance(preds, pd.DataFrame):
        #     assert 'label' in preds
        # WE NEED THIS ELIF TO BE TRUE
        # elif isinstance(preds, str):
        #     preds = load_pred_seg_from_json(preds)
        #     print("IS THIS TRUE ???????????????????????")
        # elif isinstance(preds, Dict):
        #     # move to pd dataframe
        #     # did not check dtype here, can accept both numpy / pytorch tensors
        #     preds = pd.DataFrame({
        #         'video-id' : preds['video-id'],
        #         't-start' : preds['t-start'].tolist(),
        #         't-end': preds['t-end'].tolist(),
        #         'label': preds['label'].tolist(),
        #         'score': preds['score'].tolist()
        #     })
        # always reset ap
        preds = load_pred_seg_from_json(preds) # we are passing in a dictionary anyway

        self.ap = None

        # make the label ids consistent
        preds['label'] = preds['label'].replace(self.activity_index)

        # compute mAP
        self.ap = self.wrapper_compute_average_precision(preds)
        self.recall = self.wrapper_compute_topkx_recall(preds)
        mAP = self.ap.mean(axis=1)
        mRecall = self.recall.mean(axis=2)
        average_mAP = mAP.mean()

        # print results
        if verbose:
            # print the results
            print('[RESULTS] Action detection results on {:s}.'.format(
                self.dataset_name)
            )
            block = ''
            for tiou, tiou_mAP, tiou_mRecall in zip(self.tiou_thresholds, mAP, mRecall):
                block += '\n|tIoU = {:.2f}: '.format(tiou)
                block += 'mAP = {:>4.2f} (%) '.format(tiou_mAP*100)
                for idx, k in enumerate(self.top_k):
                    block += 'Recall@{:d}x = {:>4.2f} (%) '.format(k, tiou_mRecall[idx]*100)
            print(block)
            print('Average mAP: {:>4.2f} (%)'.format(average_mAP*100))

        # return the results
        return mAP, average_mAP, mRecall


def compute_average_precision_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """

    # print("PREDICTION: ", prediction)  # BY ME

    ############################# BY ME ######################################

    # Convert predictions to ground truth format
    # ground_truth_format = convert_predictions_to_ground_truth(prediction)
    
    # Save to JSON file
    # save_ground_truth(ground_truth_format, 'eval_results.json')
    
    # Print first video entry as example
    # first_video = list(ground_truth_format['database'].keys())[0]
    # print(f"Example conversion for video {first_video}:")
    # print(json.dumps(ground_truth_format['database'][first_video], indent=4))

    ##########################################################################

    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


def compute_topkx_recall_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5),
    top_k=(1, 5),
):
    """Compute recall (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    top_k: tuple, optional
        Top-kx results of a action category where x stands for the number of 
        instances for the action category in the video.
    Outputs
    -------
    recall : float
        Recall score.
    """
    if prediction.empty:
        return np.zeros((len(tiou_thresholds), len(top_k)))

    # Initialize true positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(top_k)))
    n_gts = 0

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    prediction_gbvn = prediction.groupby('video-id')

    for videoid, _ in ground_truth_gbvn.groups.items():
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        n_gts += len(ground_truth_videoid)
        try:
            prediction_videoid = prediction_gbvn.get_group(videoid)
        except Exception as e:
            continue

        this_gt = ground_truth_videoid.reset_index()
        this_pred = prediction_videoid.reset_index()

        # Sort predictions by decreasing score order.
        score_sort_idx = this_pred['score'].values.argsort()[::-1]
        top_kx_idx = score_sort_idx[:max(top_k) * len(this_gt)]
        tiou_arr = k_segment_iou(this_pred[['t-start', 't-end']].values[top_kx_idx],
                                 this_gt[['t-start', 't-end']].values)
            
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for kidx, k in enumerate(top_k):
                tiou = tiou_arr[:k * len(this_gt)]
                tp[tidx, kidx] += ((tiou >= tiou_thr).sum(axis=0) > 0).sum()

    recall = tp / n_gts

    return recall


def k_segment_iou(target_segments, candidate_segments):
    return np.stack(
        [segment_iou(target_segment, candidate_segments) \
            for target_segment in target_segments]
    )


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap