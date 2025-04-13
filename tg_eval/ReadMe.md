# Temporal Grounding (TG) Evaluation

## Task
The goal of this task is to evaluate Temporal Grounding (TG) Question-Answering (QA) by extracting time intervals from model-generated responses and comparing them with ground-truth time interval ranges. The evaluation is performed using the mean Average Precision (mAP) metric.

We utilize the official repository of [ActionFormer](https://github.com/happyharrycn/actionformer_release) for this task.

## Requirements
The dependencies are the same as those specified in the official [ActionFormer](https://github.com/happyharrycn/actionformer_release) repository. Also, download the pretrained weights from the official repository of ActionFormer.

## Steps to Compute mAP

### Step 1: Extract Time Intervals from Responses
1. Store each model's generated response and ground-truth response (JSON files) in the folder `./TG_responses_for_all_models/`.
2. The ground-truth and predicted VQA responses should follow the structure of the sample files in `./TG_responses_for_all_models/`.
3. Use the functions in `extract_time_in_seconds.ipynb` to extract time intervals from the responses.
4. Store the extracted outputs in `./TG_converted_to_seconds_for_all_models/`.
5. Sample output files are available in this folder for reference.

### Step 2: Convert JSON Files to ActionFormer Format
1. Convert each JSON file from `./TG_converted_to_seconds_for_all_models/` into the format expected by ActionFormer.
2. Store the converted JSON files in `./TG_converted_to_seconds_for_all_models_thumosFormat/`.
3. Each model will have two JSON files:
   - `<model>_GT_thumosFormat.json` (Ground Truth)
   - `<model>_PRED_thumosFormat.json` (Predictions)
4. Run the cells in `create_thumosFormat.ipynb` to generate these formatted JSON files.
5. After successful conversion, the folder structure will be as follows:

   ```
   tg_eval/
   │   README.md
   │   ...
   │
   ├── TG_converted_to_seconds_for_all_models/
   │    ├── <model1>_all_TG_QAs_gt_preds.json
   │    ├── <model2>_all_TG_QAs_gt_preds.json
   │    ...
   │
   ├── TG_converted_to_seconds_for_all_models_thumosFormat/
   │    ├── <model1>_all_TG_QAs_gt_preds_GT_thumosFormat.json
   │    ├── <model1>_all_TG_QAs_gt_preds_PRED_thumosFormat.json
   │    ├── <model2>_all_TG_QAs_gt_preds_GT_thumosFormat.json
   │    ├── <model2>_all_TG_QAs_gt_preds_PRED_thumosFormat.json
   │    ...
   ```

### Step 3: Run the ActionFormer Evaluation
1. Update the paths to the converted JSON files in the ActionFormer repository.
2. Modify the following files:
   - `eval.py` in `actionformer_release/`
     - Update the argument at **line 131** with the location of `<model>_GT_thumosFormat.json`.
   - `metrics.py` in `actionformer_release/libs/utils/`
     - Update the variable `preds` at **line 283** with the location of `<model>_PRED_thumosFormat.json`.
3. Run the evaluation command from `actionformer_release/`:
   ```bash
   python eval.py ./configs/thumos_i3d.yaml ./pretrained/thumos_i3d_reproduce/
   ```
4. The terminal output will display evaluation results in the following format:
   ```
   ##### Model: GPT-4.0-B
   |tIoU = 0.30: mAP = 18.04 (%) Recall@1x = 40.20 (%) Recall@5x = 40.20 (%)
   |tIoU = 0.40: mAP = 11.64 (%) Recall@1x = 31.50 (%) Recall@5x = 31.50 (%)
   |tIoU = 0.50: mAP = 6.53 (%) Recall@1x = 23.82 (%) Recall@5x = 23.82 (%)
   |tIoU = 0.60: mAP = 2.34 (%) Recall@1x = 14.47 (%) Recall@5x = 14.47 (%)
   |tIoU = 0.70: mAP = 0.83 (%) Recall@1x = 8.71 (%) Recall@5x = 8.71 (%)
   Average mAP: 7.88 (%)
   ```

---
This guide provides step-by-step instructions for evaluating TG QAs using ActionFormer. For any issues, refer to the official ActionFormer documentation.

