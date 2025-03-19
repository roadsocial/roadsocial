# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        # map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
        map_location = lambda storage, loc: storage.cuda(int(cfg['devices'][0].split(':')[1]))
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint


    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            args.gt_file, # RETURN GT JSON LOCATION: ./data/thumos/annotations/thumos14.json
            val_dataset.split[0], # RETURN STRING "test"
            tiou_thresholds = val_db_vars['tiou_thresholds'] # RETURN A LIST: [0.3 0.4 0.5 0.6 0.7]
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    # print("val_loader ======= ", val_loader) 

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    mAP = valid_one_epoch(
        val_loader, # <torch.utils.data.dataloader.DataLoader object at 0x7fe7d68d17f0> as we are not passing the argument "saveonly"
        model,
        -1,
        evaluator=det_eval, # CONTAINS AN INSTANCE OF "ANETdetection" <libs.utils.metrics.ANETdetection object at 0x7fe98e739e50> as we are not passing the argument "saveonly"
        output_file=output_file, # "None" as we are not passing the argument "saveonly"
        ext_score_file=cfg['test_cfg']['ext_score_file'], # "None"
        tb_writer=None,
        print_freq=args.print_freq # 10
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    
    parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/dummy_GT_thumosFormat.json", help="Path to prediction file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/DolPhin7B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/DolPhin7B_Dashcam_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/GPT-4.o-B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/internlm-7B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/InternVL2-Llama3-76B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/llava-ov-finetuned-chck14000-test7B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/llava-ov7B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/llava-ov72B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/LLaVA-Video-7B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/LLaVA-Video-72B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/MiniCPM_8B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/Qwen2-VL-7B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/Qwen2-VL-72B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/tarsier_7B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/tarsier_34B_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/VITA_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/LongVU_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/gemini-1.5-pro-_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="/ssd_scratch/deepti/TG_converted_to_seconds_for_all_models_thumosFormat/Aria_all_TG_QAs_gt_preds_noCoT_GT_thumosFormat.json", help="Path to gt file")
    # parser.add_argument("gt_file", nargs="?", default="", help="Path to gt file")
    args = parser.parse_args()
    main(args)
