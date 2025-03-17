### Download videos
from glob import glob
from tqdm import tqdm
import os, json
import argparse
parser = argparse.ArgumentParser(description="Input arguments.")
parser.add_argument("--qas_dir", type=str, help="QAs dir")
args = parser.parse_args()
# args = parser.parse_args(['git/RoadSocial/sample/'])

sample_dir = args.qas_dir
alljsons = glob(sample_dir+'*.json')
if(sample_dir[-1]=='/'):
    sample_dir = sample_dir[:-1]
outvidsdir = sample_dir+'_videos/'
os.makedirs(outvidsdir,exist_ok=True)

for qa_path in tqdm(alljsons):
    qa_json = json.load(open(qa_path, 'r'))
    if(True):
    # if(len(qa_json['TweetVideoURLs'])==1):
        twvidurl = qa_json['TweetVideoURLs'][0]
        video_nm = twvidurl.split('.mp4')[0]+'.mp4'
        out_video_nm = outvidsdir+'v_'+str(qa_json['Tweet_ID'])+'_'+video_nm.split('/')[-1]
        if(not os.path.exists(out_video_nm)):
            os.system('wget -P '+outvidsdir+' -O '+out_video_nm+' '+video_nm)
        # print('wget -P '+outvidsdir+' -O '+out_video_nm+' '+video_nm)
    elif(len(qa_json['TweetVideoURLs'])>1):
        out_video_nm = outvidsdir+'v_'+str(qa_json['Tweet_ID'])
        allvidspth = []
        for twvidurl in qa_json['TweetVideoURLs']:
            video_nm = twvidurl.split('.mp4')[0]+'.mp4'
            os.system('wget -P '+outvidsdir+' '+video_nm)
            allvidspth.append(outvidsdir+video_nm)
            out_video_nm += twvidurl.split('.mp4')[0].split('/')[-1]+'.mp4'+'_'
        out_video_nm = out_video_nm[:-1]
        out_video_nm += '.avi'
        print(out_video_nm)
        ### combine all videos in 'allvidspth' to produce 'out_video_nm'
        pass
    else:
        print("No video found!")