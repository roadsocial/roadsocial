```
git clone https://github.com/roadsocial/roadsocial.git
cd roadsocial
sudo apt-get install ffmpeg libsm6 libxext6 

### tested on pytorch 2.5.0 for cuda 12.4 in python 3.10
conda create -n roadsocial python=3.10
conda activate roadsocial
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia

python3 -m pip install -r requirements.txt

### qwen2vl dependencies
python3 -m pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate==1.0.1
python3 -m pip install qwen-vl-utils[decord]
python3 -m pip install flash-attn==2.1.0 --no-build-isolation

### llavaov dependencies
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
git checkout 00b5b84ce8675f62eb7bb4587810366ab3770613
### Note about patch: Original llavaov was tested on pytorch==2.1.2, For compatibility with qwen2vl pytorch version is not altered in this installation
git apply ../llavaov_pyproject.toml.patch
pip install -e ".[train]"
pip install --upgrade httpx
cd ..

python3 -m pip install huggingface_hub
### gateway login for accessing RoadSocial dataset
huggingface-cli login
mkdir data
cd data
git clone https://huggingface.co/datasets/chiragp26/RoadSocial
cd ..

python3 download_videos.py --qas_dir 'data/RoadSocial/sample/'
python3 infer_on_roadsocial.py --qas_dir 'data/RoadSocial/sample/' --model_prefix 'Qwen2-VL-' --model_size 2 --gpu_id 2
python3 infer_on_roadsocial.py --qas_dir 'data/RoadSocial/sample/' --model_prefix 'llava-ov' --model_size 0.5 --gpu_id 2
python3 llmeval_roadsocial_tasks.py --qas_dir 'data/RoadSocial/sample/' --model_prefix 'Qwen2-VL-' --model_size 2
python3 llmeval_roadsocial_tasks.py --qas_dir 'data/RoadSocial/sample/' --model_prefix 'llava-ov' --model_size 0.5
```
