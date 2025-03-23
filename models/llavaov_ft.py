import torch
from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import cv2, copy, numpy as np
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")

class VideoLLM:
    def __init__(self, movel_v):
        """Initialize the VideoLLM with model configuration and required components."""
        self.MODEL_VERSION = movel_v #'7'
        self.max_frames_num = 32 #16
        self.conv_template = "qwen_1_5"
        self.pretrained = movel_v
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(self.pretrained, None, 'llava_qwen', device_map='auto', overwrite_config={'tie_word_embeddings': False, 'use_cache': True, "vocab_size": 152064})
        self.model.eval()

    
    def get_prompt_question(self, our_prompt_with_question, video_path):
        """
        Prepare the prompt question with video and text content.
        
        Args:
            our_prompt_with_question (str): The text prompt/question to process
            video_path (str): Path to the video file
            
        Returns:
            tuple: Processed prompt question and message structure
        """
        conv = copy.deepcopy(conv_templates[self.conv_template])
        
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{our_prompt_with_question}")
        ### Assitant Output Response/Answer-1
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        sample_message=video_path
        
        return prompt_question, sample_message

    ## Function to extract frames from video
    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)
    
    def proccess_prompt_question(self, prompt_question_, prompt_message_):
        """
        Process the prompt question and generate a response.
        
        Args:
            prompt_question_ (str): The processed prompt question
            prompt_message_ (list): The message structure containing video and text
            
        Returns:
            str: Generated text response
        """
        input_ids = tokenizer_image_token(prompt_question_, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
        
        video_path = prompt_message_
        video_frames = self.load_video(video_path, self.max_frames_num)
        frames = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(dtype=torch.float16).cuda()
        image_tensors = [frames]
        image_sizes = [frame.size for frame in video_frames]        

        ### Generate response
        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
        all_text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

        return all_text_outputs[0]