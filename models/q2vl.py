import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class VideoLLM:
    def __init__(self, movel_v):
        """Initialize the VideoLLM with model configuration and required components."""
        self.MODEL_VERSION = movel_v #'7'
        self.SAMPLE_FPS = 2.0
        # self.MAX_DURATION = 4.0
        
        # Initialize model and processor
        self.pretrained = f"Qwen/Qwen2-VL-{self.MODEL_VERSION}B-Instruct"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.pretrained,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.pretrained)

    # def get_prompt_quesiton(our_prompt_with_question, **xargs):
    # video_path, video_duration = xargs['video_path'], xargs['video_duration']
    def get_prompt_question(self, our_prompt_with_question, video_path):
        """
        Prepare the prompt question with video and text content.
        
        Args:
            our_prompt_with_question (str): The text prompt/question to process
            video_path (str): Path to the video file
            
        Returns:
            tuple: Processed prompt question and message structure
        """
        sample_message = [{
            "role": "user",
            "content": [{
                "type": "video",
                "video": video_path,
                #self.MAX_DURATION/video_duration if(video_duration > self.MAX_DURATION) else self.SAMPLE_FPS,  ##video_duration (float): Duration of the video in seconds
                "fps": self.SAMPLE_FPS,
            }]
        }]
        
        sample_message[0]['content'].append({
            "type": "text",
            "text": our_prompt_with_question
        })
        
        prompt_question = self.processor.apply_chat_template(
            sample_message,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt_question, sample_message

    def proccess_prompt_question(self, prompt_question_, prompt_message_):
        """
        Process the prompt question and generate a response.
        
        Args:
            prompt_question_ (str): The processed prompt question
            prompt_message_ (list): The message structure containing video and text
            
        Returns:
            str: Generated text response
        """
        image_inputs, video_inputs = process_vision_info([prompt_message_])
        
        inputs = self.processor(
            text=[prompt_question_],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Generate response
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        all_text_outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return all_text_outputs[0]