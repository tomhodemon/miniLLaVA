import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    LlavaProcessor,
    CLIPImageProcessor
)
from PIL import Image
from typing import List, Dict
from prompt_engine import PromptEngine


class ImageProcessor():
    """
    Preprocesses images for the vision encoder. Preprocessing includes croping, resizing and normalizing.
    """
    def __init__(self, 
                 vision_encoder_path, 
                 device = "cuda"
                 ) -> None:
        
        self._processor = CLIPImageProcessor.from_pretrained(vision_encoder_path, use_fast=False, device=device)

    def __call__(self, 
                 image: Image.Image,
                 ) -> torch.Tensor:
        """
        Preprocesses an image for the vision encoder.

        Example:
            Image.Image -> (3, 224, 224)
        """
        return self._processor.preprocess(image, return_tensors='pt')['pixel_values']
    

class VisionEncoder(nn.Module):
    def __init__(self, vision_encoder_path):
        super().__init__()

        self._encoder = CLIPVisionModel.from_pretrained(vision_encoder_path)
        
    @property
    def hidden_size(self):
        return self._encoder.config.hidden_size

    @torch.no_grad()
    def forward(self, x):
        outputs = self._encoder(x)
        return outputs.last_hidden_state[:, 1:] # only keep patch embeddings, remove CLS token
    

class Projector(nn.Module):
    def __init__(self, 
                 image_hidden_size: int, 
                 language_hidden_size: int):
        super().__init__()

        self._proj = nn.Linear(image_hidden_size, language_hidden_size)

    def forward(self, x):
        return self._proj(x)
    

class LanguageModel(nn.Module):
    def __init__(self, lm_path):
        super().__init__()

        self._lm = AutoModelForCausalLM.from_pretrained(lm_path)

    @property
    def hidden_size(self):
        return self._lm.config.hidden_size
    
    def forward(self, x):
        return self._lm(**x)


class LLaVAModel:
    def __init__(self, 
                 vision_encoder_path, 
                 lm_path,
                 device = "cuda"):

        self.vision_encoder = VisionEncoder(vision_encoder_path)
        self.language_model = LanguageModel(lm_path)
        self.projector = Projector(self.vision_encoder.hidden_size, self.language_model.hidden_size)


        self._image_processor = CLIPImageProcessor.from_pretrained(vision_encoder_path, use_fast=False, device=device)
        self._tokenizer = AutoTokenizer.from_pretrained(lm_path)

        self.processor = LlavaProcessor(self._image_processor, self._tokenizer)


def dummy_data(bsz = 1, img_size = 224):
    dummy_img = torch.randn((bsz, 3, img_size, img_size), dtype=torch.float32).uniform_(0, 1)
    dummy_text = "hello, how are you?"
    return dummy_img, dummy_text


if __name__ == "__main__":

    prompt_template = "USER: <image>\n<prompt> ASSISTANT:"
    lm_path = "Qwen/Qwen2-0.5B"
    vision_encoder_path = "openai/clip-vit-large-patch14"

    model = LLaVAModel(vision_encoder_path, lm_path)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]

    inputs = model.processor.apply_chat_template(
        conversation,
        chat_template=prompt_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    print(inputs)

    print(model.processor.tokenizer.decode(inputs['input_ids'][0]))
