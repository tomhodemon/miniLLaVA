import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    CLIPImageProcessor
)

class ImageProcessor():
    def __init__(self, vision_encoder_path):
        
        self._processor = CLIPImageProcessor.from_pretrained(vision_encoder_path)

    def __call__(self, image: torch.Tensor):
        return self._processor.preprocess(image, return_tensors='pt')['pixel_values']
    

class PromptEngine():
    def __init__(self, processor, prompt_template):
        
        self._processor = processor
        self.prompt_template = prompt_template

    def __call__(self, visual_tokens, text):
        return self._processor.apply_template(self.prompt_template, visual_tokens, text)
    

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


class LLaVAModel(nn.Module):
    def __init__(self, vision_encoder_path, lm_path, prompt_template):
        super().__init__()     

        self.vision_encoder = VisionEncoder(vision_encoder_path)
        self.language_model = LanguageModel(lm_path)
        self.projector = Projector(self.vision_encoder.hidden_size, self.language_model.hidden_size)

        self.image_processor = ImageProcessor(vision_encoder_path)
        self.prompt_engine = PromptEngine(lm_path, prompt_template)

    def forward(self, conversation):
        img, text = conversation

        x = self.image_processor(img) # (bsz, 3, 224, 224) -> (bsz, 3, 224, 224)
        x = self.vision_encoder(x) # (bsz, 3, 224, 224) -> (bsz, num_patches, vision_encoder.hidden_size)
        visual_tokens = self.projector(x) # (bsz, num_patches, vision_encoder.hidden_size) -> (bsz, num_patches, language_model.hidden_size)

        text_prompt = self.prompt_engine(visual_tokens, text)

        out = self.language_model(text_prompt)
        return out

def dummy_data(bsz = 1, img_size = 224):
    dummy_img = torch.randn((bsz, 3, img_size, img_size), dtype=torch.float32).uniform_(0, 1)
    dummy_text = "hello, how are you?"
    return dummy_img, dummy_text


if __name__ == "__main__":

    prompt_template = "USER: <image>\n<prompt> ASSISTANT:"
    lm_path = "Qwen/Qwen2-0.5B"
    vision_encoder_path = "openai/clip-vit-large-patch14"

    model = LLaVAModel(vision_encoder_path, lm_path, prompt_template)

    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

    # text = "Hello, how are you?"

    # inputs = tokenizer(text, return_tensors="pt")

    # print(inputs)
    

    # x = torch.randn(1, 3, 224, 224)

    # vision_encoder = VisionEncoder()
    # x = vision_encoder(x) # (1, 257, 1024)

    # language_hidden_size = 768

    # projector = Projector(vision_encoder.hidden_size, language_hidden_size)


    # x = projector(x)

    # print(x.shape)

    conversation = dummy_data()

    out = model(conversation)
    print(out.logits)
    print(out.logits.shape)
