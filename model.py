import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self._encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # freeze by default
        for param in self._encoder.parameters():
            param.requires_grad = False

    @property
    def hidden_size(self):
        return self._encoder.config.hidden_size

    def forward(self, x):
        outputs = self._encoder(x)
        return outputs.last_hidden_state
    

class Projector(nn.Module):
    def __init__(self, 
                 image_hidden_size: int, 
                 language_hidden_size: int):
        super().__init__()

        self._proj = nn.Linear(image_hidden_size, language_hidden_size)

    def forward(self, x):
        return self._proj(x)
    

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

    @property
    def hidden_size(self):
        return self._lm.config.hidden_size
    
    def forward(self, x):
        pass



class LLaVAModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision_encoder = VisionEncoder()
        self.language_model = LanguageModel()

        self.projector = Projector(self.vision_encoder.hidden_size, self.language_model.hidden_size)
        

    def forward(self, x):
        pass



if __name__ == "__main__":

    model = LLaVAModel()
    # x = torch.randn(1, 3, 224, 224)

    # vision_encoder = VisionEncoder()
    # x = vision_encoder(x) # (1, 257, 1024)

    # language_hidden_size = 768

    # projector = Projector(vision_encoder.hidden_size, language_hidden_size)


    # x = projector(x)

    # print(x.shape)


