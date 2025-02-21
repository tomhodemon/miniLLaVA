from transformers import (AutoProcessor, CLIPImageProcessor)
import numpy as np

class Processor:
    def __init__(self, vision_encoder_path: str, lm_path: str):

        self.image_processor = CLIPImageProcessor.from_pretrained(vision_encoder_path)
        self.tokenizer = AutoProcessor.from_pretrained(lm_path)        

    def process_image(self, image: np.ndarray):
        return self.image_processor(image)

    def process_text(self, text: str):
        return self.tokenizer(text)
    
#"USER: <image>\n<prompt> ASSISTANT:"

# conversation = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
#             {"type": "text", "text": "What is shown in this image?"},
#         ],
#     },
# ]

# processor = Processor()

# from PIL import Image

# image = Image.open("./australia.jpg")
# img = processor.process_image(image)
# print(img)

# text = "What is shown in this image?"
# tokens = processor.process_text(text)
# print(tokens)
