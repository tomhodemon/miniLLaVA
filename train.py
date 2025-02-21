import torch
from data import LLaVAInstruct
from model import LLaVAModel
from PIL import Image


def train_step(model, batch):
    # Get image and text from conversation
    image_path = batch["image"]
    text = batch["conversation"]
    
    # Process image
    image = Image.open(image_path)
    image_tensor = model.image_processor(image)
    
    # Get visual features
    visual_features = model.vision_encoder(image_tensor)
    visual_tokens = model.projector(visual_features)
    
    # Process text and combine with visual tokens
    model_inputs = model._processor(images=image_tensor, text=text, return_tensors="pt")
    
    # Forward pass through language model
    outputs = model.language_model(model_inputs)
    return outputs

def main():
    # Model configuration
    vision_encoder_path = "openai/clip-vit-large-patch14"
    lm_path = "Qwen/Qwen2-0.5B"
    prompt_template = "USER: <image>\n<prompt> ASSISTANT:"
    
    # Initialize model
    model = LLaVAModel(
        vision_encoder_path=vision_encoder_path,
        lm_path=lm_path,
        prompt_template=prompt_template
    )
    
    # Load dataset
    dataset = LLaVAInstruct(data_path="./datasets/complex_reasoning_77k.json", 
                            image_dir="/home/thodemon/workspace/datasets/coco/train2017")
    
    # Training loop
    for idx in range(min(1, len(dataset))):
        batch = dataset[idx]
        outputs = train_step(model, batch)
        
        print(f"Processed sample {idx}")
        print(f"Output logits shape: {outputs.logits.shape}")
        print("-" * 50)

if __name__ == "__main__":
    main()

    

    
