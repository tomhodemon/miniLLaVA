from torch.utils.data import Dataset
import json
import pathlib

class LLaVAInstruct(Dataset):
    """
    A dataset for the LLaVA Instruct dataset. It returns a dict with the following keys:
    - "id": The id of the conversation.
    - "image": The image file name.
    - "conversation": A list of dicts with the following keys:
        - "role": The role of the speaker (always "human" for this dataset)
        - "content": The content of the message.
    """
    def __init__(self, 
                 data_path: str,
                 image_dir: str,
                 ) -> None:
    
        self.data = json.load(open(data_path, "r"))
        self.image_dir = pathlib.Path(image_dir)

        print(f"Loaded {len(self.data)} conversation samples from {data_path}")
        print(f"Image directory: {self.image_dir}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        conversation = sample["conversations"]
        image_path = self.image_dir / sample["image"]
        return {
            "id": sample["id"],
            "image": image_path,
            "conversation": conversation
        }
    

if __name__ == "__main__":
    # For testing
    dataset = LLaVAInstruct(data_path= "./datasets/complex_reasoning_77k.json")
    print(len(dataset))
    print(dataset[1])
