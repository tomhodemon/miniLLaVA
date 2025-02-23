import json
import os
from pathlib import Path
from typing import List
from pydantic import BaseModel
from torch.utils.data import Dataset

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    id: str
    image: str
    conversations: List[Message]

class LLaVAInstruct(Dataset):
    def __init__(self, 
                 data_path: Path, 
                 image_dir: Path
                ) -> None:
        
        self.data = json.load(open(data_path, "r"))
        self.image_dir = image_dir

        print(f"Loaded {len(self.data)} conversation samples from {data_path}")
        print(f"Image directory: {self.image_dir}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Conversation:
        sample = self.data[idx]
        image_path = os.path.join(self.image_dir, sample['image'])
        sample['image'] = image_path
        
        for conv in sample['conversations']:
            conv['role'] = conv.pop('from')
            conv['content'] = conv.pop('value')
        
        return Conversation(**sample)


if __name__ == "__main__":
    # Example usage

    datasets_dir = Path('/Users/thod/workspace/datasets')
    dataset = LLaVAInstruct(
        data_path=datasets_dir / 'llava-instruct/complex_reasoning_77k.json', 
        image_dir=datasets_dir / 'coco/train2017'
    )

    example = dataset[0]
    print(example)