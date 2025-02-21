from transformers import AutoTokenizer

class PromptEngine():
    """
    Injects visual placeholder and performs embedding replacement.
    """
    def __init__(self, lm_path, prompt_template):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_path)
        self.prompt_template = prompt_template
        
        # Special tokens for image placeholders
        self.image_token = "<image>"
        self.default_image_token = "<image>"
        self._image_tokens = ["<image>"]
        
    def _get_image_token_span(self, prompt):
        """Find the span of the image token in the prompt"""
        pattern_start = prompt.find(self.image_token)
        if pattern_start == -1:
            return None
        return pattern_start, pattern_start + len(self.image_token)
    
    def __call__(self, visual_tokens, text):
        """
        Args:
            visual_tokens: Tensor of shape (batch_size, n_patches, hidden_size)
            text: String containing the conversation text
        Returns:
            Dict containing input_ids and attention_mask ready for the language model
        """
        # Format conversation messages
        messages = [{"role": "user", "content": text}]
        
        # Apply chat template to format the conversation
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        )
        
        # Find where to insert image tokens
        image_span = self._get_image_token_span(self.prompt_template)
        if image_span is None:
            raise ValueError(f"Prompt template must contain {self.image_token}")
            
        # Get token indices for image placeholder
        token_spans = []
        for idx, (start, end) in enumerate(prompt.encodings[0].offsets):
            if start is None or end is None:
                continue
            if start >= image_span[0] and end <= image_span[1]:
                token_spans.append(idx)
                
        if not token_spans:
            raise ValueError("Could not find image token span in tokenized prompt")
            
        # Get input tensors
        input_ids = prompt.input_ids
        attention_mask = prompt.attention_mask
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_token_spans": token_spans,
            "visual_tokens": visual_tokens
        }