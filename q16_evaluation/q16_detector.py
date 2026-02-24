import torch
import os
from PIL import Image
from .q16_classifier import ClipWrapper, SimClassifier, compute_embeddings, load_prompts

class Q16Detector:
    """Q16 Violence Detection Classifier"""
    
    def __init__(self, device=None, prompts_path=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if prompts_path is None:
            self.prompts_path = "q16_evaluation/prompts.p"
        else:
            self.prompts_path = prompts_path
            
        self.clip_model = None
        self.classifier = None
        self._initialize_models()
    
    def _initialize_models(self):
        trained_prompts = load_prompts(self.prompts_path, device=self.device)
        
        self.clip_model = ClipWrapper(self.device)
        self.classifier = SimClassifier(trained_prompts, self.device)
        self.classifier.to(self.device)
    
    def is_available(self):
        return self.clip_model is not None and self.classifier is not None
    
    def detect_harm(self, image_path):
        if not self.is_available():
            return {
                'is_harmful': False,
                'label': 0,
                'score': 0.0,
                'logits': [1.0, 0.0],
                'error': ''
            }
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        
        x = compute_embeddings(self.clip_model, image, self.device)
        y = self.classifier(x)
        label = torch.argmax(y, dim=0).item()
        logits = torch.softmax(y, dim=-1).squeeze()
        score = logits[1].item() if len(logits) > 1 else 0.0
        
        return {
            'is_harmful': label == 1,
            'label': label,
            'score': score,
            'logits': logits.tolist() if isinstance(logits, torch.Tensor) else [logits]
        }
    
    def get_detection_score(self, image_path):
        result = self.detect_harm(image_path)
        return {
            'harmful_score': result['score'],
            'benign_score': 1.0 - result['score'] if 'error' not in result else 1.0,
            'is_harmful': result['is_harmful'],
            'label': result['label']
        } 