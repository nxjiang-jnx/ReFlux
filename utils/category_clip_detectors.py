import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch.nn.functional as F

CATEGORY_PROMPTS = {
    'entity': [
        'A photo of fruit',
        'A photo of ball', 
        'A photo of car',
        'A photo of airplane',
        'A photo of tower',
        'A photo of building',
        'A photo of celebrity',
        'A photo of shoes',
        'A photo of cat',
        'A photo of dog'
    ],
    'abstraction': [
        'A scene featuring explosion',
        'A scene featuring green bag',
        'A scene featuring yellow bag', 
        'A scene featuring time',
        'A scene featuring two cats',
        'A scene featuring three cats',
        'A scene featuring shadow',
        'A scene featuring smoke',
        'A scene featuring dust',
        'A scene featuring environmental simulation'
    ],
    'relationship': [
        'A shake hand B',
        'A kiss B',
        'A hug B',
        'A in B',
        'A on B',
        'A back to back B',
        'A jump B',
        'A burrow B',
        'A hold B',
        'A amidst B'
    ]
}

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")

def calculate_category_clip_score(image_array: np.ndarray, category: str, device: str = "cuda") -> float:
    category_prompts = CATEGORY_PROMPTS[category]
    scores = []
    
    for prompt in category_prompts:
        score = clip_score_fn(
            torch.from_numpy(image_array[None, ...]).to(device), 
            [prompt]
        ).detach()
        scores.append(float(score))
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return round(avg_score, 4)

def calculate_category_clip_classifier(image_array: np.ndarray, target_category: str, target_prompt: str, device: str = "cuda") -> Dict[str, float]:
    category_prompts = CATEGORY_PROMPTS[target_category]
    raw_scores = []
    
    for i, prompt in enumerate(category_prompts, 1):        
        score = clip_score_fn(
            torch.from_numpy(image_array[None, ...]).to(device), 
            [prompt]
        ).detach()
        prompt_score = float(score)
        raw_scores.append(prompt_score)
    
    raw_scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
    probabilities = torch.softmax(raw_scores_tensor, dim=0)
    
    target_index = category_prompts.index(target_prompt)
    target_probability = float(probabilities[target_index]) * 100 
    
    return {
        target_prompt: round(target_probability, 2)
    }

class CategoryCLIPDetector:
    
    def __init__(self, target_category: str, target_prompt: str, device: str = "cuda", **kwargs):
        self.target_category = target_category.lower()
        self.target_prompt = target_prompt
        self.device = device
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        category_scores = calculate_category_clip_classifier(image_array, self.target_category, self.target_prompt, self.device)
        target_score = category_scores.get(self.target_prompt, 0.0)
        
        return {
            'score': target_score,
            'is_detected': target_score >= 90.0,
            'details': {
                'target_category': self.target_category,
                'target_prompt': self.target_prompt,
                'target_score': target_score,
                'category_scores': category_scores,
                'detector_type': 'category_clip'
            }
        }
    
    def get_reward_score(self, image_path: str) -> float:
        result = self.detect(image_path)
        return result.get('score', 0.0)
    
    def should_early_stop(self, image_path: str, threshold: float = 90.0) -> bool:
        result = self.detect(image_path)
        return result.get('score', 0.0) >= threshold