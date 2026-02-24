import os
import sys
from typing import Dict, Any, List
from transformers import pipeline

class VanGoghStyleDetector:
    
    def __init__(self, classifier_path: str, device: str = "cuda"):
        self.classifier_path = classifier_path
        self.device = device
        self.classifier = None
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        if os.path.exists(self.classifier_path):
            self.classifier = pipeline(
                'image-classification',
                model=self.classifier_path,
                device=self.device
            )
    
    def detect_style(self, image_path: str, top_k: int = 129) -> List[Dict[str, Any]]:        
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        results = self.classifier(image, top_k=top_k)
        return results
    
    def get_vangogh_score(self, image_path: str) -> float:
        style_results = self.detect_style(image_path)
        vangogh_labels = ['vincent-van-gogh', 'van-gogh', 'vangogh', 'Vincent van Gogh']
        
        for result in style_results:
            label = result.get('label', '').lower()
            print(f"all detect_style: {label}")
            for vangogh_label in vangogh_labels:
                if vangogh_label.lower() in label:
                    return result.get('score', 0.0)
        
        return 0.0
    
    def is_vangogh_in_top3(self, image_path: str) -> bool:
        style_results = self.detect_style(image_path, top_k=3)
        vangogh_labels = ['vincent-van-gogh', 'van-gogh', 'vangogh', 'Vincent van Gogh']
        
        for result in style_results:
            label = result.get('label', '').lower()
            for vangogh_label in vangogh_labels:
                if vangogh_label.lower() in label:
                    return True
        
        return False
    
    def is_vangogh_top1(self, image_path: str) -> bool:
        style_results = self.detect_style(image_path, top_k=1)
        
        if not style_results:
            return False
        
        vangogh_labels = ['vincent-van-gogh', 'van-gogh', 'vangogh', 'Vincent van Gogh']
        
        top1_label = style_results[0].get('label', '').lower()
        for vangogh_label in vangogh_labels:
            if vangogh_label.lower() in top1_label:
                return True
        
        return False
    
    def is_available(self) -> bool:
        return self.classifier is not None
    
    def get_detailed_results(self, image_path: str) -> Dict[str, Any]:
        style_results = self.detect_style(image_path)
        vangogh_score = self.get_vangogh_score(image_path)
        is_top1 = self.is_vangogh_top1(image_path)
        is_top3 = self.is_vangogh_in_top3(image_path)
        
        return {
            'vangogh_score': vangogh_score,
            'is_vangogh_top1': is_top1,
            'is_vangogh_top3': is_top3,
            'all_predictions': style_results[:10],
            'detector_type': 'vangogh_style'
        }


class PicassoStyleDetector:
    def __init__(self, classifier_path: str, device: str = "cuda"):
        self.classifier_path = classifier_path
        self.device = device
        self.classifier = None
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        if os.path.exists(self.classifier_path):
            self.classifier = pipeline(
                'image-classification',
                model=self.classifier_path,
                device=self.device
            )
    
    def detect_style(self, image_path: str, top_k: int = 129) -> List[Dict[str, Any]]:        
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        results = self.classifier(image, top_k=top_k)
        return results
    
    def get_picasso_score(self, image_path: str) -> float:
        style_results = self.detect_style(image_path)
        picasso_labels = ['pablo-picasso', 'picasso', 'Pablo Picasso']
        
        for result in style_results:
            label = result.get('label', '').lower()
            print(f"all detect_style: {label}")
            for picasso_label in picasso_labels:
                if picasso_label.lower() in label:
                    return result.get('score', 0.0)
        
        return 0.0
    
    def is_picasso_in_top3(self, image_path: str) -> bool:
        style_results = self.detect_style(image_path, top_k=3)
        
        picasso_labels = ['pablo-picasso', 'picasso', 'Pablo Picasso']
        
        for result in style_results:
            label = result.get('label', '').lower()
            for picasso_label in picasso_labels:
                if picasso_label.lower() in label:
                    return True
        
        return False
    
    def is_picasso_top1(self, image_path: str) -> bool:
        style_results = self.detect_style(image_path, top_k=1)
        
        if not style_results:
            return False
        picasso_labels = ['pablo-picasso', 'picasso', 'Pablo Picasso']
        
        top1_label = style_results[0].get('label', '').lower()
        for picasso_label in picasso_labels:
            if picasso_label.lower() in top1_label:
                return True
        
        return False
    
    def is_available(self) -> bool:
        return self.classifier is not None
    
    def get_detailed_results(self, image_path: str) -> Dict[str, Any]:
        style_results = self.detect_style(image_path)
        picasso_score = self.get_picasso_score(image_path)
        is_top1 = self.is_picasso_top1(image_path)
        is_top3 = self.is_picasso_in_top3(image_path)
        
        return {
            'picasso_score': picasso_score,
            'is_picasso_top1': is_top1,
            'is_picasso_top3': is_top3,
            'all_predictions': style_results[:10],
            'detector_type': 'picasso_style'
        } 