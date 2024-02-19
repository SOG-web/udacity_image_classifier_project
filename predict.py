import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json

class FlowerClassifier:
    def __init__(self, checkpoint_path, category_names_path='cat_to_name.json'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_checkpoint(checkpoint_path)
        self.model.eval()
        self.category_to_name = self.load_category_names(category_names_path)
        
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        model = getattr(models, checkpoint['vgg_type'])(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        return model.to(self.device)
    
    def load_category_names(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
        
    def process_image(self, image_path):
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = preprocess(image).unsqueeze(0)
        return image.to(self.device)
    
    def predict(self, image_path, topk=5):
        image = self.process_image(image_path)
        with torch.no_grad():
            output = F.softmax(self.model(image), dim=1)
        probabilities, indices = output.topk(topk)
        probabilities = probabilities.cpu().numpy().tolist()[0]
        indices = indices.cpu().numpy().tolist()[0]
        classes = [self.category_to_name[str(idx)] for idx in indices]
        return probabilities, classes

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Flower Image Classifier Prediction")
    parser.add_argument('image_path', metavar='path', type=str, help='path to image file')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, help='path to checkpoint file')
    parser.add_argument('--category_names', type=str, help='path to category names JSON file', default='cat_to_name.json')
    parser.add_argument('--top_k', type=int, help='return top K most likely classes', default=5)
    args = parser.parse_args()
    
    classifier = FlowerClassifier(args.checkpoint, args.category_names)
    probabilities, classes = classifier.predict(args.image_path, args.top_k)
    for i, (prob, cls) in enumerate(zip(probabilities, classes), 1):
        print(f"{i}) Class: {cls}, Probability: {prob:.2%}")
