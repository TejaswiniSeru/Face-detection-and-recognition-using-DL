import os
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import InceptionResnetV1
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder

from src.detector import detect_faces

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class ModifiedInceptionResnetV1(nn.Module):
    def __init__(self, embedding_size=128):
        super(ModifiedInceptionResnetV1, self).__init__()

        self.inception_resnet = InceptionResnetV1(pretrained='vggface2', classify=False)

        self.embedding_layer = nn.Linear(512, embedding_size)

    def forward(self, x):
        x = self.inception_resnet(x)
        x = self.embedding_layer(x)

        return x
    
def recognize_and_visualize(input_image_path, model, aggregated_embeddings_dict):
    input_image = Image.open(input_image_path)

    bounding_boxes, _ = detect_faces(input_image)
        
    bounding_boxes = bounding_boxes[bounding_boxes[:,4]>0.98]
    bounding_boxes = np.array(bounding_boxes[:,:4], dtype=np.int16)

    label_encoder = LabelEncoder()
    label_encoder.fit(list(aggregated_embeddings_dict.keys()))

    transform = transforms.Compose([
                          transforms.Resize((160, 160)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                      ])

    recognized_faces = []

    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        if x1>=20 and y1>=20:
            face_image = input_image.crop((x1-15, y1-15, x2+15, y2+15))
        else:
            face_image = input_image.crop((x1, y1, x2+15, y2+15))

        face_image = input_image.crop((x1, y1, x2, y2))

        face_image_tensor = transform(face_image).unsqueeze(0)

        with torch.no_grad():
            face_embedding = model(face_image_tensor).squeeze(0)

        recognized_label = None
        
        min_distance = float('inf')

        for label, aggregated_embedding in aggregated_embeddings_dict.items():
            aggregated_embedding = torch.tensor(aggregated_embedding)

            similarity_score = torch.nn.functional.cosine_similarity(face_embedding, aggregated_embedding, dim=0)
            distance = 1 - similarity_score.item()

            if distance < min_distance:
                min_distance = distance
                recognized_label = label

        recognized_faces.append({
            'bbox': bbox,
            'label': recognized_label,
        })
    font_size = 45
    font = ImageFont.truetype("Arial.ttf", font_size)

    draw = ImageDraw.Draw(input_image)

    for face_info in recognized_faces:
        bbox = face_info['bbox']
        recognized_label = face_info['label']

        x1, y1, x2, y2 = bbox

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        text_x = x1
        text_y = y1 - 45

        draw.text((text_x, text_y), recognized_label, fill="red", font=font)

    return input_image

