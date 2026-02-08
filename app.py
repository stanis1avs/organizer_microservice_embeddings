from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os
import logging
import base64
from PIL import Image
import io

# sentence-transformers
from sentence_transformers import SentenceTransformer
# torch for image embeddings
import torch
import torchvision.transforms as transforms
from torchvision import models

logger = logging.getLogger("uvicorn")
app = FastAPI(title="Embedding Service")

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# normalise output? True recommended
NORMALIZE = os.environ.get("EMBEDDING_NORMALIZE", "1") == "1"

# Load models
print("Loading text model:", MODEL_NAME)
text_model = SentenceTransformer(MODEL_NAME)
print("Text model loaded.")

# Load image model (ResNet for image embeddings)
print("Loading image model...")
image_model = models.resnet50(pretrained=True)
image_model.eval()
# Remove final classification layer for feature extraction
image_model = torch.nn.Sequential(*list(image_model.children())[:-1])
print("Image model loaded.")

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class EmbedRequest(BaseModel):
    text: str
    size: Optional[int] = 384

class EmbedResponse(BaseModel):
    embedding: List[float]
    size: int

class ImageEmbedRequest(BaseModel):
    image_path: Optional[str] = None
    image_data: Optional[str] = None  # base64 encoded
    size: Optional[int] = 512

class ImageEmbedResponse(BaseModel):
    embedding: List[float]
    size: int

def normalize(v: np.ndarray):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(req: EmbedRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    try:
        vec = text_model.encode(req.text, show_progress_bar=False)
        if NORMALIZE:
            vec = normalize(np.array(vec))
        else:
            vec = vec.tolist()
        
        # Handle different vector sizes
        if req.size and len(vec) != req.size:
            if req.size == 512:
                # For 512D, we need to pad or project the vector
                # Simple approach: pad with zeros or repeat values
                if len(vec) > 512:
                    vec = vec[:512]
                else:
                    # Pad with zeros
                    vec = np.pad(vec, (0, req.size - len(vec)), 'constant')
            else:
                # For other sizes, truncate or pad
                if len(vec) > req.size:
                    vec = vec[:req.size]
                else:
                    vec = np.pad(vec, (0, req.size - len(vec)), 'constant')
        
        return {"embedding": vec.tolist() if isinstance(vec, np.ndarray) else vec, "size": len(vec)}
    except Exception as e:
        logger.exception("Text embedding failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed-image", response_model=ImageEmbedResponse)
async def embed_image(req: ImageEmbedRequest):
    try:
        # Load image
        if req.image_path:
            image = Image.open(req.image_path).convert('RGB')
        elif req.image_data:
            # Decode base64 image
            image_bytes = base64.b64decode(req.image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Preprocess and extract features
        image_tensor = image_transform(image).unsqueeze(0)
        with torch.no_grad():
            features = image_model(image_tensor)
            # Flatten and convert to numpy
            features = features.squeeze().numpy()
        
        # Normalize if needed
        if NORMALIZE:
            features = normalize(features)
        
        # Ensure features is a numpy array and convert to list
        if isinstance(features, np.ndarray):
            embedding = features.tolist()
        else:
            embedding = list(features)
        
        return {"embedding": embedding, "size": len(embedding)}
    except Exception as e:
        logger.exception("Image embedding failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
