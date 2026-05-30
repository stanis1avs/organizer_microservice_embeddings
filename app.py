from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os
import logging
import base64
import asyncio
from functools import partial
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
NORMALIZE = os.environ.get("EMBEDDING_NORMALIZE", "1") == "1"

ALLOWED_IMAGE_DIR = os.path.abspath(os.environ.get("ALLOWED_IMAGE_DIR", "/app/files"))

# Load models
print("Loading text model:", MODEL_NAME)
text_model = SentenceTransformer(MODEL_NAME)
print("Text model loaded.")

# Load image model (ResNet for image embeddings)
print("Loading image model...")
image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
image_model.eval()
# Remove final classification layer for feature extraction (output: 2048D)
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
    image_data: Optional[str] = None  # base64 encoded (предпочтительный способ)
    size: Optional[int] = 512

class ImageEmbedResponse(BaseModel):
    embedding: List[float]
    size: int

def normalize_vec(v: np.ndarray) -> list:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()

def _run_text_inference(text: str) -> np.ndarray:
    return text_model.encode(text, show_progress_bar=False)

def _run_image_inference(image_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        features = image_model(image_tensor)
    return features.squeeze().numpy()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(req: EmbedRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    try:
        loop = asyncio.get_event_loop()
        vec_raw = await loop.run_in_executor(None, partial(_run_text_inference, req.text))
        vec = normalize_vec(np.array(vec_raw)) if NORMALIZE else np.array(vec_raw).tolist()


        if req.size and req.size != len(vec):
            logger.warning(
                "Text embedding size mismatch: model produces %d, requested %d. "
                "Returning actual model size to avoid corrupting vector index.",
                len(vec), req.size
            )

        final = vec.tolist() if isinstance(vec, np.ndarray) else vec
        return {"embedding": final, "size": len(final)}
    except Exception as e:
        logger.exception("Text embedding failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed-image", response_model=ImageEmbedResponse)
async def embed_image(req: ImageEmbedRequest):
    try:
        if req.image_path:
            abs_path = os.path.abspath(req.image_path)
            if not abs_path.startswith(ALLOWED_IMAGE_DIR + os.sep) and abs_path != ALLOWED_IMAGE_DIR:
                logger.warning("Blocked image_path outside allowed dir: %s", req.image_path)
                raise HTTPException(status_code=403, detail="Access to this path is not allowed")
            if not os.path.isfile(abs_path):
                raise HTTPException(status_code=404, detail="Image file not found")
            image = Image.open(abs_path).convert("RGB")
        elif req.image_data:
            if len(req.image_data) > 28_000_000:
                raise HTTPException(status_code=413, detail="image_data too large (max ~20MB)")
            image_bytes = base64.b64decode(req.image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="No image provided: pass image_path or image_data")

        image_tensor = image_transform(image).unsqueeze(0)

        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(None, partial(_run_image_inference, image_tensor))

        embedding = normalize_vec(features) if NORMALIZE else features.tolist()

        if req.size and req.size != len(embedding):
            logger.warning(
                "Image embedding size mismatch: model produces %d, requested %d. "
                "Returning actual model size. Update Qdrant collection to %d dimensions.",
                len(embedding), req.size, len(embedding)
            )

        return {"embedding": embedding, "size": len(embedding)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Image embedding failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
