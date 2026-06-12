try:
    from dotenv import load_dotenv
    import os as _os
    load_dotenv(dotenv_path=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env"), override=True)
except ImportError:
    pass  # python-dotenv не установлен — переменные берутся из окружения

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os
import logging
import base64
import asyncio
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from PIL import Image
import io

from sentence_transformers import SentenceTransformer
import torch
import torchvision.transforms as transforms
from torchvision import models

logger = logging.getLogger("uvicorn")
app = FastAPI(title="Embedding Service")

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
NORMALIZE = os.environ.get("EMBEDDING_NORMALIZE", "1") == "1"
ALLOWED_IMAGE_DIR = os.path.abspath(os.environ.get("ALLOWED_IMAGE_DIR", "/app/files"))

# S-05: API-ключ для защиты эндпоинтов.
# Если EMBEDDING_API_KEY не задан — проверка отключена (обратная совместимость).
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def _verify_api_key(api_key: Optional[str] = Security(_api_key_header)):
    if not EMBEDDING_API_KEY:
        return
    if api_key != EMBEDDING_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# P: выделенный thread pool для inference.
# Дефолт 2 — не перегружает CPU при параллельных запросах.
_INFERENCE_WORKERS = int(os.environ.get("INFERENCE_WORKERS", "2"))
_executor = ThreadPoolExecutor(max_workers=_INFERENCE_WORKERS)

# P: LRU-кэш для текстовых эмбеддингов.
# Поисковые запросы часто повторяются — кэш даёт O(1) вместо forward pass (~50ms).
class _LRUCache:
    def __init__(self, maxsize: int = 512):
        self._d: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key: str, value):
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        if len(self._d) > self._maxsize:
            self._d.popitem(last=False)

_embed_cache = _LRUCache(maxsize=int(os.environ.get("EMBED_CACHE_SIZE", "512")))

# Load models
print("Loading text model:", MODEL_NAME)
text_model = SentenceTransformer(MODEL_NAME)
print("Text model loaded.")

print("Loading image model...")
image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
image_model.eval()
# Remove final classification layer — output: 2048D
image_model = torch.nn.Sequential(*list(image_model.children())[:-1])
print("Image model loaded.")

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
    image_data: Optional[str] = None  # base64 (предпочтительный способ)
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


@app.on_event("startup")
async def warmup():
    """Прогрев моделей при старте — первый реальный запрос не будет медленным."""
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_executor, partial(_run_text_inference, "warmup"))
        dummy = torch.zeros(1, 3, 224, 224)
        await loop.run_in_executor(_executor, partial(_run_image_inference, dummy))
        logger.info("Models warmed up.")
    except Exception as e:
        logger.warning("Warmup failed (non-critical): %s", e)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "cache_size": len(_embed_cache._d)}


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(_verify_api_key)])
async def embed_text(req: EmbedRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    try:
        # P: проверяем кэш перед inference
        cache_key = hashlib.md5(req.text.encode("utf-8")).hexdigest()
        cached = _embed_cache.get(cache_key)
        if cached is not None:
            return {"embedding": cached, "size": len(cached)}

        # P: get_running_loop() вместо устаревшего get_event_loop()
        loop = asyncio.get_running_loop()
        vec_raw = await loop.run_in_executor(_executor, partial(_run_text_inference, req.text))
        vec = normalize_vec(np.array(vec_raw)) if NORMALIZE else np.array(vec_raw).tolist()
        final = vec.tolist() if isinstance(vec, np.ndarray) else vec

        if req.size and req.size != len(final):
            logger.warning(
                "Text embedding size mismatch: model=%d, requested=%d. Returning model size.",
                len(final), req.size
            )

        _embed_cache.put(cache_key, final)
        return {"embedding": final, "size": len(final)}
    except Exception as e:
        logger.exception("Text embedding failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed-image", response_model=ImageEmbedResponse, dependencies=[Depends(_verify_api_key)])
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

        # P: get_running_loop() вместо get_event_loop()
        loop = asyncio.get_running_loop()
        features = await loop.run_in_executor(_executor, partial(_run_image_inference, image_tensor))

        embedding = normalize_vec(features) if NORMALIZE else features.tolist()

        if req.size and req.size != len(embedding):
            logger.warning(
                "Image embedding: model=%d, requested=%d. Returning model size. "
                "Update Qdrant collection to %d dimensions.",
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
