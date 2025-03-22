from fastapi import FastAPI, File, UploadFile
import numpy as np
import torch
from PIL import Image
import io

app = FastAPI()

from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))

    img_array = np.array(image) / 255.0  
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

    return img_tensor.tolist()

@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    return {"processed_image": processed_image}
