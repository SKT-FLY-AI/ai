import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import torch
from segmenatation.predict_plsam import model_load, predict
from mmpretrain.apis import ImageClassificationInferencer
from classification.kmeans import ImageProcessor, rgb_to_hex
from PIL import Image
import io
import json

app = FastAPI()

class ImageData(BaseModel):
    image_path: str

# 메인 함수 - 서버 시작 시 모델 로드
@app.on_event("startup")
def load_model():
    global encoder, decoder
    encoder_path = "weights/sam_vit_b_01ec64.pth"
    decoder_path = "weights/sam_enc_custom_decoder.pt"
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    encoder, decoder = model_load(encoder_path, decoder_path, device)
    print("Segmentation Model loaded successfully!")
    
    global cls_model
    config_path = "weights/resnet101_8xb32_in1k.py"
    ckpt_path = "weights/resnet_261.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cls_model = ImageClassificationInferencer(model=config_path, pretrained=ckpt_path, device=device)
    print("Classification Model loaded successfully!")
    return encoder, decoder, cls_model



# /analysis 경로에 대한 POST 요청 처리
@app.post("/analysis")
async def analyze_image(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        file_size = len(file_content)
        file_name = file.filename
        # 파일 내용을 PIL 이미지로 변환
        input_img = Image.open(io.BytesIO(file_content)).convert("RGB")
        device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        masked_img, _ = predict(decoder, input_img, device)
        processor = ImageProcessor(None, masked_img)
        # 화이트 밸런스 적용 및 색상 양자화 실행
        whitebalanced_image, (quantized_image, color_group) = processor.process_image()
        print(color_group)
        print("processor successfully!")
        os.makedirs("dummy", exist_ok=True)
        result_type = cls_model(inputs=quantized_image, show_dir="dummy")[0]
        print(result_type)
        pred_class, pred_score = result_type["pred_class"], round(result_type["pred_score"], 4)
        
        # type = 4
        result = {"poo_type" : pred_class, "poo_color" : color_group}
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
# uvicorn main:app --reload --host=0.0.0.0 --port=8000