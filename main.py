from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import torch
from segmenatation.predict_plsam import model_load, predict
from classification.extract_color import calAvgColor, checkBlood, rgb2hex
from PIL import Image
import io

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
    print("Model loaded successfully!")

# /analysis 경로에 대한 POST 요청 처리
@app.post("/analysis")
async def analyze_image(file: UploadFile = File(...)):
    # @TODO : 이미지를 로드하는 부분 
    try:
        file_content = await file.read()
        file_size = len(file_content)
        file_name = file.filename
        # 파일 내용을 PIL 이미지로 변환
        input_img = Image.open(io.BytesIO(file_content)).convert("RGB")
        thrshold_blood = 100
        device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        masked_img, _ = predict(decoder, input_img, device)
        # @TODO : masked_img를 입력으로 사용하는 Classification
        mean_val_bgr, _ = calAvgColor(masked_img)
        isBlood, _ = checkBlood(mean_val_bgr, masked_img, thrshold_blood)
        type = 4
        result = {"poo_type" : type, "poo_color" : rgb2hex(mean_val_bgr), "poo_blood" : isBlood}
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# uvicorn main:app --reload --host=0.0.0.0 --port=8000