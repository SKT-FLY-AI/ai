import os
import torch
from segment_anything import sam_model_registry
from segmenatation.models.sam_decoder import SAM_Decoder
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision

def model_load(encoder_path, decoder_path, device):
    print(f"gpu available : {device}")
    sam = sam_model_registry["vit_b"](checkpoint=encoder_path)
    sam = sam.to(device)

    sam_decoder = SAM_Decoder(sam_encoder = sam.image_encoder, sam_preprocess = sam.preprocess)
    sam_decoder = sam_decoder.to(device)
    # 디코더 가중치 로드
    sam_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    sam_decoder.eval()
    print("SAM model + Custom Decoder set to EVAL mode")
    return sam, sam_decoder

def predict(sam_decoder, input_img, device):
    resize = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.NEAREST)
    resized_image = resize(input_img)
    image_np = np.array(resized_image)
    image = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
    print(image.shape)
    with torch.no_grad():
        pred_masks = sam_decoder(image.to(device).unsqueeze(0))
    np_pred = ((pred_masks > 0.5) * 1).to("cpu").numpy()[0].transpose(1, 2, 0)
    inverted_mask = 1 - np_pred  # 1이면 배경(흰색), 0이면 객체(검은색)
    original_image = image.cpu().numpy().transpose(1, 2, 0)
    white_background_image = np.ones_like(original_image) * 255
    masked_image = np.where(inverted_mask == 1, white_background_image, original_image)
    masked_image_uint8 = masked_image.squeeze().astype(np.uint8)

    return masked_image_uint8, inverted_mask.squeeze().astype(np.uint8) * 255

def predict_save(sam_decoder, input_path, save_path, device):
    # 이미지 전처리
    resize = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.NEAREST)
    image = torchvision.io.read_image(input_path)
    image = resize(image)

    with torch.no_grad():
        # 모델에 입력하여 예측 수행
        pred_masks = sam_decoder(image.to(device).unsqueeze(0))
    
    np_pred = ((pred_masks > 0.5) * 1).to("cpu").numpy()[0].transpose(1, 2, 0)
    pred_image = Image.fromarray(np_pred.squeeze().astype(np.uint8) * 255)  # 이진화된 마스크를 0-255로 변환
    filename = os.path.splitext(os.path.basename(input_path))[0]
    mask_save_path = os.path.join(save_path, filename + "_mask.png")
    pred_image.save(mask_save_path)
    
    # 마스크를 적용한 원본 이미지 생성 및 저장
    original_image = image.cpu().numpy().transpose(1, 2, 0)
    masked_image = original_image * np_pred  # 마스크 적용

    masked_image = Image.fromarray(masked_image.squeeze().astype(np.uint8))
    masked_image.save(os.path.join(save_path, filename + "_masked.png"))
    
    # 원본 이미지 저장
    original_image = Image.fromarray(original_image)
    original_image.save(os.path.join(save_path, filename + "_original.png"))
    
def predict_path(sam_decoder, input_path, device):
    resize = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.NEAREST)
    image = torchvision.io.read_image(input_path)
    image = resize(image)

    with torch.no_grad():
        pred_masks = sam_decoder(image.to(device).unsqueeze(0))
    
    np_pred = ((pred_masks > 0.5) * 1).to("cpu").numpy()[0].transpose(1, 2, 0)
    inverted_mask = 1 - np_pred  # 1이면 배경(흰색), 0이면 객체(검은색)
    original_image = image.cpu().numpy().transpose(1, 2, 0)
    white_background_image = np.ones_like(original_image) * 255
    masked_image = np.where(inverted_mask == 1, white_background_image, original_image)
    masked_image_uint8 = masked_image.squeeze().astype(np.uint8)

    return masked_image_uint8, inverted_mask.squeeze().astype(np.uint8) * 255

if __name__ == "__main__":
    encoder_path = '/root/ai/weights/sam_vit_b_01ec64.pth'
    decoder_path = '/root/ai/weights/sam_enc_custom_decoder.pt'
    input_path = '/root/ai/dataset/mask_1513_split/img/33_jpeg_jpg.rf.afdfbc41a987670e2159597db4fc3f25.jpg'
    input_path = "/root/ai/segmenatation/infant_poo.png"
    save_path = '/root/ai/predict'
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    print(device)
    
    sam_decoder = model_load(encoder_path, decoder_path, device)
    predict_save(sam_decoder, input_path, save_path, device)
