import torch
import torch.nn as nn
import torch.optim as optim
from segment_anything import sam_model_registry, SamPredictor  # Placeholder for actual SAM library import
import torch
from preprocess import get_bbox_coords, get_bounding_box, get_ground_truth_masks, preprocess
import numpy as np
from statistics import mean
from torch.nn.functional import threshold, normalize
from utils.plot import plot_mean_losses
from models.sam_decoder import SAM_Decoder
import os

if __name__ == "__main__" :
    # Preprocess
    dataset_dir = '/root/ai/dataset/puppy_poo/dataset_seg/'

    bbox_coords = get_bbox_coords(dataset_dir)
    ground_truth_masks = get_ground_truth_masks(bbox_coords, dataset_dir)
    
    # Prepare FineTune
    model_type = 'vit_b'
    checkpoint = '/root/ai/weights/sam_vit_b_01ec64.pth'
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    print(device)

    # Load pre-trained SAM
    sam_encoder = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_encoder.to(device)
    
    sam_decoder = SAM_Decoder(sam_encoder = sam_encoder.image_encoder, sam_preprocess = sam_encoder.preprocess)
    sam_decoder = sam_decoder.to(device)
    # 학습 모드 시작
    sam_decoder.train()
    
    transformed_data, transform = preprocess(sam_encoder, bbox_coords, device, dataset_dir)
    
    # Set up the optimizer, hyperparameter tuning will improve performance here
    lr = 0.001
    optimizer = torch.optim.Adam(sam_decoder.parameters(), lr=lr)

    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.BCELoss()
    keys = list(bbox_coords.keys())
    # Finetune
    num_epochs = 100
    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        # Just train on the first 20 examples
        for k in keys[:20]:
            input_image = transformed_data[k]['image'].to(device)
            input_size = transformed_data[k]['input_size']
            original_image_size = transformed_data[k]['original_image_size']

            # No grad here as we don't want to optimise the encoders
            with torch.no_grad():
                image_embedding = sam_encoder.image_encoder(input_image)

                prompt_box = bbox_coords[k]
                box = transform.apply_boxes(prompt_box, original_image_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                box_torch = box_torch[None, :]

                sparse_embeddings, dense_embeddings = sam_encoder.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
            low_res_masks, iou_predictions = sam_encoder.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = sam_encoder.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            loss = loss_fn(binary_mask, gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(epoch_losses)
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
        # 매 5 에폭마다 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(sam_decoder.state_dict(), os.path.join(save_path, "sam_enc_custom_decoder.pt"))
            save_path = f'/root/ai/segmenatation/result/poopy_sam/sam_model_epoch_{epoch+1}.pt'
            torch.save(sam_encoder.state_dict(), save_path)
            print(f'Model saved to {save_path}')
    plot_mean_losses(mean, losses, '/root/ai/segmenatation/result')

    