import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from statistics import mean
from torch.nn.functional import threshold, normalize
from segment_anything import sam_model_registry, SamPredictor  # Placeholder for actual SAM library import
from preprocess import get_bbox_coords, get_bounding_box, get_ground_truth_masks, preprocess
from utils.plot import plot_mean_losses
import matplotlib.pyplot as plt

def train_epoch(sam_model, optimizer, loss_fn, transformed_data, bbox_coords, ground_truth_masks, device, transform, keys):
    epoch_losses = []
    for k in keys:
        input_image = transformed_data[k]['image'].to(device)
        input_size = transformed_data[k]['input_size']
        original_image_size = transformed_data[k]['original_image_size']

        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)

            prompt_box = bbox_coords[k]
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

        loss = loss_fn(binary_mask, gt_binary_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    return mean(epoch_losses)

def validate_epoch(sam_model, loss_fn, transformed_data, bbox_coords, ground_truth_masks, device, transform, keys):
    epoch_losses = []
    with torch.no_grad():
        for k in keys:
            input_image = transformed_data[k]['image'].to(device)
            input_size = transformed_data[k]['input_size']
            original_image_size = transformed_data[k]['original_image_size']

            image_embedding = sam_model.image_encoder(input_image)

            prompt_box = bbox_coords[k]
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            loss = loss_fn(binary_mask, gt_binary_mask)
            epoch_losses.append(loss.item())
    
    return mean(epoch_losses)

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
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    # 학습 모드 시작
    sam_model.train()
    
    transformed_data, transform = preprocess(sam_model, bbox_coords, device, dataset_dir)
    
    # Split keys into train and validation sets with shuffling
    keys = list(bbox_coords.keys())
    random.shuffle(keys)  # Shuffle the keys randomly
    split_idx = int(0.8 * len(keys))  # 80% for training, 20% for validation
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]
    
    # Set up the optimizer, hyperparameter tuning will improve performance here
    lr = 1e-4
    wd = 0
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()

    # Finetune
    num_epochs = 300
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        sam_model.train()
        train_loss = train_epoch(sam_model, optimizer, loss_fn, transformed_data, bbox_coords, ground_truth_masks, device, transform, train_keys)
        train_losses.append(train_loss)
        
        sam_model.eval()
        val_loss = validate_epoch(sam_model, loss_fn, transformed_data, bbox_coords, ground_truth_masks, device, transform, val_keys)
        val_losses.append(val_loss)

        print(f'EPOCH: {epoch}')
        print(f'Train loss: {train_loss}')
        print(f'Validation loss: {val_loss}')
        os.makedirs(f"/root/ai/segmenatation/result/poopy_samv2", exist_ok=True)
        # 매 10 에폭마다 모델 저장
        if (epoch + 1) % 10 == 0:
            save_path = f'/root/ai/segmenatation/result/poopy_samv2/sam_model_epoch_{epoch+1}.pth'
            torch.save(sam_model.state_dict(), save_path)
            print(f'Model saved to {save_path}')

    # Plotting train and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Losses')
    plt.savefig('/root/ai/segmenatation/result/train_val_losses.png')


