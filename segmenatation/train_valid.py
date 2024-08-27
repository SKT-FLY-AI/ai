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
from models.sam_decoder import SAM_Decoder
from torchvision import transforms
import torchvision

def train_epoch(sam_decoder, optimizer, loss_fn, transformed_data, bbox_coords, ground_truth_masks, device, transform, keys, resize):
    epoch_losses = []
    for i, k in enumerate(keys):
        input_image = transformed_data[k]['image'].to(device)
        input_image = resize(input_image)  # Resize the image to 1024x1024
        input_size = transformed_data[k]['input_size']
        original_image_size = transformed_data[k]['original_image_size']

        pred_masks = sam_decoder(input_image)

        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
        gt_mask_resized = resize(gt_mask_resized)  # Resize the ground truth mask to 1024x1024
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32).to(device)

        loss = loss_fn(pred_masks, gt_binary_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

        # Log the details of the training process
        print(f"[Train] Batch {i+1}/{len(keys)}, Loss: {loss.item():.4f}")

    return mean(epoch_losses)

def validate_epoch(sam_decoder, loss_fn, transformed_data, bbox_coords, ground_truth_masks, device, transform, keys, resize):
    epoch_losses = []
    with torch.no_grad():
        for i, k in enumerate(keys):
            input_image = transformed_data[k]['image'].to(device)
            input_image = resize(input_image)  # Resize the image to 1024x1024
            input_size = transformed_data[k]['input_size']
            original_image_size = transformed_data[k]['original_image_size']

            pred_masks = sam_decoder(input_image)

            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
            gt_mask_resized = resize(gt_mask_resized)  # Resize the ground truth mask to 1024x1024
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            loss = loss_fn(pred_masks, gt_binary_mask)
            epoch_losses.append(loss.item())

            # Log the details of the validation process
            print(f"[Validation] Batch {i+1}/{len(keys)}, Loss: {loss.item():.4f}")

    return mean(epoch_losses)

if __name__ == "__main__":
    save_path = "/root/ai/segmenatation/result/poopy_samv2"
    dataset_dir = '/root/ai/dataset/puppy_poo/dataset_seg/'
    bbox_coords = get_bbox_coords(dataset_dir)
    ground_truth_masks = get_ground_truth_masks(bbox_coords, dataset_dir)

    model_type = 'vit_b'
    checkpoint = '/root/ai/weights/sam_vit_b_01ec64.pth'
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    print(device)

    sam_encoder = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_encoder.to(device)

    sam_decoder = SAM_Decoder(sam_encoder=sam_encoder.image_encoder, sam_preprocess=sam_encoder.preprocess)
    sam_decoder = sam_decoder.to(device)
    sam_decoder.train()

    transformed_data, transform = preprocess(sam_encoder, bbox_coords, device, dataset_dir)

    keys = list(bbox_coords.keys())
    random.shuffle(keys)
    split_idx = int(0.8 * len(keys))
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]

    lr = 0.001
    optimizer = torch.optim.Adam(sam_decoder.parameters(), lr=lr)

    loss_fn = torch.nn.BCELoss()

    resize = torchvision.transforms.Resize(
        (1024, 1024),
        interpolation=torchvision.transforms.InterpolationMode.NEAREST
    )

    num_epochs = 100
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")

        sam_decoder.train()
        train_loss = train_epoch(sam_decoder, optimizer, loss_fn, transformed_data, bbox_coords, ground_truth_masks, device, transform, train_keys, resize)
        train_losses.append(train_loss)

        sam_decoder.eval()
        val_loss = validate_epoch(sam_decoder, loss_fn, transformed_data, bbox_coords, ground_truth_masks, device, transform, val_keys, resize)
        val_losses.append(val_loss)

        print(f'EPOCH: {epoch + 1}/{num_epochs}')
        print(f'Train loss: {train_loss:.4f}')
        print(f'Validation loss: {val_loss:.4f}')

        os.makedirs(f"{save_path}", exist_ok=True)
        if (epoch + 1) % 10 == 0:
            save_path_epoch = f'{save_path}/sam_model_epoch_{epoch+1}.pt'
            torch.save(sam_decoder.state_dict(), save_path_epoch)
            print(f'Model saved to {save_path_epoch}')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Losses')
    plt.savefig(f'{save_path}/train_val_losses.png')
