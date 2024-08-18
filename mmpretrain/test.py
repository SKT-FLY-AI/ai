from mmpretrain.apis import init_model, inference_model
import torch

if __name__ == "__main__":
    config_file = '/root/ai/mmpretrain/work_dirs/swin/20240814_200125/vis_data/config.py'
    checkpoint_file = '/root/ai/mmpretrain/work_dirs/swin/epoch_100.pth'
    type = "inferencer_type"
    input_path = "/root/ai/dataset/classification/train/4/Type3_iter348_jpg.rf.0996ba2bab045b26264af1a3418c7f19.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model(config_file, checkpoint_file, device=device)  # or device='cuda:0'
    result = inference_model(model, input_path, type='Image Classification')
    print(result)