from mmpretrain.apis import ImageClassificationInferencer

if __name__ == "__main__":
    config_file = '/root/ai/weights/resnet101_8xb32_in1k.py'
    checkpoint_file = '/root/ai/weights/best_accuracy_top1_epoch_261.pth'
    input_path = "/root/ai/a-rang-pddong.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inferencer = ImageClassificationInferencer(model=config_file, pretrained=checkpoint_file)
    result = inferencer(inputs=input_path, show_dir="/root/ai/")[0]
    # result = inference_model(inferencer, input_path, type="Image Classification")
    print(result)