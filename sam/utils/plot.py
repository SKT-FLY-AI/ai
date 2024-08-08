from matplotlib import pyplot as plt

def plot_mean_losses(mean, losses, save_path):
    mean_losses = [mean(x) for x in losses]

    plt.plot(list(range(len(mean_losses))), mean_losses)
    plt.title('Mean epoch loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')

    # 그래프를 파일로 저장
    plt.savefig(save_path)
    plt.close()
    