import matplotlib.pyplot as plt
import numpy as np

def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss, loss_test):
    plt.figure()
    plt.plot(iteration, loss, color='red', label='Train')
    plt.plot(iteration, loss_test, color='blue', label='Test')
    plt.legend()  # 显示图例

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()