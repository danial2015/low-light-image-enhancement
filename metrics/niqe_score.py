import pyiqa
import torchvision.transforms as T

class Niqe_score:
    def __init__(self):
        pass

    def count(self, image):
        niqe_metric = pyiqa.create_metric('niqe', device='cpu')
        transform = T.ToTensor()
        img_tensor = transform(image).unsqueeze(0) # (1, 3, H, W)

        # Hitung skor NIQE
        score = niqe_metric(img_tensor)
        # print('NIQE Score:', score.item())
        return score.item()