import pyiqa
import torchvision.transforms as T

class Brisque_score:
    def __init__(self):
        pass

    def count(self, image):
        brisque_metric = pyiqa.create_metric('brisque', device='cpu')
        transform = T.ToTensor()

        # Convert ke tensor (hasil = (3, H, W))
        img_tensor = transform(image).unsqueeze(0)  # shape (1, 3, H, W)
        brisque_score = brisque_metric(img_tensor)
        return brisque_score.item()