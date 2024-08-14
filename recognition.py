import math
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image

import numpy as np


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
        self.n_samples = len(image_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        img = self.image_list[index]
        img = Image.fromarray(img)
        img = img.point(lambda p: 255 if p > 150 else 0)
        return img.convert("1")


def process_for_recognition(image, canvas_width=200, canvas_height=64):
    background_color = image.getpixel((0, 0))
    canvas = Image.new("1", (canvas_width, canvas_height), background_color)
    mag_factor = min(canvas_width / image.width, canvas_height / image.height)
    image = image.resize((int(image.width * mag_factor), int(image.height * mag_factor)))
    canvas.paste(image, ((canvas_width - image.width) // 2, (canvas_height - image.height) // 2))
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(canvas)


def collate(images):
    resized_images = []
    for image in images:
        resized_images.append(process_for_recognition(image))
    image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
    return image_tensors


def get_recognizer(filepath, device):
    return torch.load(filepath, map_location=device).module.to(device)


def recognizer_predict(model, converter, test_loader, device):
    model.eval()
    result = []
    with torch.no_grad():
        for image_tensors in test_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            preds = model(image)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)
            for pred in preds_str:
                result.append(pred)
    return result


def get_text(recognizer, converter, image_list, device="cpu"):
    img_list = [item[1] for item in image_list]
    test_data = collate(ListDataset(img_list))
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, pin_memory=True
    )
    return recognizer_predict(recognizer, converter, test_loader, device)
