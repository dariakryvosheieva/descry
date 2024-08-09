from PIL import Image
import torch
import torch.utils.data
import torchvision.transforms as transforms
import math

class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='center'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        qty = (self.max_size[2] - w) // 2
        Pad_img[:, :, qty:qty+w] = img  # center pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        img = Image.fromarray(img).point(lambda p: 255 if p > 150 else 0)
        return img.convert("1")


class AlignCollate(object):
    def __init__(self, imgH=64, imgW=200, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images = batch

        resized_max_w = self.imgW
        input_channel = 1
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

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
    AlignCollate_normal = AlignCollate(keep_ratio_with_pad=True)
    test_data = ListDataset(img_list)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False,
        collate_fn=AlignCollate_normal, pin_memory=True
    )
    return recognizer_predict(recognizer, converter, test_loader, device)
