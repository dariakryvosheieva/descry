import os
import torch
import random

from bidi import get_display
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from config.load_config import load_yaml, DotDict

config = DotDict(load_yaml("kayahli_recognition_config")) # adlam

script = config.script_name

def concatenate(unicode_ranges):
    rng = []
    for unicode_range in unicode_ranges:
        rng += list(range(int(unicode_range[0], 16), int(unicode_range[1], 16) + 1))
    return rng

unicode_values = concatenate(config.unicode_ranges)

def get_random_string():
    length = random.randint(1, config.max_word_length)
    word = ""
    for _ in range(length):
        word += chr(random.choice(unicode_values))
    return word

def get_random_font():
    font_file = random.choice(os.listdir(f"fonts/{script}"))
    return f"fonts/{script}/{font_file}"

def draw_random_string():
    W, H = config.image_width, config.image_height
    background_color = 0 if random.random() < config.p_white_on_black else 1
    font = ImageFont.truetype(get_random_font(), config.font_size)
    string = get_random_string()
    string_to_draw = get_display(string) if config.right_to_left else string
    img = Image.new("1", (W, H), background_color)
    d = ImageDraw.Draw(img)
    _, _, w, h = d.textbbox((0, 0), string_to_draw, font=font)
    d.text(((W-w)/2, (H-h)/2), string_to_draw, font=font, fill=1-background_color)
    return img, string

class Dataset:
    def __init__(self, data):
        self.data = data
        self.trans = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return self.trans(img), label
    
def generate_dataset(size, filepath):
    dataset = [draw_random_string() for _ in range(size)]
    dataloader = torch.utils.data.DataLoader(
        dataset=Dataset(dataset),
        batch_size=config.batch_size,
        shuffle=False # images already in no particular order
    )
    torch.save(dataloader, filepath)

if __name__ == "__main__":
    generate_dataset(config.train_set_size, config.train_set_file)
    generate_dataset(config.test_set_size, config.test_set_file)