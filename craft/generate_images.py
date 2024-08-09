import os
import math
import random

from bidi import get_display
from PIL import Image, ImageFont, ImageDraw

from config.load_config import load_yaml, DotDict

config = DotDict(load_yaml("custom_data_train"))

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

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def displace(self, dx, dy):
        self.x += dx
        self.y += dy
    
class Rectangle:
    def __init__(self, tl, br):
        self.tl = tl
        self.br = br
        self.width = self.br.x - self.tl.x
        self.height = self.br.y - self.tl.y
        
    def anchor_at(self, x, y):
        dx = x - self.tl.x
        dy = y - self.tl.y
        self.tl.displace(dx, dy)
        self.br.displace(dx, dy)
    
    def intersects(self, other):
        return not (
            other.tl.y > self.br.y
         or other.br.y < self.tl.y
         or other.tl.x > self.br.x
         or other.br.x < self.tl.x
        )

def intersects_any(r, s):
    for other_r in s:
        if r.intersects(other_r):
            return True
    return False

def draw_random_strings(training_or_test, i):
    dim = config.image_dimension
    background_color = 0 if random.random() < config.p_white_on_black else 1
    img = Image.new("1", (dim, dim), background_color)
    d = ImageDraw.Draw(img)
    text_boxes = []
    strings = []
    for _ in range(random.randint(1, config.max_words_per_image)):
        while True:
            font = ImageFont.truetype(get_random_font(), random.randint(config.min_font_size, config.max_font_size))
            string = get_random_string()
            string_to_draw = get_display(string) if config.right_to_left else string
            _, _, w, h = d.textbbox((0, 0), string_to_draw, font=font)
            tb = Rectangle(Point(0, 0), Point(w, h))
            x = random.randint(0, dim - tb.width)
            y = random.randint(0, dim - tb.height)
            tb.anchor_at(x, y)
            if tb.br.x != tb.tl.x and tb.br.y != tb.tl.y and not intersects_any(tb, text_boxes):
                text_boxes.append(tb)
                strings.append(string)
                d.text((x, y), string_to_draw, font=font, fill=1-background_color)
                break
    img.save(f"data_root_dir/ch4_{training_or_test}_images/img_{i}.jpeg", "jpeg")
    with open(f"data_root_dir/ch4_{training_or_test}_localization_transcription_gt/gt_img_{i}.txt", "a", encoding="utf-8") as f:
        for (tb, s) in zip(text_boxes, strings):
            f.write(f"{tb.tl.x},{tb.tl.y},{tb.br.x},{tb.tl.y},{tb.br.x},{tb.br.y},{tb.tl.x},{tb.br.y},{s}\n")

num_training_images = int(config.num_images * config.fraction_train)
num_test_images = int(config.num_images * (1 - config.fraction_train))

# for i in range(num_training_images):
#     draw_random_strings("training", i)

for i in range(num_test_images):
    draw_random_strings("test", i)
