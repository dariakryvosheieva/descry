import os
import random

from bidi import get_display
from PIL import Image, ImageFont, ImageDraw

from config.load_config import load_yaml, DotDict
from utils.util import concatenate


config = DotDict(load_yaml("custom_data_train"))

script = config.script_name

unicode_values = concatenate(config.unicode_ranges)


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

    def intersects_any(self, s):
        for other in s:
            if self.intersects(other):
                return True
        return False


def get_random_string():
    length = random.randint(1, config.max_word_length)
    word = ""
    for _ in range(length):
        word += chr(random.choice(unicode_values))
    return word


def get_random_font():
    font_file = random.choice(os.listdir(f"fonts/{script}"))
    return f"fonts/{script}/{font_file}"


def draw_random_strings(dataset, i):
    dim = config.image_dimension
    background_color = 0 if random.random() < config.p_white_on_black else 1
    img = Image.new("1", (dim, dim), background_color)
    d = ImageDraw.Draw(img)
    text_boxes = []
    strings = []
    for _ in range(random.randint(1, config.max_words_per_image)):
        drawn = False
        while not drawn:
            font = ImageFont.truetype(
                get_random_font(),
                random.randint(config.min_font_size, config.max_font_size)
            )
            string = get_random_string()
            string_to_draw = get_display(string) if config.right_to_left else string
            _, _, w, h = d.textbbox((0, 0), string_to_draw, font=font)
            tb = Rectangle(Point(0, 0), Point(w, h))
            x = random.randint(0, dim - tb.width)
            y = random.randint(0, dim - tb.height)
            tb.anchor_at(x, y)
            if not tb.intersects_any(text_boxes):
                text_boxes.append(tb)
                strings.append(string)
                d.text((x, y), string_to_draw, font=font, fill=1-background_color)
                drawn = True
    img.save(
        f"data_root_dir/ch4_{dataset}_images/img_{i}.jpeg",
        "jpeg"
    )
    with open(
        f"data_root_dir/ch4_{dataset}_localization_transcription_gt/gt_img_{i}.txt",
        "a",
        encoding="utf-8"
    ) as f:
        for (tb, s) in zip(text_boxes, strings):
            f.write(
                f"{tb.tl.x},{tb.tl.y},{tb.br.x},{tb.tl.y},{tb.br.x},{tb.br.y},{tb.tl.x},{tb.br.y},{s}\n"
            )


if __name__ == "__main__":
    if not os.path.exists("data_root_dir"):
        os.makedirs("data_root_dir")
        for dataset in ["training", "test"]:
            for file_type in ["images", "localization_transcription_gt"]:
                os.makedirs(f"data_root_dir/{dataset}/{file_type}")

    num_training_images = int(config.num_images * config.fraction_train)
    num_test_images = int(config.num_images * (1 - config.fraction_train))

    for i in range(num_training_images):
        draw_random_strings("training", i)
    for i in range(num_test_images):
        draw_random_strings("test", i)
