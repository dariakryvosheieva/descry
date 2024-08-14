import torch

from detection import get_detector, get_textbox
from recognition import get_recognizer, get_text
from utils import concatenate, group_text_box, get_image_list, diff


unicode_ranges = {
    "adlam": [["1E900", "1E94B"], ["1E950", "1E959"], ["1E95E", "1E95F"]],
    "kayahli": [["A900", "A92F"]]
}

right_to_left = ["adlam"]


class Converter:
    def __init__(self, script):
        self.unicode_values = concatenate(unicode_ranges[script])
        self.labels_to_chars = ["blank"] + [chr(unicode_value) for unicode_value in self.unicode_values]

    def decode(self, text_index, length):
        texts = []
        index = 0
        for l in length:
            t = text_index[index : index + l]
            char_list = []
            for i in range(l):
                if (t[i] != 0) and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.labels_to_chars[t[i]])
            text = "".join(char_list)
            texts.append(text)
            index += l
        return texts


class Reader:
    def __init__(self, script):
        self.script = script
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = get_detector(f"detection_models/{script}.pth", self.device)
        self.recognizer = get_recognizer(f"recognition_models/{script}.pth", self.device)
        self.converter = Converter(script)

    def detect(self, image, min_size=20):
        text_box_list = get_textbox(self.detector, image, self.device)
        horizontal_list_agg, free_list_agg = [], []
        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(text_box)
            horizontal_list_agg.append(
                [i for i in horizontal_list if max(i[1] - i[0], i[3] - i[2]) > min_size]
            )
            free_list_agg.append(
                [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
            )
        return horizontal_list_agg, free_list_agg

    def recognize(self, img, horizontal_list=None, free_list=None):
        if horizontal_list is None and free_list is None:
            y_max, x_max = img.height, img.width
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []
        result = []
        for bbox in horizontal_list:
            h_list = [bbox]
            f_list = []
            image_list = get_image_list(h_list, f_list, img)
            result0 = get_text(self.recognizer, self.converter, image_list, self.device)
            result += result0
        for bbox in free_list:
            h_list = []
            f_list = [bbox]
            image_list = get_image_list(h_list, f_list, img)
            result0 = get_text(self.recognizer, self.converter, image_list, self.device)
            result += result0
        return result

    def readtext(self, image):
        horizontal_list, free_list = self.detect(image)
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        result = self.recognize(image, horizontal_list, free_list)
        if self.script in right_to_left:
            result = [string[::-1] for string in result]
        return result
