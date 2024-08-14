import time

import torch

from crnn import *
from generate_datasets import Dataset, unicode_values
from config.load_config import load_yaml, DotDict


device = "cuda" if torch.cuda.is_available() else "cpu"

config = DotDict(load_yaml("kayahli_recognition_config")) # adlam

script = config.script_name

num_chars = len(unicode_values)

labels_to_chars = ["blank"] + [chr(unicode_value) for unicode_value in unicode_values]

chars_to_labels = {char: i for i, char in enumerate(labels_to_chars)}


def encode(text):
    length = [len(s) for s in text]
    text = ''.join(text)
    text = [chars_to_labels[char] for char in text]
    return torch.IntTensor(text), torch.IntTensor(length)


def decode(text_index, length):
    texts = []
    index = 0
    for l in length:
        t = text_index[index : index + l]
        char_list = []
        for i in range(l):
            if (t[i] != 0) and (not (i > 0 and t[i - 1] == t[i])):
                char_list.append(labels_to_chars[t[i]])
        text = ''.join(char_list)
        texts.append(text)
        index += l
    return texts


def initialize_model():
    model = Model(config.input_channel, config.output_channel, config.hidden_size, num_chars + 1).to(device)
    for name, param in model.named_parameters():
        try:
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.kaiming_normal_(param)
        except Exception as e:
            if 'weight' in name:
                param.data.fill_(1)
            continue
    torch.save(model, f"recognition_models/{script}.pth")


def train():
    train_set = torch.load(config.train_set_file)
    train_set_iter = iter(train_set)

    model = torch.load(f"recognition_models/{script}.pth")
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    filtered_parameters = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)

    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)

    optimizer = torch.optim.Adadelta(
        filtered_parameters, lr=config.learning_rate, rho=config.rho, eps=config.epsilon
    )

    i = 0
    start_time = time.time()
    while True:
        try:
            optimizer.zero_grad()
            image_tensors, labels = next(train_set_iter)
            image = image_tensors.to(device)
            text, length = encode(labels)
            batch_size = image.size(0)
            preds = model(image).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
            torch.backends.cudnn.enabled = True
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm) 
            optimizer.step()
            if i % config.interval == 0:
                print(f"Iteration: {i} | time: {time.time() - start_time} | loss: {cost}")
            if i == config.num_iterations:
                break
            i += 1
        except StopIteration:
            train_set_iter = iter(train_set)
    torch.save(model, f"recognition_models/{script}.pth")


def test():
    test_set = torch.load(config.test_set_file)

    model = torch.load(f"recognition_models/{script}.pth", map_location=device)
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    n_correct = 0
    length_of_data = 0

    with torch.no_grad():
        for image_tensors, labels in test_set:
            image = image_tensors.to(device)
            batch_size = image.size(0)
            length_of_data = length_of_data + batch_size
            preds = model(image)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = decode(preds_index.data, preds_size.data)
            for gt, pred in zip(labels, preds_str):
                if pred == gt:
                    n_correct += 1

    accuracy = n_correct / float(length_of_data)
    print(f"{accuracy = }")


if __name__ == "__main__":
    initialize_model()
    train()
    test()
