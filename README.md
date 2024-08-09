# Descry OCR

Welcome to Descry, a project that brings optical character recognition technology to rare writing systems!

Currently, the project supports two alphabets: **Adlam** and **Kayah Li**. More coming soon.

The OCR engine uses <a href="https://github.com/clovaai/CRAFT-pytorch">CRAFT</a> to detect words on an image and CRNN to recognize individual words; see the <a href="dariakryvosheieva.github.io/pdfs/descry-project-report.pdf">technical report</a> for details. Model training code is available under the repo's `training` branch.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#name">Name</a></li>
    <li><a href="#app-overview">App Overview</a></li>
    <li><a href="#installation-guide">Installation Guide</a></li>
  </ol>
</details>

## Name

> **descry** verb
>
> 1. **a:** catch sight of <br>
>    **b:** find out, discover
> 2. *(obsolete)* make known, reveal
>
> **Etymology (sense 1)**: ultimately from Latin *dēscrībere* 'describe', from *dē-* + *scrībere* 'write'.
> <div align="right"> <a href="https://www.merriam-webster.com/dictionary/descry">Merriam-Webster</a> </div>

This name was chosen because its meaning related to seeing, together with its origins in the Latin verb 'to write', make it perfect for a project that applies computer vision to writing systems.

## App Overview

Use the menu on the home page to select a writing system and upload an image containing text. The OCR engine will recognize text on the image and print it in the "Model prediction" box.

<p align="center">
  <kbd><img src="../assets/app_demo.png" width="600px"></kbd>
</p>

## Installation Guide

### 1. Install the app on your local machine

Clone the repo's `main` branch by pasting the following into your command prompt:
```shell
git clone -b main --single-branch https://github.com/dariakryvosheieva/descry-ocr.git
```

### 2. Run the app

To install required dependencies, open the command prompt in the **descry-ocr** folder and run
```shell
pip install -r requirements.txt
```
Next, run
```shell
python -m flask --app app run
```
After a few seconds, you will see `Running on http://127.0.0.1:5000`. Follow the link to open the web app in your browser.
