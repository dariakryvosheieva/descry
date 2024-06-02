# Descry OCR

Welcome to Descry, a project that brings optical character recognition technology to rare writing systems!

Currently, the project supports three alphabets: **Adlam**, **N'Ko**, and **Kayah Li**. Additional scripts may be included in the future.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
	<a href="#name">Name</a>
    </li>
    <li>
	<a href="#name">App Demo</a>
    </li>
    <li>
      <a href="#supplementary-materials">Supplementary Materials</a>
    </li>
    <li>
      <a href="#installation-guide">Installation Guide</a>
    </li>
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

## App Demo

Use the menu on the home page to select a writing system.

<p align="center">
  <kbd><img src="../assets/Screenshot 2024-06-02 145702.png" width="600px"></kbd>
</p>

You will see an interface to upload an image and get the model prediction, along with basic information about the script.

<p align="center">
  <kbd><img src="../assets/Screenshot 2024-06-02 151423.png" width="600px"></kbd>
</p>

## Supplementary Materials

<table>
  <tr>
    <td></td>
    <th scope="col">Adlam</th>
    <th scope="col">N'Ko</th>
    <th scope="col">Kayah Li</th>
  </tr>
  <tr>
    <th scope="row">Dataset before augmentation</th>
    <td><a href="../assets/adlam_raw.zip">adlam_raw.zip</a></td>
    <td><a href="../assets/nko_raw.zip">nko_raw.zip</a></td>
    <td><a href="../assets/kayahli_raw.zip">kayahli_raw.zip</a></td>
  </tr>
  <tr>
    <th scope="row">Dataset after augmentation</th>
    <td><a href="https://drive.google.com/file/d/19TsTVjOTMvAs_5pXGkBRRIP01ObnAW7y/view?usp=drive_link">adlam_augmented.npy</a></td>
    <td><a href="https://drive.google.com/file/d/1GTorzYHArB6JkXRiYWqOaysVHus_Q422/view?usp=drive_link">nko_augmented.npy</a></td>
    <td><a href="https://drive.google.com/file/d/1GJ06DbU7_05NngvNzXV-_XCCJsnDwcb-/view?usp=drive_link">kayahli_augmented.npy</a></td>
  </tr>
  <tr>
    <th scope="row">Code for training the model</th>
    <td><a href="https://colab.research.google.com/drive/1c8dMSP5c98caC9wTwaO5fkXhClkuQKBj?usp=sharing">adlam_cnn.ipynb</a></td>
    <td><a href="https://colab.research.google.com/drive/1-1xravE86dtpqv6wSyXui3SuhBPRaJ49?usp=sharing">nko_cnn.ipynb</a></td>
    <td><a href="https://colab.research.google.com/drive/1OssGzEgzO5MtJJ4Mumq8us9wa92S3ARN?usp=sharing">kayahli_cnn.ipynb</a></td>
  </tr>
  <tr>
    <th scope="row">Code for filter visualizations</th>
    <td><a href="https://colab.research.google.com/drive/1v9j__6EL1Dce4vhHVSvguJG_07taY-HE?usp=sharing">adlam_filter_visualizations.ipynb</a></td>
    <td><a href="https://colab.research.google.com/drive/11A47dnhfZox6cMPqnOBxvy0MfAY0TPBK?usp=sharing">nko_filter_visualizations.ipynb</a></td>
    <td><a href="https://colab.research.google.com/drive/1cYxxEYHyHUVhNOkeSwGfIxG2cPTu-u7_?usp=sharing">kayahli_filter_visualizations</a></td>
  </tr>
</table>

For model weights - see the 'models' folder in the repo.

## Installation Guide

### 1. Install the repository on your local machine

This can be done in two ways. If you have Git, clone the repo by pasting the following into your command prompt:
```shell
git clone https://github.com/dariakryvosheieva/descry.git
```
Otherwise, click `<> Code` > `Download ZIP` and then unzip the folder.

### 2. Run the app

To install required dependencies, open the command prompt **in the folder** and run
```shell
pip install -r requirements.txt
```
Next, run
```shell
python -m flask --app app run
```
After a few seconds, you will see `Running on http://127.0.0.1:5000`. Follow the link to open the web app in your browser.
