# Descry OCR

Welcome to Descry, a project that brings optical character recognition technology to rare writing systems!

Currently, the project supports three alphabets: **Adlam**, **N'Ko**, and **Kayah Li**. Additional scripts may be included in the future.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#overview">Overview</a>
      <ul>
        <li><a href="#name">Name</a></li>
        <li><a href="#methodology">Methodology</a></li>
        <ul>
            <li><a href="#datasets">Datasets</a></li>
            <li><a href="#models">Models</a></li>
        </ul>
        <li><a href="#filter-visualizations">Filter Visualizations</a></li>
      </ul>
    </li>
    <li>
      <a href="#supplementary-materials">Supplementary Materials</a>
    </li>
    <li>
      <a href="#installation-guide">Installation Guide</a>
    </li>
  </ol>
</details>

## Overview

### Name

> **descry** verb
>
> 1. **a:** catch sight of <br>
>    **b:** find out, discover
> 2. *(obsolete)* make known, reveal
>
> **Etymology (sense 1)**: ultimately from Latin *dēscrībere* 'describe', from *dē-* + *scrībere* 'write'.
> <div align="right"> <a href="https://www.merriam-webster.com/dictionary/descry">Merriam-Webster</a> </div>

This name was chosen because its meaning related to seeing, together with its origins in the Latin verb 'to write', make it perfect for a project that applies computer vision to writing systems.

### Methodology

#### Datasets

Currently, the project focuses on the recognition of individual printed characters, excluding digits, diacritics, and supplemental characters used only in loanwords (if present in the writing system).

For each writing system, the primary dataset consisted of 1200-1400 labeled 28x28 character images collected from the Internet (40-50 images per character). Within a writing system, all characters were equally represented. Wherever possible, the datasets accounted for italic and non-italic characters, various degrees of bolding, various relative sizes and positions of the character with respect to the image frame, differences in character shapes due to different fonts, and 'light' versus 'dark' themes (dark character on light background or vice versa). To extract characters from raw images and convert them into a 28x28 format, we used <a href="https://www.imageresizeonline.com/convert-image-to-28x28-pixels.php">ImageResizeOnline.com</a>.

<div align="center">
  	<img src="../assets/lowercase-0061.jpg" width="70"/>
	<img src="../assets/lowercase-0285.jpg" width="70"/>
	<img src="../assets/lowercase-0425.jpg" width="70"/>
	<img src="../assets/lowercase-0481.jpg" width="70"/>
</div>
<div align="center">
	Fig. 1. Variants of the same symbol (lowercase 'b') from the Adlam dataset.
</div>

<br>

Making the datasets suitable for training required additional preprocessing steps. First, because the color of the character or background is irrelevant to the classification task, colored images were converted into grayscale. Next, each dataset was augmented by a factor of 50 by rotating images by angles between -10° and 10°, translating them by at most 2 pixels up, down, left, or right, and scaling them by fractions of the original size between 0.93 and 1.07. Finally, each dataset was randomly shuffled, and 80% was used for training while the remaining 20% was used for evaluation.

<div align="center">
  	<img src="../assets/example.png" width="200"/>
</div>
<div align="center">
	Fig. 2. A sample image from an augmented dataset (Adlam, uppercase 's').
</div>

#### Models

We trained a family of CNNs (one per alphabet) with the same general architecture inspired by <a href="https://www.jstage.jst.go.jp/article/transinf/E106.D/7/E106.D_2022EDL8098/_pdf/-char/en">EnsNet</a>. Compared to EnsNet, our models are simplified (most importantly lacking subnetworks) but have more output classes (corresponding to the number of letters in the alphabet). All models were trained on 10 epochs with batch size set to 64.

<div align="center">

<table style="border: 1px solid; text-align: center;" rules="all">
	<tr>
		<td><div align="center">Input: 28x28 character image</div></td>
	</tr>
	<tr>
		<td><div align="center">Conv3-64 <br> BatchNormalization <br> Dropout(0.35) <br> Conv3-128 <br> BatchNormalization <br> Dropout(0.35) <br> Conv3-256 <br> BatchNormalization</div></td>
	</tr>
	<tr>
		<td><div align="center">maxpool(2x2)</div></td>
	</tr>
	<tr>
		<td><div align="center">Dropout(0.35) <br> Conv3-512 <br> BatchNormalization <br> Dropout(0.35) <br> Conv3-1024 <br> BatchNormalization</div></td>
	</tr>
	<tr>
		<td><div align="center">maxpool(2x2)</div></td>
	</tr>
	<tr>
		<td><div align="center">Dropout(0.35)</div></td>
	</tr>
	<tr>
		<td><div align="center">FC-512 <br> BatchNormalization <br> Dropout(0.35)</div></td>
	</tr>
	<tr>
		<td><div align="center">Flatten</div></td>
	</tr>
	<tr>
		<td><div align="center">FC-num_classes + softmax</div></td>
	</tr>
</table>

Fig. 3. The structure of the CNNs.

</div>

### Filter Visualizations

We used gradient ascent (code adapted from <a href="https://keras.io/examples/vision/visualizing_what_convnets_learn/">keras.io</a>) to generate artificial images that maximize filter activations. We observed that patterns in those artificial images depend on the index of the corresponding filter within its convolutional layer.

Many images exhibit patterns that seem random; filters that yield random patterns are interspersed between those yielding interpretable patterns.

<div align="center">
  	<img src="../assets/nko-64-42.png" width="200"/>
</div>
<div align="center">
	Fig. 4. Input that maximizes the response of filter 42 in layer Conv3-64 (N'Ko CNN): an example of a random pattern.
</div>

<br>

One of the most common patterns is diagonal lines. This pattern tends to correspond to filters at index > 70 in their layers and is especially frequent in the Adlam CNN.

<div align="center">
  	<img src="../assets/1024-337.png" width="200"/>
</div>
<div align="center">
	Fig. 5. Input that maximizes the response of filter 337 in layer Conv3-1024 (Adlam CNN): an example of the diagonal line pattern.
</div>

<br>

Patterns get more sophisticated as index increases.

<div align="center">
  	<img src="../assets/nko-128-117.png" width="200"/>
	<img src="../assets/1024-462.png" width="200"/>
	<img src="../assets/kayahli-1024-677.png" width="200"/>
</div>
<div align="center">
	Fig. 6. Selected images displaying complex patterns: filter 117 in layer Conv3-128 (N'Ko CNN), filter 462 in layer Conv3-1024 (Adlam CNN), filter 677 in layer Conv3-1024 (Kayah Li CNN).
</div>

<br>

Filters located at the same index in their respective layers yield very similar patterns.

<div align="center">
  	<img src="../assets/64-37.png" width="150"/>
  	<img src="../assets/128-37.png" width="150"/> 
  	<img src="../assets/256-37.png" width="150"/>
	<img src="../assets/512-37.png" width="150"/>
	<img src="../assets/1024-37.png" width="150"/>
</div>
<div align="center">
	Fig. 7. Similar across all layers: Adlam CNN, filter 37 in layers Conv3-[64, 128, 256, 512, 1024].
</div>

<br>

<div align="center">
  	<img src="../assets/nko-64-18.png" width="150"/>
  	<img src="../assets/nko-128-18.png" width="150"/> 
  	<img src="../assets/nko-256-18.png" width="150"/>
	<img src="../assets/nko-512-18.png" width="150"/>
	<img src="../assets/nko-1024-18.png" width="150"/>
</div>
<div align="center">
	Fig. 8. Similar across most but not all layers: N'Ko CNN, filter 18 in layers Conv3-[64, 128, 256, 512, 1024].
</div>

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
