# Descry OCR

Welcome to Descry, a project that brings optical character recognition technology to rare writing systems!

Currently, the project supports three alphabets: **Adlam**, **N'Ko**, and **Kayah Li**. OCR models for these alphabets have not existed before. Additional scripts may be included in the future.

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
            <li><a href="models">Models</a></li>
        </ul>
        <li><a href="#interpretability">Interpretability</a></li>
      </ul>
    </li>
    <li>
      <a href="#installation-guide">Installation Guide</a>
    </li>
  </ol>
</details>

<!-- OVERVIEW -->
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

So far, the project focuses on the recognition of individual printed characters, excluding digits, diacritics, and supplemental characters used only in loanwords (if present in the writing system).

The primary datasets consisted of 1200-1400 labeled 28x28 character images collected from the Internet (40-50 images per character). Within a writing system, all characters were equally represented. Wherever possible, the datasets accounted for italic and non-italic characters, various degrees of bolding, various relative sizes and positions of the character with respect to the image frame, differences in character shapes due to fonts, and 'light' versus 'dark' themes (dark character on light background or vice versa). To extract characters from raw images and convert them into a 28x28 format, we used <a href="https://www.imageresizeonline.com/convert-image-to-28x28-pixels.php">ImageResizeOnline.com</a>.

Making the datasets suitable for training required additional preprocessing steps. First, because the color of the character or background is irrelevant to the classification task, colored images were converted into grayscale. Next, each dataset was augmented by a factor of 50 by rotating images by angles between -10° and 10°, translating them by at most 2 pixels up, down, left, or right, and scaling them by fractions of the original size between 0.93 and 1.07. Finally, each dataset was randomly shuffled, and 80% was used for training while the remaining 20% was used for evaluation.

#### Models

We trained a family of CNNs with the same general architecture inspired by <a href="https://www.jstage.jst.go.jp/article/transinf/E106.D/7/E106.D_2022EDL8098/_pdf/-char/en">EnsNet</a>. Compared to EnsNet, our models are simplified (most importantly lacking subnetworks) but have more output classes (corresponding to the number of letters in the alphabet). All models were trained on 10 epochs with batch size set to 64.

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

The structure of the CNNs

</div>

### Interpretability

We used gradient ascent (code adapted from <a href="https://keras.io/examples/vision/visualizing_what_convnets_learn/">keras.io</a>) to generate artificial images that maximize filter activations.

Filters with random activation patterns are interspersed between those with interpretable patterns.

<div align="center">
  	<img src="../assets/nko-64-42.png" width="150" />
</div>
<div align="center">
	N'Ko CNN, Conv3-64-42: an example of a filter with a random activation pattern.
</div>

<br>

For filters with small ids within a layer, the dominant interpretable pattern is high activation along the borders and low activation in the middle.

<div align="center">
  	<img src="../assets/kayahli-64-11.png" width="150" />
	<img src="../assets/kayahli-128-26.png" width="150" />
	<img src="../assets/kayahli-256-45.png" width="150" />
</div>
<div align="center">
	Kayah Li CNN, Conv3-[64-11, 128-26, 256-45]
</div>

<br>

For higher filter ids, the diversity of patterns increases. One common pattern is diagonal lines; these occur especially frequently in the Adlam CNN.

<div align="center">
  	<img src="../assets/1024-337.png" width="150" />
</div>
<div align="center">
	Adlam CNN, Conv3-1024-337: an example of the diagonal line pattern.
</div>

<br>

Filters with the same index within a layer yield very similar activation patterns across all or most layers.

<div align="center">
  	<img src="../assets/64-37.png" width="150" />
  	<img src="../assets/128-37.png" width="150" /> 
  	<img src="../assets/256-37.png" width="150" />
	<img src="../assets/512-37.png" width="150" />
	<img src="../assets/1024-37.png" width="150" />
</div>
<div align="center">
	Similar across all layers: Adlam CNN, filter 37 in layers Conv3-[64, 128, 256, 512, 1024]
</div>

<br>

<div align="center">
  	<img src="../assets/nko-64-18.png" width="150" />
  	<img src="../assets/nko-128-18.png" width="150" /> 
  	<img src="../assets/nko-256-18.png" width="150" />
	<img src="../assets/nko-512-18.png" width="150" />
	<img src="../assets/nko-1024-18.png" width="150" />
</div>
<div align="center">
	Similar across most but not all layers: N'Ko CNN, filter 18 in layers Conv3-[64, 128, 256, 512, 1024] — 1024 is different, having no diagonal lines inside
</div>

<br>

See the app for details specific to each CNN.

## Installation Guide

### 1. Install the repository on your local machine

This can be done in two ways. If you have Git, clone the repo by typing the following into your command prompt:
```shell
git clone https://github.com/dariakryvosheieva/descry.git
```
Otherwise, click `<> Code` > `Download ZIP` and then unzip the folder.

### 2. Run the app

Make sure you have Python and pip.

To install required dependencies, open the command prompt **in the folder** and type
```shell
pip install -r requirements.txt
```
Next, type
```shell
python -m flask --app app run
```
After a while, you will see something like `Running on http://127.0.0.1:5000`. Follow the link to open the web app in your browser.
