# Generic-Framework-for-Modelling-PUFs
This repository contains the official implementation of the following paper:
> I Don’t Know What it is, but I Know I Can Break it: Attacking PUFs with Minimal Adversarial Knowledge

## Abstract
 Physically Unclonable Functions (PUFs) provide a streamlined solution for lightweight device authentication. Delay-based Arbiter PUFs, with their ease of implementation and vast challenge space, have received significant attention; however, they are not immune to modelling attacks that exploit correlations between their inputs and outputs. Research is therefore polarized between developing modelling-resistant PUFs and devising machine learning attacks against them. This dichotomy often results in exaggerated concerns and overconfidence in PUF security, primarily because there lacks a universal tool to gauge a PUF's security. In many scenarios, attacks require additional information, such as PUF type or configuration parameters. Alarmingly, new PUFs are often branded `secure' if they lack a specific attack model upon introduction. To impartially assess the security of delay-based PUFs, we present a versatile framework featuring a Mixture-of-PUF-Experts (MoPE) layer for mounting attacks on various PUFs with minimal adversarial knowledge. We demonstrate the capability of our model to attack different PUF types, including the \textit{first} successful attack on Heterogeneous Feed-Forward PUFs using only a reasonable amount of challenges and responses. Our proposed framework makes use of a Multi-gate Mixture-of-PUF-Experts (MMoPE) layer, facilitating multi-task learning across diverse PUFs to recognise commonalities across PUF designs. This allows a streamlining of training periods for attacking multiple PUFs simultaneously. We conclude by showcasing the potent performance of MoPE and MMoPE across a spectrum of PUF types, employing simulated, real-world unbiased, and biased data sets for analysis.

## Prerequisties
- Software
    - python 3.8
    - tensorflow 2.4
    - NVIDIA-SMI 536.99
    - CUDA 12.2
- Hardware Implementation
    - XC7Z010 FPGA
    - PYNQ

## Installation
```
conda env create -f environment.yml
```

## Usage
This repository provides jupyter notebook (.ipynb) with clear instructions.

>MoPE.ipynb: Mixture-of-PUF-Expert layer of single task for modelling any kinds of delay-based PUFs.

>MMoPE.ipynb: Multiple-Mixture-of-PUF-Expert layer of multiple tasks for modelling combinations of different delay-based PUFs.

>Hardware Implementation: bitstream file and jupyter notebook for PS control (PYNQ).
<!-- ## Results -->

## Acknowledgments
The code is built upon the [work](https://github.com/eminorhan/mixture-of-experts) by Emin Orhan and [work](https://github.com/drawbridge/keras-mmoe) by Deng.