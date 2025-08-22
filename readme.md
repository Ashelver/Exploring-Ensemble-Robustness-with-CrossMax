# Exploring Ensemble Robustness with CrossMax

This repository contains the code for the empirical study: **"Exploring Ensemble Robustness with CrossMax: From Provable Defenses to Confidence Analysis"**.

## Abstract

Adversarial robustness remains a critical challenge in deep learning. This project conducts an extensive empirical study on ensemble robustness against strong white-box adversarial attacks, focusing on ensembles constructed from models trained via standard procedures and Linear Programming (LP). 

We introduce and evaluate **CrossMax**, a novel ensemble aggregation technique that produces set-valued predictions to better capture uncertainty under attack, and propose a confidence measurement **CrossMax Confidence**. Our experiments on CIFAR-10 reveal that LP-trained ensembles consistently outperform standard-trained ones. Notably, CrossMax achieves about **5% to 15% robustness accuracy improvements** compared to traditional voting schemes, especially under large threat perturbation budgets. We show that CrossMax Confidence better reflects the ensemble’s sensitivity to adversarial perturbations and can serve as a practical indicator of robustness reliability.

This work provides insights bridging the gap between certified and empirical defenses and highlights the potential of set-based ensemble predictions.

## Dependencies

The core requirements are listed in `requirements.txt`.

## Attribution
This project builds upon code and pre-trained models from the following repository:

- locuslab/convex_adversarial ([Github Link](https://github.com/locuslab/convex_adversarial.git))

    - The entire convex_adversarial/ directory and a modified utils.py are used from this source.

    - Pre-trained models located in the models_scaled/ directory are provided by them.

    - This external code is used under its original license. Please see the respective files for details.


## Usage

1. Train the Standard-trained models(`train_std.py`):
    ```bash
    python train_std.py --model small --epochs 20 --save_path models_standard/cifar_small_std.pth
    python train_std.py --model large --epochs 20 --save_path models_standard/cifar_large_std.pth
    python train_std.py --model resnet --epochs 20 --save_path models_standard/cifar_resnet_std.pth
    ```
    The script will print training loss and accuracy for each epoch, as well as test accuracy. Model checkpoints will be saved every 10 epochs with the naming pattern: [save_path]_epoch[number].pth

2. Evaluating LP Robust Models(`robust_evaluate.py`):
    ```bash
    python eval_lp_models.py
    ```
    This script evaluates the robust and standard accuracy of pre-trained Linear Programming (LP) robust models on the CIFAR-10 test set.

3. AutoAttack Evaluation(`adv_generate.py`):
    ```bash
    python adv_generate.py --models_type std --mode single
    python adv_generate.py --models_type std --mode ensemble
    python adv_generate.py --models_type lp --mode single
    python adv_generate.py --models_type lp --mode ensemble
    ```

4. Ensemble Methods Test(`ensemble_test.py`):
    ```bash
    python ensemble_test.py --models-type std --test-type nat
    python ensemble_test.py --models-type std --test-type adv
    python ensemble_test.py --models-type lp --test-type nat
    python ensemble_test.py --models-type lp --test-type adv
    ```
    This script (`ensemble_test.py`) evaluates ensemble models on both natural and adversarial examples, providing performance metrics and diversity analysis.