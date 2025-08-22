import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import utils
from autoattack import AutoAttack
import torch.nn as nn

# Argument parser for configuration
parser = argparse.ArgumentParser(description='Run AutoAttack on standard or LP models with multiple epsilon values.')
parser.add_argument('--output-root', type=str, default='./', help='Root directory for saving adversarial examples')
parser.add_argument('--models-type', type=str, choices=['std', 'lp'], default='std', help='Type of models to attack: standard or LP-trained')
parser.add_argument('--mode', type=str, choices=['single', 'ensemble'], default='single', help='Attack mode: single model or ensemble')
# parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run attack on')
args = parser.parse_args()

# Predefined epsilon values and corresponding directory names
EPSILON_CONFIGS = {
    '2px': 2.0 / 255,
    '4px': 4.0 / 255,
    '6px': 6.0 / 255,
    '8px': 8.0 / 255
}

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Load appropriate models based on type
if args.models_type == 'std':
    models = utils.load_std_models()
    output_dir_prefix = 'adv_outputs_STD'
elif args.models_type == 'lp':
    models = utils.load_lp_models()
    output_dir_prefix = 'adv_outputs_LP'
else:
    raise ValueError("Unknown model type!")

# Load CIFAR-10 test data without normalization (for attack)
_, test_loader = utils.cifar_loaders_no_normalizred(batch_size=100, shuffle_test=False)

# Concatenate all test images and labels
all_images = []
all_labels = []
for images, labels in test_loader:
    all_images.append(images)
    all_labels.append(labels)

images = torch.cat(all_images).to(device)
labels = torch.cat(all_labels).to(device)

# Run attacks for each epsilon value
for eps_name, epsilon in EPSILON_CONFIGS.items():
    # Create output directory for this epsilon
    output_dir = os.path.join(args.output_root, f"{output_dir_prefix}_{eps_name}")
    print(f"\n{'='*60}")
    print(f"Processing epsilon: {eps_name} ({epsilon:.6f})")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    # Single model attack mode
    if args.mode == 'single':
        for name, model in models.items():
            print(f'\nRunning AutoAttack on model: {name} with epsilon {eps_name}')
            model = model.to(device)
            model.eval()

            # Initialize AutoAttack adversary
            adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
            
            # Generate adversarial examples
            adv_images = adversary.run_standard_evaluation(images, labels, bs=100)

            # Save adversarial examples and original data
            model_save_dir = os.path.join(output_dir, name)
            os.makedirs(model_save_dir, exist_ok=True)
            np.save(os.path.join(model_save_dir, 'adv_images.npy'), adv_images.cpu().numpy())
            np.save(os.path.join(model_save_dir, 'original_images.npy'), images.cpu().numpy())
            np.save(os.path.join(model_save_dir, 'labels.npy'), labels.cpu().numpy())

            with torch.no_grad():
                preds = model(adv_images).argmax(dim=1)
                acc = (preds == labels).float().mean().item()
            print(f'[{name}] Accuracy under attack: {acc * 100:.2f}%')

    # Ensemble attack mode
    elif args.mode == 'ensemble':
        print(f'\nRunning AutoAttack on ensemble model with epsilon {eps_name}')
        
        # Create ensemble model by averaging logits
        ensemble_model = utils.AverageLogitModel(list(models.values())).to(device)
        ensemble_model.eval()

        # Initialize AutoAttack adversary for ensemble
        adversary = AutoAttack(ensemble_model, norm='Linf', eps=epsilon, version='standard')
        
        # Generate adversarial examples against ensemble
        adv_images = adversary.run_standard_evaluation(images, labels, bs=100)

        # Save adversarial examples and original data
        ensemble_save_dir = output_dir + "_surrogate"
        os.makedirs(ensemble_save_dir, exist_ok=True)
        np.save(os.path.join(ensemble_save_dir, 'adv_images.npy'), adv_images.cpu().numpy())
        np.save(os.path.join(ensemble_save_dir, 'original_images.npy'), images.cpu().numpy())
        np.save(os.path.join(ensemble_save_dir, 'labels.npy'), labels.cpu().numpy())

        # Calculate and print ensemble accuracy under attack
        with torch.no_grad():
            preds = ensemble_model(adv_images).argmax(dim=1)
            acc = (preds == labels).float().mean().item()
        print(f'[Ensemble-{args.models_type}] Accuracy under attack ({eps_name}): {acc * 100:.2f}%')

    else:
        raise ValueError("Unknown mode!")

print(f"\n{'='*60}")
print("All attacks completed! Results saved in the following directories:")
for eps_name in EPSILON_CONFIGS.keys():
    if args.mode == 'ensemble':
        dir_name = f"{output_dir_prefix}_{eps_name}_surrogate"
        print(f"  - {os.path.join(args.output_root, dir_name)}")
    else:
        dir_name = f"{output_dir_prefix}_{eps_name}"
        print(f"  - {os.path.join(args.output_root, dir_name)}")        
print(f"{'='*60}")