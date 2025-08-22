import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import utils
from torch.utils.data import TensorDataset, DataLoader
from ensemble_vote import test_ensemble
import sys
from convex_adversarial import robust_loss
from torch.autograd import Variable
import os
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
cudnn.benchmark = True

def load_lp_models():
    """
    Loads pre-trained LP (Linear Programming) robust models from specified paths.
    
    Returns:
        Dictionary of loaded and evaluated models with their names as keys.
    """
    MODEL_CONFIGS = {
        'resnet_8px': 'models_scaled/cifar_resnet_8px.pth',
        'resnet_2px': 'models_scaled/cifar_resnet_2px.pth',
        'large_8px':  'models_scaled/cifar_large_8px.pth',
        'large_2px':  'models_scaled/cifar_large_2px.pth',
        'small_8px':  'models_scaled/cifar_small_8px.pth',
        'small_2px':  'models_scaled/cifar_small_2px.pth',
    }

    models = {}
    for name, path in MODEL_CONFIGS.items():
        # Load checkpoint
        ckpt = torch.load(path)
        # Initialize appropriate model architecture
        model = utils.select_cifar_model(name)
        # Handle different checkpoint formats
        if isinstance(ckpt['state_dict'], list):
            model.load_state_dict(ckpt['state_dict'][0])
        else:
            model.load_state_dict(ckpt['state_dict'])
        model.eval()  # Set to evaluation mode
        models[name] = model
    
    return models

# Load all pre-trained models
models = load_lp_models()

# Load CIFAR-10 test dataset
_, test_loader = utils.cifar_loaders(batch_size=1, shuffle_test=False)

def evaluate_robust_and_standard_accuracy(model, images, labels, epsilon, device):
    """
    Evaluates both robust and standard accuracy of a model on given data.
    
    Args:
        model: The model to evaluate.
        images: Input images tensor.
        labels: Ground truth labels tensor.
        epsilon: Perturbation bound for robust evaluation.
        device: Device to run evaluation on.
    
    Returns:
        robust_acc: Robust accuracy percentage.
        standard_acc: Standard accuracy percentage.
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    batch_size = 1

    correct_certified = 0  # Robust accuracy numerator
    correct_standard = 0   # Standard accuracy numerator
    total = 0

    # Create DataLoader for batch processing
    data_loader = DataLoader(TensorDataset(images, labels), 
                           batch_size=batch_size, 
                           shuffle=False)
    
    # Evaluate with progress bar
    for x_batch, y_batch in tqdm(data_loader, 
                                desc="Evaluating", 
                                unit="batch"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            # Robust evaluation using convex adversarial framework
            _, robust_err = robust_loss(model, epsilon, x_batch, y_batch)
            correct_certified += (1 - robust_err) * x_batch.size(0)

            # Standard evaluation
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1)
            correct_standard += (preds == y_batch).sum().item()

        total += x_batch.size(0)

    robust_acc = 100.0 * correct_certified / total
    standard_acc = 100.0 * correct_standard / total

    return robust_acc, standard_acc

# ------------------ Batch Evaluation for All Models ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load entire test set into memory for batch processing
all_images = []
all_labels = []
for X, y in test_loader:
    all_images.append(X)
    all_labels.append(y)
all_images = torch.cat(all_images, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# Output filename
filename = "robust_std_acc_results.txt"

# Write header if file doesn't exist
if not os.path.exists(filename):
    with open(filename, "w") as f:
        f.write("Model\tEpsilon\tRobust Acc\tStandard Acc\n")

# Evaluate each model
for name, model in models.items():
    print(f"\nEvaluating model: {name}")
    
    # Set epsilon based on model name
    if "2px" in name:
        epsilon = (2.0 / 255) / 0.225  # Denormalized
        eps_display = "2/255"
    elif "8px" in name:
        epsilon = (8.0 / 255) / 0.225  # Denormalized
        eps_display = "8/255"
    else:
        raise ValueError(f"Unknown epsilon for model {name}")

    model = model.to(device)

    # Evaluate model
    robust_acc, std_acc = evaluate_robust_and_standard_accuracy(
        model, all_images, all_labels, epsilon, device
    )

    # Append results to file
    with open("robust_std_acc_results.txt", "a") as f:
        f.write(f"{name}\t{eps_display}\t{robust_acc:.4f}\t{std_acc:.4f}\n")
    
    print(f"Results saved: Robust Acc = {robust_acc:.2f}%, Standard Acc = {std_acc:.2f}%")

print("\nEvaluation completed. Results saved to robust_std_acc_results.txt")