import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import utils
from torch.utils.data import TensorDataset, DataLoader
from ensemble_vote import test_ensemble, evaluate_diversity_confidence
import argparse
import os

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
cudnn.benchmark = True

# Argument parser for configuration
parser = argparse.ArgumentParser(description='Evaluate ensemble models on natural or adversarial examples.')
parser.add_argument('--models-type', type=str, choices=['std', 'lp'], default='std', 
                   help='Type of models to evaluate: standard (std) or LP-trained (lp)')
parser.add_argument('--test-type', type=str, choices=['nat', 'adv'], default='adv', 
                   help='Type of test data: natural (nat) or adversarial (adv)')
parser.add_argument('--batch-size', type=int, default=100, 
                   help='Batch size for evaluation')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                   help='Device to run evaluation on')
args = parser.parse_args()

# Define directory configurations based on model type and test type
if args.models_type == 'std':
    models = utils.load_std_models()
    adv_directories = [
        './adv_outputs_STD_2px_surrogate',
        './adv_outputs_STD_4px_surrogate',
        './adv_outputs_STD_6px_surrogate',
        './adv_outputs_STD_8px_surrogate'
    ]
elif args.models_type == 'lp':
    models = utils.load_lp_models()
    adv_directories = [
        './adv_outputs_LP_2px_surrogate',
        './adv_outputs_LP_4px_surrogate',
        './adv_outputs_LP_6px_surrogate',
        './adv_outputs_LP_8px_surrogate'
    ]
else:
    raise ValueError("Unknown model type!")

# Move models to the specified device
for name, model in models.items():
    models[name] = model.to(args.device)

print(f"Evaluating {args.models_type.upper()} models on {args.test_type.upper()} data")
print(f"Using device: {args.device}")
print(f"Number of models: {len(models)}")

# --------- Load and evaluate test data ---------
if args.test_type == 'adv':
    # Evaluate on all adversarial directories
    for adv_dir in adv_directories:
        print(f"\n{'='*60}")
        print(f"Evaluating on adversarial examples from: {adv_dir}")
        print(f"{'='*60}")
        
        # Check if directory exists
        if not os.path.exists(adv_dir):
            print(f"Warning: Directory {adv_dir} does not exist. Skipping...")
            continue
            
        # Load adversarial examples and labels
        adv_images_path = os.path.join(adv_dir, 'adv_images.npy')
        labels_path = os.path.join(adv_dir, 'labels.npy')
        
        if not os.path.exists(adv_images_path) or not os.path.exists(labels_path):
            print(f"Warning: Required files not found in {adv_dir}. Skipping...")
            continue
            
        try:
            adv_images = np.load(adv_images_path)
            labels = np.load(labels_path)
            
            # Convert to tensors and move to device
            adv_images = torch.tensor(adv_images, dtype=torch.float32).to(args.device)
            labels = torch.tensor(labels, dtype=torch.long).to(args.device)
            
            # Create DataLoader
            adv_dataset = TensorDataset(adv_images, labels)
            adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Evaluate ensemble performance
            print("Ensemble performance:")
            test_ensemble(models=models, test_loader=adv_loader)
            
            # Evaluate diversity and confidence
            print("\nDiversity and confidence analysis:")
            evaluate_diversity_confidence(models=models, test_loader=adv_loader)
            
        except Exception as e:
            print(f"Error processing {adv_dir}: {e}")
            continue

elif args.test_type == "nat":
    # Evaluate on natural (clean) test data
    print(f"\n{'='*60}")
    print("Evaluating on natural (clean) test data")
    print(f"{'='*60}")
    
    _, test_loader = utils.cifar_loaders_no_normalizred(batch_size=args.batch_size, shuffle_test=False)
    
    # Evaluate ensemble performance
    print("Ensemble performance on natural data:")
    test_ensemble(models=models, test_loader=test_loader)
    
    # Evaluate diversity and confidence
    print("\nDiversity and confidence analysis on natural data:")
    evaluate_diversity_confidence(models=models, test_loader=test_loader)

else:
    raise ValueError("Unknown test type!")

print(f"\n{'='*60}")
print("Evaluation completed!")
print(f"{'='*60}")