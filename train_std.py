import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils import cifar_loaders, cifar_model, cifar_model_large, cifar_model_resnet

def train(model, train_loader, test_loader, epochs, lr, device, save_path):
    """
    Trains a model on the CIFAR-10 dataset.
    
    Args:
        model: The neural network model to train.
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test set.
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        device: Device to train on ('cuda' or 'cpu').
        save_path: Base path for saving model checkpoints.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        # Print training progress
        train_acc = 100. * correct / total
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {running_loss/total:.4f}, Train Acc: {train_acc:.2f}%")

        # Evaluate on test set every epoch
        test_acc = evaluate(model, test_loader, device)
        print(f"           --> Test Acc: {test_acc:.2f}%")

        # Save checkpoint every 10 epochs (epoch 10, 20, 30...)
        if (epoch + 1) % 10 == 0 and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_file = save_path.replace(".pth", f"_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_file)
            print(f"Saved checkpoint at epoch {epoch+1} → {save_file}")

def evaluate(model, dataloader, device):
    """
    Evaluates the model's accuracy on a given dataset.
    
    Args:
        model: The neural network model to evaluate.
        dataloader: DataLoader for the evaluation dataset.
        device: Device to evaluate on ('cuda' or 'cpu').
    
    Returns:
        Accuracy percentage.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return 100. * correct / total

def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train a model on CIFAR-10.')
    parser.add_argument('--model', choices=['small', 'large', 'resnet'], default='small', help='Type of model architecture')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models_standard/cifar_std.pth', help='Path to save model checkpoints')
    parser.add_argument('--cuda', action='store_true', help='Use GPU for training if available')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = cifar_loaders(batch_size=args.batch_size)
    
    # Initialize model
    if args.model == 'small':
        model = cifar_model()
    elif args.model == 'large':
        model = cifar_model_large()
    elif args.model == 'resnet':
        model = cifar_model_resnet(N=1, factor=1)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training {args.model} model on CIFAR-10 for {args.epochs} epochs...")
    
    # Start training
    train(model, train_loader, test_loader, args.epochs, args.lr, device, args.save_path)

if __name__ == "__main__":
    main()