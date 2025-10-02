import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import yaml
import shutil

from src.data_loader import load_data
from src.model_loader_new import load_model
# from src.model_loader_concat import load_model
from src.utils import setup_logger, plot_confusion_matrix, plot_results, visualize_d_matrix

# Function to evaluate the model
def evaluate_model(model, dataloader, criterion, output_path, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    true_labels = []

    visualization_dir = os.path.join(output_path,"Training")
    os.makedirs(os.path.join(output_path,"Training"), exist_ok=True)
    
    with torch.no_grad():
        count = 0 # Counter for saving image
        for batch in dataloader:
            # batch is a list of length 2 or 3
            inputs, labels = batch[:2]
            inputs, labels = inputs.to(device), labels.to(device)
            # outputs, D_matrices = model(inputs)
            all_outputs = model(inputs)
            outputs = all_outputs[0] if isinstance(all_outputs, tuple) else all_outputs
            D_matrices = all_outputs[1] if isinstance(all_outputs, tuple) else None

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Visualize D-matrix for all images
            # if D_matrices is not None:

            #     for j in range(inputs.shape[0]): # Loop through batch
            #         original_image = inputs[j].cpu()
            #         d_matrix = D_matrices[j].cpu()
            #         image_name = f"sample_{count}.png"
            #         count += 1
            #         save_path = os.path.join(visualization_dir, image_name)
            #         visualize_d_matrix(original_image, d_matrix, save_path)


    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, predictions, true_labels

# Function to train the model
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device, logger):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            # batch is a list of length 2 or 3
            if len(batch) == 2:
                inputs, labels = batch
            elif len(batch) == 3:
                inputs, labels, metadata = batch # metadata is not used in this example
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            all_outputs = model(inputs)
            outputs = all_outputs[0] if isinstance(all_outputs, tuple) else all_outputs

            loss = criterion(outputs, labels)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                # batch is a list of length 2 or 3
                if len(batch) == 2:
                    inputs, labels = batch
                elif len(batch) == 3:
                    inputs, labels, metadata = batch # metadata is not used in this example
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                inputs, labels = inputs.to(device), labels.to(device)

                all_outputs = model(inputs)
                outputs = all_outputs[0] if isinstance(all_outputs, tuple) else all_outputs

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate loss and accuracy
        train_epoch_loss = train_loss / len(train_loader)
        train_epoch_acc = 100. * train_correct / train_total
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100. * val_correct / val_total

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        train_accuracies.append(train_epoch_acc)
        val_accuracies.append(val_epoch_acc)

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_epoch_loss}, Train Accuracy: {train_epoch_acc}%, Val Loss: {val_epoch_loss}, Val Accuracy: {val_epoch_acc}%")

    return model, train_losses, val_losses, train_accuracies, val_accuracies

def main(config_path):
    # Load config file
    with open(config_path, 'r') as file:
        args = yaml.safe_load(file)
    os.makedirs(args['output_path'], exist_ok=True) # create output directory if it doesn't exist
    shutil.copy(config_path, os.path.join(args['output_path'], os.path.basename(config_path))) # copy yaml file to output path

    # Set seed
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    # Setup logger
    logger = setup_logger(args['output_path'])
    # start logging
    logger.info("=" * 50)
    logger.info(f"Training {args['backbone']} model on {args['dataset_name']} dataset")
    logger.info("=" * 50)

    #devices = parse_device(args['device'])
    devices = args['device']
    if not isinstance(devices, list):
        devices = [devices]  # Convert to list for compatibility with multi-GPU code
    multi_gpu = len(devices) > 1
    device = torch.device(f'cuda:{devices[0]}')  # Use first device for as main device

    # Load data
    # dataset_name: 'camelyon17-wilds'/'bbbc'/'crops' / 'test'
    train_loader, val_loader, test_loader, num_classes, class_names = load_data(args['dataset_path'], args['dataset_name'], False, args['batch_size'], args['num_workers'])
    logger.info(f"Loaded {args['dataset_name']} dataset.")

    model = load_model(args['backbone'], args['model_head'], num_classes, device)
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=devices)
    model = model.to(device)

    # Define loss function and optimizer
    if args['criterion'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Criterion {args['criterion']} not supported")
    
    if args['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    else:
        raise ValueError(f"Optimizer {args['optimizer']} not supported")

    start_time = time.time() # Start time
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, criterion, optimizer, train_loader, val_loader, args['num_epochs'], device, logger)

    # Record end time and calculate training duration
    end_time = time.time()
    logger.info("=" * 50)
    logger.info(f"\nTraining completed in {(end_time - start_time) / 3600:.2f} hours")
    logger.info("=" * 50)

    # Save the model
    model_save_name = f"{args['backbone']}_{args['model_head']}.pth"
    save_path = os.path.join(args['output_path'], model_save_name)

    # If model is wrapped in DataParallel, save the inner model
    if hasattr(trained_model, 'module'):
        torch.save(trained_model.module.state_dict(), save_path)
    else:
        torch.save(trained_model.state_dict(), save_path)

    # Plot and save results
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies, args['output_path'])
    
    # Evaluate on test set and save confusion matrix
    test_loss, test_acc, test_predictions, test_true_labels = evaluate_model(trained_model, test_loader, criterion, args['output_path'], device)
    
    plt.figure(figsize=(6, 6))
    test_relaxed_acc = plot_confusion_matrix(test_true_labels, test_predictions, classes=class_names, title='Confusion Matrix - Test')
    plt.savefig(os.path.join(args['output_path'], 'confusion_matrix_test.png'))
    plt.show()
    logger.info(f"Test Overall Accuracy: {test_acc:.2f}%, Test Relaxed Accuracy: {test_relaxed_acc:.2f}%")

    # Evaluate on the validation set and save confusion matrix
    val_loss, val_acc, val_predictions, val_true_labels = evaluate_model(trained_model, val_loader, criterion, args['output_path'], device)

    plt.figure(figsize=(6, 6))
    val_relaxed_acc = plot_confusion_matrix(val_true_labels, val_predictions, classes=class_names, title='Confusion Matrix - Val')
    plt.savefig(os.path.join(args['output_path'], 'confusion_matrix_val.png'))
    plt.show()
    logger.info(f"Validation Overall Accuracy: {val_acc:.2f}%, Validation Relaxed Accuracy: {val_relaxed_acc:.2f}%")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)

