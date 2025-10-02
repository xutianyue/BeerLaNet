import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse
import yaml

from src.data_loader import load_data
from src.model_loader_new import load_model
# from src.model_loader_concat import load_model
from src.utils import plot_confusion_matrix
from src.utils import visualize_d_matrix


# Function to perform inference
def inference(model, dataloader, device, visual_save_path=None):
    """
    Perform inference on the model using the provided dataloader.
    Optionally, save the D-matrix visualizations to the specified path.
    """
    model.eval()
    predictions = []
    true_labels = []

    count = 0 # Counter for saving images
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Inference"):
            inputs = inputs.to(device)

            all_outputs = model(inputs)
            outputs = all_outputs[0] if isinstance(all_outputs, tuple) else all_outputs
            D_matrices = all_outputs[1] if isinstance(all_outputs, tuple) else None
    
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            ## Visualize D-matrix for the first 100 images
            if visual_save_path is not None:
                os.makedirs(visual_save_path, exist_ok=True)
                for j in range(inputs.shape[0]): # Loop through batch
                    # if count >= 100: # Only process the first 100 images
                    #     break
                    original_image = inputs[j].cpu()
                    d_matrix = D_matrices[j].cpu()
                    
                    # Use sequential naming
                    image_name = f"sample_{count}.png"
                    save_path = os.path.join(visual_save_path, image_name)
                    visualize_d_matrix(original_image, d_matrix, save_path)
                    count += 1
            ## End of visualization

    return predictions, true_labels

def main(config_path):
    # Load config file
    with open(config_path, 'r') as file:
        args = yaml.safe_load(file)

    # Set up device (single GPU)
    device_number = args['device']
    device = torch.device(f'cuda:{device_number}' if torch.cuda.is_available() else 'cpu')

    backbone = args['backbone']
    model_head = args['model_head']
    dataset_path = args['dataset_path']
    dataset_name = args['dataset_name']
    batch_size = args['batch_size']
    model_path = args['model_path']
    output_path = args['output_path']

    visualize_D = args.get('visualize_D', False)
    if visualize_D:
        visual_save_path = os.path.join(output_path, "D_matrices")
    else:
        visual_save_path = None
    
    # Load data
    _, _, test_loader, num_classes, class_names = load_data(dataset_path, dataset_name,dataset_is_testset=True, batch_size=batch_size)
    print("Test data loaded.")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Load model
    model = load_model(backbone, model_head, num_classes, device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded for inference.")

    # Perform inference
    predictions, true_labels = inference(model, test_loader, device, visual_save_path)

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 6))
    plot_confusion_matrix(true_labels, predictions, classes=class_names)
    os.makedirs(output_path, exist_ok=True)
    # plt.savefig(os.path.join(output_path, 'confusion_matrix_inference.png'))
    plt.savefig(os.path.join(output_path, f'confusion_matrix_inference_{dataset_name}.png'))
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)
