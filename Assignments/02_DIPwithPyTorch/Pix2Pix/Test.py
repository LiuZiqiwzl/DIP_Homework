import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.
    """
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    return image

def validate_model(model, dataloader, device):
    """
    Validate the model on the validation dataset and save comparison images.
    """
    model.eval()

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Convert tensors to images
            input_img_np = tensor_to_image(image_rgb[0])
            target_img_np = tensor_to_image(image_semantic[0])
            output_img_np = tensor_to_image(outputs[0])

            # Concatenate the images and save
            comparison = np.hstack((input_img_np, target_img_np, output_img_np))
            cv2.imwrite(f'validation_results/comparison_{i + 1}.png', comparison)

            # Optionally, display images (comment out if running in a non-GUI environment)
            cv2.imshow('Comparison', comparison)
            cv2.waitKey(0)  # Wait for a key press to display the next image

    cv2.destroyAllWindows()

def load_and_preprocess_image(image_path):
    """
    Load and preprocess the input image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    #image = cv2.resize(image, (256, 256))  # Resize to match model input size
    image = (image / 255.0) * 2 - 1  # Normalize to [-1, 1]
    image = np.transpose(image, (2, 0, 1))  # Change to (C, H, W)
    image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
    return image.unsqueeze(0)  # Add batch dimension


def main():
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = FullyConvNetwork().to(device)
    model.load_state_dict(torch.load('checkpoints/pix2pix_model_epoch_800.pth'))  # Adjust epoch number as needed
    model.eval()

    input_image_path = 'doro.jpg'  # Input image path

    input_image = load_and_preprocess_image(input_image_path).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)

    # Convert output tensor to image
    output_img_np = tensor_to_image(output[0])

    # Save or display the output image
    cv2.imwrite('output_doro.png', output_img_np)  # Save to file
    cv2.imshow('Output', output_img_np)  # Show the image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
