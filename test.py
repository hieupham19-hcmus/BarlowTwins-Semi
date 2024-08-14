import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from Models.Transformer.SwinUnet import SwinUnet

def load_and_test_image(model_class, model_path, image_path, img_size, device='cuda'):
    """
    Load a model from a .pth file, pass a single image through it, and display the output.

    Args:
        model_class (torch.nn.Module): The model class definition.
        model_path (str): Path to the .pth file.
        image_path (str): Path to the input image.
        img_size (int): Size to which the input image should be resized.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        output (numpy.ndarray): The model's output for the image.
    """
    # Load the model
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    # Pass the image through the model
    with torch.no_grad():
        output = torch.sigmoid(model(image)).squeeze().cpu().numpy()

    # Display the input and output images
    display_image(image_path, output)

    return output

def display_image(original_image_path, output):
    """
    Display the original image and the model's output side by side.

    Args:
        original_image_path (str): Path to the original input image.
        output (numpy.ndarray): The model's output for the image.
    """
    original_image = Image.open(original_image_path).convert('RGB')
    
    plt.figure(figsize=(10, 5))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    # Display the model output
    plt.subplot(1, 2, 2)
    plt.imshow(output, cmap='gray')
    plt.title('Model Output')
    plt.axis('off')

    plt.show()

# Assuming SwinUnet is your model class
model_path = '/mnt/d/RESEARCH/BarlowTwins-Semi/checkpoints/polypgen/tmp_0.1/fold1/best.pth'
image_path = '/mnt/d/RESEARCH/BarlowTwins-Semi/data/polypgen/images/126.jpg'
img_size = 224  # Example image size

def save_image(image, path):
    """
    Save an image to disk.

    Args:
        image (numpy.ndarray): The image to save.
        path (str): The path to save the image to.
    """
    plt.imsave(path, image, cmap='gray')

# Load the model, process the image, and display the output
output = load_and_test_image(SwinUnet, model_path, image_path, img_size)


