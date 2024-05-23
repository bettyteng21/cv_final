import torch
import torch.nn as nn
import cv2
import numpy as np

# Define the SRCNN model
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def post_process(image):
    # Load the pre-trained SRCNN model
    model = SRCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    model.load_state_dict(torch.load('srcnn_x4.pth', map_location=device))
    model.eval()

    # Convert the image to YCrCb color space
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image_ycrcb)

    # Normalize the Y channel
    y = y.astype(np.float32) / 255.0

    # Convert to PyTorch tensor
    y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)

    # Move the tensor to the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    model = model.to(device)

    # Apply SRCNN
    with torch.no_grad():
        predicted_y = model(y)

    # Convert the output to a NumPy array
    predicted_y = predicted_y.squeeze().cpu().numpy()

    # Denormalize the Y channel
    predicted_y = (predicted_y * 255.0).clip(0, 255).astype(np.uint8)

    # Merge channels and convert back to BGR color space
    output_image = cv2.merge([predicted_y, cr, cb])
    output_image = cv2.cvtColor(output_image, cv2.COLOR_YCrCb2BGR)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    return output_image
