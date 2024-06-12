import os
from flask import Flask,render_template, request
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from unet_model import build_unet

app = Flask(__name__)

# Load your pre-trained UNet model
model = build_unet()
checkpoint_path = 'checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        # Open and preprocess the image
        image = Image.open(image_file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        # Perform inference
        with torch.inference_mode():
            pred_mask = model(image_tensor)
            pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)

        # Save the input image and predicted mask
        input_image_path = os.path.join('static', 'input_image.png')
        image.save(input_image_path)
        mask_image = Image.fromarray(pred_mask * 255)
        mask_path = os.path.join('static', 'mask.png')
        mask_image.save(mask_path)

        return render_template('result.html', input_image_path=input_image_path, mask_path=mask_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)