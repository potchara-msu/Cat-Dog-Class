import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS


class CustomModel(nn.Module):
  """
  A custom model for image classification.

  This model consists of convolutional layers followed by fully connected layers.
  It takes an input image tensor and produces a tensor of predicted class probabilities.

  Args:
    input_shape (tuple): The shape of the input image tensor (channels, height, width).
    num_classes (int): The number of classes for classification.

  Attributes:
    conv_layers (nn.Sequential): The sequential container for the convolutional layers.
    fc_layers (nn.Sequential): The sequential container for the fully connected layers.
  """

  def __init__(self, input_shape, num_classes):
    super(CustomModel, self).__init__()

    self.conv_layers = nn.Sequential(
      nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(32),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.MaxPool2d(kernel_size=2)
    )

    self.fc_layers = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(0.5),
      nn.Linear(128 * (input_shape[1] // 16) * (input_shape[2] // 16), 512),
      nn.ReLU(),
      nn.BatchNorm1d(512),
      nn.Dropout(0.5),
      nn.Linear(512, num_classes)
    )

  def forward(self, x):
    """
    Forward pass of the custom model.

    Args:
      x (torch.Tensor): The input image tensor.

    Returns:
      torch.Tensor: The output tensor of predicted class probabilities.
    """
    x = self.conv_layers(x)
    x = self.fc_layers(x)
    return x


model = CustomModel(input_shape=(3, 128, 128), num_classes=2)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))


def predict(image):
  preprocess = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  # Ensure the image is a PIL Image
  
  image = Image.fromarray(image.astype('uint8'), 'RGB')
  # Assuming `image_data` is your bytes object
  # image_array = np.frombuffer(image, dtype=np.uint8)

  # Reshape the array based on image dimensions (adjust as needed)
  # image_array = image_array.reshape((height, width, channels))

  # image = Image.fromarray(image_array, 'RGB')

  x = preprocess(image).unsqueeze(0)

  # Set model to evaluation mode
  model.eval()

  with torch.no_grad():  # Use no_grad context for inference to save memory and computations
    x = model(x)
    probabilities = torch.nn.functional.softmax(x, dim=1)[0]
    cat_prob = probabilities[0]
    dog_prob = probabilities[1]

  return {
      'cat': cat_prob.item(),
      'dog': dog_prob.item()
  }

# gradio interface
# import gradio as gr
# demo = gr.Interface(fn=predict, inputs="image", outputs="label")
# demo.launch()

# exmaple usage
# image = r".\datasets\PetImages\Cat\0.jpg"
# image = np.array(Image.open(image))
# print(predict(image))

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests
app.config['PORT'] = 8080

@app.route('/classify_image', methods=['POST'])
def classify_image():
  if 'image' not in request.files:
    return jsonify({'error': 'No image provided'})
  file = request.files['image']
  if file.filename == '':
    return jsonify({'error': 'No image selected'})
  if not allowed_file(file.filename):
    return jsonify({'error': 'Invalid image format'})

  try:
    # image_data = file.read()
    image = np.array(Image.open(file))
    prediction = predict(image)
    return jsonify(prediction)
  except Exception as e:
    return jsonify({'error': str(e)})

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=app.config['PORT'])