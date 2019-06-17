import os
import numpy as np
from PIL import Image
from dlr import DLRModel

# Load the compiled model
input_shape = {'data': [1, 3, 224, 224]} # A single RGB 224x224 image
output_shape = [1, 1000]                 # The probability for each one of the 1,000 classes
device = 'cpu'                           # Go, Raspberry Pi, go!
model = DLRModel('resnet50', input_shape, output_shape, device)#inception-BN-8MS

# Load names for ImageNet classes
synset_path = os.path.join('resnet50', 'synset.txt')
with open(synset_path, 'r') as f:
    synset = eval(f.read())

# Load the image
image = Image.open('image.jpeg')
image.load()

# Resize the image
new_width  = 224
new_height = 224
image = image.resize((new_width, new_height), Image.ANTIALIAS)
image.save('image224.jpeg')

# Create image numpy array
image = np.array(image) - np.array([123.68, 116.779, 103.939])
image /= np.array([58.395, 57.12, 57.375])
image = image.transpose((2, 0, 1))
image = image[np.newaxis, :]

# Load an image stored as a numpy array
#image = np.load('dog.npy').astype(np.float32)
#print image.shape
input_data = {'data': image}
#print(input_data)

# Predict 
out = model.run(input_data)
#print(out)
top1 = np.argmax(out[0])
prob = np.max(out)
print"Class: %s, probability: %f" % (synset[top1], prob)
