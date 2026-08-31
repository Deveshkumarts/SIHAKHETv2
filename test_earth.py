import requests
from PIL import Image
import io
import numpy as np
import base64

# Download a small equirectangular earth map
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Land_ocean_ice_2048.jpg/320px-Land_ocean_ice_2048.jpg"
resp = requests.get(url)
img = Image.open(io.BytesIO(resp.content)).convert("L")
img = img.resize((100, 100)) # Match our u, v linspace size

arr = np.array(img)
print(arr.shape)
print(arr[50, 50])
