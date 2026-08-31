import plotly.graph_objects as go
import numpy as np
import requests
from PIL import Image
import io

def get_earth():
    url = "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg"
    resp = requests.get(url, headers={'User-Agent': 'Mozilla'}, verify=False)
    img = Image.open(io.BytesIO(resp.content)).convert('L')
    # Resize to 200x100 (width x height)
    img = img.resize((200, 100))
    # Convert to array, shape is (100, 200) -> (height, width) -> (lat, lon)
    arr = np.array(img) / 255.0
    return arr

earth_topo = get_earth()
# Transpose to match (lon, lat) meshgrid
earth_topo = earth_topo.T

# Create meshgrid
# u = longitude (200 points), v = latitude (100 points)
u = np.linspace(0, 2 * np.pi, 200)
v = np.linspace(0, np.pi, 100)
r = 6371

# Outer product: cos(u) is 200x1, sin(v) is 1x100 => 200x100
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

fig = go.Figure(data=[go.Surface(
    x=x, y=y, z=z,
    surfacecolor=earth_topo,
    colorscale=[[0, 'blue'], [1, 'green']]
)])
# fig.show() # Can't show in headless
print(f"X shape: {x.shape}")
print(f"Topo shape: {earth_topo.shape}")
