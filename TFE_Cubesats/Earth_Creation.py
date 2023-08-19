import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

# Load the Earth texture image
texture_path = "/Users/eliotthubin/Downloads/earth_texture.jpg"
texture_image = Image.open(texture_path)

# Convert the image to RGBA format
texture_image_rgba = texture_image.convert("RGBA")

# Earth parameters
radius = 6371  # Radius of the Earth in kilometers
center = np.array([0, 0, 0])

# Create a figure and axes with 3D projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a mesh grid for the sphere
u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
x = center[0] + radius * np.cos(u) * np.sin(v)
y = center[1] + radius * np.sin(u) * np.sin(v)
z = center[2] + radius * np.cos(v)

# Normalize texture coordinates to range [0, 1]
norm_u = (u - u.min()) / (u.max() - u.min())
norm_v = (v - v.min()) / (v.max() - v.min())

# Resize texture image to match the shape of the mesh grid
texture_image_resized = texture_image_rgba.resize((u.shape[0], v.shape[0]))

# Extract texture data
texture_data = np.array(texture_image_resized) / 255.0  # Normalize RGBA values to [0, 1] range

# Adjust transparency (alpha) of the texture
alpha = 0.1  # Set the desired transparency value (0.0 - transparent, 1.0 - opaque)
texture_data[..., 3] = alpha  # Set alpha channel to the specified value

# Apply the texture to the surface of the sphere
ax.plot_surface(x, y, z, facecolors=texture_data, rstride=1, cstride=1, shade=False)

ax.set_xlim(-radius, radius)
ax.set_ylim(-radius, radius)
ax.set_zlim(-radius, radius)
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.view_init(elev=30, azim=45)

# Hide grid lines
ax.grid(False)

# Hide ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Remove axis spines
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()




