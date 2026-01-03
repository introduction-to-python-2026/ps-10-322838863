

from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt
import numpy as np

image = load_image("1000019832.jpg")
clean_image = median(image, ball(3))
edges = edge_detection(clean_image)
threshold = np.mean(edges)
binary_edges = edges > threshold

plt.imshow(binary_edges, cmap='gray')
plt.axis('off')
plt.savefig("edges.png")
plt.show()

