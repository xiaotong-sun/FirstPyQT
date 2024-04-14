import cv2
from PIL import Image
import numpy as np

img = Image.open('./data/query/118.7809181_32.03969824_180_0.png')
img = np.array(img)
pt = [1235625.2546045068, 3571435.4221651196]
cv2.rectangle(img, pt[0], pt[1], (0, 255, 0), 1, 4)
