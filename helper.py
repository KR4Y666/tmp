import numpy as np 
import cv2
from PIL import Image 
from PIL import ImageChops

def colors(image, subcontour):
  subcontour = np.flip(subcontour)

  colors = []
  for n in range(len(subcontour)-3):
    (y,x) = subcontour[n]
    (y1,x1) = subcontour[n+3]
    h, w = y1 - y, x1 - x
    colors.append(image[y-w, x+h, :3] + image[y+w, x-h, :3])

  colors = np.array(colors, 'uint8').reshape(-1,1,3)
  colors = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV)
  
  return colors.reshape(-1,3)

def choose_piece(arr_img, point, angle, center=(700,700)):
  img = Image.fromarray(arr_img)
  img = ImageChops.offset(img, center[1] - point[1], center[0] - point[0])
  img = img.rotate(angle)

  return np.array(img)

def rotate(point, angle, center=(700,700)):
  dy, dx = center[0]-point[0], point[1]-center[1]
  distance = np.sqrt(np.square(point[0]-center[0]) + np.square(point[1]-center[1]))
  if dx==0: dx = 1
  base = 90*(1-np.sign(dx)) + np.degrees(np.arctan(dy/dx))
  
  y = round(center[0] - distance * np.sin(np.pi * (base + angle)/180))
  x = round(center[1] + distance * np.cos(np.pi * (base + angle)/180))

  return (y,x)

def rescale(point, position, center=(150,150)):
  cy, cx, angle = position
  if angle!=0: (y, x) = rotate(point, angle, center)
  else: (y, x) = point

  return (y + cy - center[0], x + cx - center[1])
