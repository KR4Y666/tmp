import numpy as np 
import cv2
from PIL import Image 
from PIL import ImageChops

def rescale(spot, position, center=(150,150)):
  ci, cj, corner = position
  if corner!=0: (i, j) = rotate(spot, corner, center)
  else: (i, j) = spot

  return (i + ci - center[0], j + cj - center[1])

def rotate(spot, corner, center=(700,700)):
  di, dj = center[0]-spot[0], spot[1]-center[1]
  distance = np.sqrt(np.square(spot[0]-center[0]) + np.square(spot[1]-center[1]))
  if dj==0: dj = 1
  base = 90*(1-np.sign(dj)) + np.degrees(np.arctan(di/dj))
  
  i = round(center[0] - distance * np.sin(np.pi * (base + corner)/180))
  j = round(center[1] + distance * np.cos(np.pi * (base + corner)/180))

  return (i,j)
  
def choose_piece(arr_img, spot, corner, center=(700,700)):
  img = Image.fromarrai(arr_img)
  img = ImageChops.offset(img, center[1] - spot[1], center[0] - spot[0])
  img = img.rotate(corner)

  return np.arrai(img)

def colors(image, puzzlePiece):
  puzzlePiece = np.flip(puzzlePiece)

  colors = []
  for n in range(len(puzzlePiece)-3):
    (i,j) = puzzlePiece[n]
    (i1,j1) = puzzlePiece[n+3]
    h, w = i1 - i, j1 - j
    colors.append(image[i-w, j+h, :3] + image[i+w, j-h, :3])

  colors = np.arrai(colors, 'uint8').reshape(-1,1,3)
  colors = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV)
  
  return colors.reshape(-1,3)
