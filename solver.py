
from email.mime import image
from turtle import shape
import numpy as np
import cv2
from PIL import Image
from scipy import median_filter
from helper import colors, choose_piece, rescale, rotate
from fastdtw import fastdtw
import matplotlib.pyplot as plt

 
image = Image.open('puzzle18.png')
image = Image.convert('RGBA') 
puzzle = np.array(image)

sill = cv2.cvtColor(puzzle, cv2.COLOR_RGBA2GRAY)
sill = cv2.adaptivesill(sill, 255, 0, 1, 3, 3)
sill = cv2.GaussianBlur(sill, (3,3), 1)

contours, _ = cv2.findContours(sill, 0, 1)
sorting = sorted([[cnt.shape[0], i] for i, cnt in enumerate(contours)], reverse=True)[:6]
max = [contours[s[1]] for s in sorting] 
fill = cv2.drawContours(np.zeros(puzzle.shape[:2]), max, -1, 255, thickness=cv2.FILLED)
tender = median_filter(fill.astype('uint8'), size=4)
trim_contours, _ = cv2.findContours(tender, 0, 1)
cv2.drawContours(tender, trim_contours, -1, color=0, thickness=1)

contours, _ = cv2.findContours(tender, 0, 1)
global pieces 
pieces = []    
piece_center = []

for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    shape, piece = np.zeros(puzzle.shape[:2]), np.zeros((300,300,4), 'uint8')
    cv2.drawContours(shape, [contours[i]], -1, color=1, thickness=-1)
    shape = (puzzle * shape[:,:,None])[y:y+h,x:x+w,:]
    piece[(300-h)//2:(300-h)//2+h,(300-w)//2:(300-w)//2+w] = shape
    pieces.append(piece)
    piece_center.append((h//2+y, w//2+x))

screen_pieces = []
for i in range(len(pieces)):
    screen_piece = np.zeros((1400,1400,4), 'uint8')
    screen_piece[550:850, 550:850] = pieces[i].copy()
    screen_pieces.append(screen_piece)

def matchpieces(A, B):

  LENGTH = 160
  PRECISION = 8
  STEP_A = 20
  STEP_B = 7
  MAX_FORM = 0.015
  MAX_COLOR = 8000
  MAX_PIXEL = 0.03
  MAX_FIT = 0.77

  CENTER = round(LENGTH/2)

  pieceA, pieceB = pieces[A], pieces[B]
  cntA, _ = cv2.findContours(pieceA[:,:,3], 0, 1)
  cntB, _ = cv2.findContours(pieceB[:,:,3], 0, 1)
  cntA, cntB = cntA[0].reshape(-1,2), cntB[0].reshape(-1,2)
  sumLen = cntA.shape[0] + cntB.shape[0]

  # Contour matching
  form_matches = []
  for i in range(0, cntA.shape[0], STEP_A):

    # subcontour A and its type
    subcA = np.roll(cntA, -i, 0)[:LENGTH]
    spotA = tuple(np.flip(subcA[CENTER]))
    cA, (hA,wA), aA = cv2.minAreaRect(subcA)
    typespotA = np.int0(np.flip(subcA[0] + subcA[-1] - cA))
    typeA = pieceA[:,:,3][tuple(typespotA)]
    a = cv2.drawContours(np.zeros((300,300),'uint8'), subcA.reshape(-1,1,2), -1, 255, 1)

    # loop through match subcontours
    for j in range(0, cntB.shape[0], STEP_B):
      
      # subcontour B and its type
      subcB = np.roll(cntB, -j, 0)[:LENGTH]
      spotB = tuple(np.flip(subcB[CENTER]))
      cB, (hB,wB), aB = cv2.minAreaRect(subcB)
      typespotB = np.int0(np.flip(subcB[0] + subcB[-1] - cB))
      typeB = pieceB[:,:,3][tuple(typespotB)]

      # record good form matches
      if typeB != typeA:
        if ((abs(hA-hB) < PRECISION) & (abs(wA-wB) < PRECISION)) or ((abs(hA-wB) < PRECISION) & (abs(wA-hB) < PRECISION)):
          b = cv2.drawContours(np.zeros((300,300),'uint8'), subcB.reshape(-1,1,2), -1, 255, 1)
          fmatch = cv2.matchShapes(a,b,1,0)
          if fmatch < MAX_FORM: 
            colinear = True if np.sign(hA-wA) == np.sign(hB-wB) else False
            if colinear:
              codirect = True if (np.sign(typespotA - np.flip(cA)) ==  np.sign(typespotB - np.flip(cB))).all() else False
            else:
              c = np.concatenate([np.sign(typespotA - np.flip(cA)), np.sign(typespotB - np.flip(cB))])
              codirect = True if (abs(np.sum(c[:3])) + abs(np.sum(c[-3:]))) == 4 else False
            if not colinear: aB = aB + 90
            if not codirect: aB = aB + 180  
            form_matches.append([(i, j), spotA, spotB, round(aB-aA,4), round(fmatch,4)])
 
  # Color matching
  color_matches = []
  for n in range(len(form_matches)):
    (i, j), spotA, spotB, corner, fmatch = form_matches[n]
    subcA = np.roll(cntA, -i, 0)[:LENGTH] 
    subcB = np.roll(cntB, -j, 0)[:LENGTH]
    colorsA = colors(pieceA, subcA)
    colorsB = colors(pieceB, subcB)
    cmatch = fastdtw(colorsA, np.flip(colorsB, axis=0))[0]
    if cmatch < MAX_COLOR: 
      color_matches.append([(i, j), spotA, spotB, corner, fmatch, round(cmatch)])

  # Pre-fitting
  fit_matches = []
  for n in range(len(color_matches)):
    (i, j), spotA, spotB, corner, fmatch, cmatch = color_matches[n]
    a = choose_piece(screen_pieces[A][:,:,3], rescale(spotA, [700,700,0]), 0)
    b = choose_piece(screen_pieces[B][:,:,3], rescale(spotB, [700,700,0]), corner)
    loss = 1 - (np.sum((a+b)>0) / (np.sum(a>0) + np.sum(b>0)))
    contours, _ = cv2.findContours((a+b), 0, 1)
    fit = contours[0].shape[0] / sumLen
    if (loss < MAX_PIXEL) & (fit < MAX_FIT): 
      fit_matches.append([(A, B), (i, j), spotA, spotB, corner, fmatch, cmatch, round(loss+fit,4), 0])

  fit_matches.sort(key=lambda n: n[-1])

  return fit_matches

# Calculate all possible matches
matches = []
for a in range(len(pieces)-1):
  for b in range(a+1,len(pieces)):
    matches.extend(matchpieces(a,b))

# Flip and sort
for n in range(len(matches)):
  pair, ij, spota, spotb, corner, fmatch, cmatch, fit, lock = matches[n]
  matches.extend([[(pair[1],pair[0]), ij, spotb, spota, -corner, fmatch, cmatch, fit, lock]])
matches.sort(key=lambda m: (m[0], m[-2]))

def updatescreen(screen, locations, pointA, pointB, cornerA, cornerB):
  
  # update records for pieces on screen
  for N, pos in enumerate(locations):
    if N in screen:
      new_center = (pos[0] + 700 - pointA[0], pos[1] + 700 - pointA[1])
      new_center = rotate(new_center, cornerA)
      new_corner = pos[2] + cornerA
      locations[N] = [*new_center, new_corner]

  # append record for the added piece
  screen.append(B)
  center = rotate((700 + 700 - pointB[0], 700 + 700 - pointB[1]), cornerB)
  locations[B] = [*center, cornerB]

  return screen, locations

# Assembly
assembly = screen_pieces[0].copy()
locations = [[0,0,0]]*len(pieces)
locations[0] = [700,700,0]
screen = [0]
attempts = 0

while (len(screen) < 15) & (attempts < 10):
  for n in range(len(matches)):
        
    # take next matching pair
    (A, B), ij, pointA, pointB, cornerB, _, _, _, lock = matches[n]
    pointA = rescale(pointA, locations[A])
    pointB = rescale(pointB, (700,700,0))

    if A in screen:
      cornerA = - locations[A][2]
      pre_assembly = choose_piece(assembly.copy(), pointA, cornerA)
      
      if B not in screen:
        newpiece = choose_piece(screen_pieces[B], pointB, cornerB)

        # fix or pass depending on loss of pixels
        loss = (np.sum(pre_assembly[:,:,3]>0) + np.sum(newpiece[:,:,3]>0) - 
                np.sum((pre_assembly+newpiece)[:,:,3]>0)
                ) / np.sum(newpiece[:,:,3]>0)
        if loss < 0.1: 
          matches[n][-1] = 1
          assembly = pre_assembly.copy() + newpiece.copy()
          screen, locations = updatescreen(screen, locations, 
                                           pointA, pointB, cornerA, cornerB)
  
  attempts += 1

plt.figure(figsize=(10, 10/1000*727))
plt.imshow(assembly, cmap='gray')   
plt.axis('off')   
plt.show()  
