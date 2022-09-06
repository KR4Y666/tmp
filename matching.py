import preprocessing
import numpy as np 
import cv2
from PIL import Image, ImageChops
from fastdtw import fastdtw
from helper import choose_piece, colors, rescale






def matchTiles(A, B):

  LENGTH = 160
  PRECISION = 8
  STEP_A = 20
  STEP_B = 7
  MAX_FORM = 0.015
  MAX_COLOR = 8000
  MAX_PIXEL = 0.03
  MAX_FIT = 0.77

  CENTER = round(LENGTH/2)

  tileA, tileB = tiles[A], tiles[B]
  cntA, _ = cv2.findContours(tileA[:,:,3], 0, 1)
  cntB, _ = cv2.findContours(tileB[:,:,3], 0, 1)
  cntA, cntB = cntA[0].reshape(-1,2), cntB[0].reshape(-1,2)
  sumLen = cntA.shape[0] + cntB.shape[0]

  # Contour matching
  form_matches = []
  for i in range(0, cntA.shape[0], STEP_A):

    # subcontour A and its type
    subcA = np.roll(cntA, -i, 0)[:LENGTH]
    pointA = tuple(np.flip(subcA[CENTER]))
    cA, (hA,wA), aA = cv2.minAreaRect(subcA)
    typepointA = np.int0(np.flip(subcA[0] + subcA[-1] - cA))
    typeA = tileA[:,:,3][tuple(typepointA)]
    a = cv2.drawContours(np.zeros((300,300),'uint8'), subcA.reshape(-1,1,2), -1, 255, 1)

    # loop through match subcontours
    for j in range(0, cntB.shape[0], STEP_B):
      
      # subcontour B and its type
      subcB = np.roll(cntB, -j, 0)[:LENGTH]
      pointB = tuple(np.flip(subcB[CENTER]))
      cB, (hB,wB), aB = cv2.minAreaRect(subcB)
      typepointB = np.int0(np.flip(subcB[0] + subcB[-1] - cB))
      typeB = tileB[:,:,3][tuple(typepointB)]

      # record good form matches
      if typeB != typeA:
        if ((abs(hA-hB) < PRECISION) & (abs(wA-wB) < PRECISION)) or ((abs(hA-wB) < PRECISION) & (abs(wA-hB) < PRECISION)):
          b = cv2.drawContours(np.zeros((300,300),'uint8'), subcB.reshape(-1,1,2), -1, 255, 1)
          fmatch = cv2.matchShapes(a,b,1,0)
          if fmatch < MAX_FORM: 
            colinear = True if np.sign(hA-wA) == np.sign(hB-wB) else False
            if colinear:
              codirect = True if (np.sign(typepointA - np.flip(cA)) ==  np.sign(typepointB - np.flip(cB))).all() else False
            else:
              c = np.concatenate([np.sign(typepointA - np.flip(cA)), np.sign(typepointB - np.flip(cB))])
              codirect = True if (abs(np.sum(c[:3])) + abs(np.sum(c[-3:]))) == 4 else False
            if not colinear: aB = aB + 90
            if not codirect: aB = aB + 180  
            form_matches.append([(i, j), pointA, pointB, round(aB-aA,4), round(fmatch,4)])
 
  # Color matching
  color_matches = []
  for n in range(len(form_matches)):
    (i, j), pointA, pointB, angle, fmatch = form_matches[n]
    subcA = np.roll(cntA, -i, 0)[:LENGTH] 
    subcB = np.roll(cntB, -j, 0)[:LENGTH]
    colorsA = colors(tileA, subcA)
    colorsB = colors(tileB, subcB)
    cmatch = fastdtw(colorsA, np.flip(colorsB, axis=0))[0]
    if cmatch < MAX_COLOR: 
      color_matches.append([(i, j), pointA, pointB, angle, fmatch, round(cmatch)])

  # Pre-fitting
  fit_matches = []
  for n in range(len(color_matches)):
    (i, j), pointA, pointB, angle, fmatch, cmatch = color_matches[n]
    a = choose_piece(canvas_tiles[A][:,:,3], rescale(pointA, [700,700,0]), 0)
    b = choose_piece(canvas_tiles[B][:,:,3], rescale(pointB, [700,700,0]), angle)
    loss = 1 - (np.sum((a+b)>0) / (np.sum(a>0) + np.sum(b>0)))
    contours, _ = cv2.findContours((a+b), 0, 1)
    fit = contours[0].shape[0] / sumLen
    if (loss < MAX_PIXEL) & (fit < MAX_FIT): 
      fit_matches.append([(A, B), (i, j), pointA, pointB, angle, fmatch, cmatch, round(loss+fit,4), 0])

  fit_matches.sort(key=lambda n: n[-1])

  return fit_matches

# Calculate all possible matches
matches = []
for a in range(len(tiles)-1):
  for b in range(a+1,len(tiles)):
    matches.extend(matchTiles(a,b))

# Flip and sort
for n in range(len(matches)):
  pair, ij, pointa, pointb, angle, fmatch, cmatch, fit, lock = matches[n]
  matches.extend([[(pair[1],pair[0]), ij, pointb, pointa, -angle, fmatch, cmatch, fit, lock]])
matches.sort(key=lambda m: (m[0], m[-2]))