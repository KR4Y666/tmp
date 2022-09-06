import preprocessing
import matching 
from helper import rotate, choose_piece, rescale
import numpy as np

def updateCanvas(canvas, positions, pointA, pointB, angleA, angleB):
  
  # update records for tiles on canvas
  for N, pos in enumerate(positions):
    if N in canvas:
      new_center = (pos[0] + 700 - pointA[0], pos[1] + 700 - pointA[1])
      new_center = rotate(new_center, angleA)
      new_angle = pos[2] + angleA
      positions[N] = [*new_center, new_angle]

  # append record for the added tile
  canvas.append(B)
  center = rotate((700 + 700 - pointB[0], 700 + 700 - pointB[1]), angleB)
  positions[B] = [*center, angleB]

  return canvas, positions

# Assembly
assembly = canvas_tiles[0].copy()
positions = [[0,0,0]]*len(tiles)
positions[0] = [700,700,0]
canvas = [0]
attempts = 0

while (len(canvas) < 15) & (attempts < 10):
  for n in range(len(matches)):
        
    # take next matching pair
    (A, B), ij, pointA, pointB, angleB, _, _, _, lock = matches[n]
    pointA = rescale(pointA, positions[A])
    pointB = rescale(pointB, (700,700,0))

    if A in canvas:
      angleA = - positions[A][2]
      pre_assembly = choose_piece(assembly.copy(), pointA, angleA)
      
      if B not in canvas:
        newtile = choose_piece(canvas_tiles[B], pointB, angleB)

        # fix or pass depending on loss of pixels
        loss = (np.sum(pre_assembly[:,:,3]>0) + np.sum(newtile[:,:,3]>0) - 
                np.sum((pre_assembly+newtile)[:,:,3]>0)
                ) / np.sum(newtile[:,:,3]>0)
        if loss < 0.1: 
          matches[n][-1] = 1
          assembly = pre_assembly.copy() + newtile.copy()
          canvas, positions = updateCanvas(canvas, positions, 
                                           pointA, pointB, angleA, angleB)
  
  attempts += 1

showpic(assembly)
