import preprocessing
import matching 
from helper import rotate, choose_piece, rescale
import numpy as np

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

showpic(assembly)
