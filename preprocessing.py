import numpy as np
import cv2
from PIL import Image
from scipy import median_filter


def preprocess_image():
    img = Image.open('puzzle18.png')
    img = Image.convert('RGBA')
    global puzzle 
    puzzle = np.array(img)

def adaptive_threshold():
    global threshold
    threshold = cv2.cvtColor(puzzle, cv2.COLOR_RGBA2GRAY)
    threshold = cv2.adaptiveThreshold(threshold, 255, 0, 1, 3, 3)
    threshold = cv2.GaussianBlur(threshold, (3,3), 1)

def contours():
    contours, _ = cv2.findContours(threshold, 0, 1)
    sorting = sorted([[cnt.shape[0], i] for i, cnt in enumerate(contours)], reverse=True)[:6]
    biggest = [contours[s[1]] for s in sorting] 
    fill = cv2.drawContours(np.zeros(puzzle.shape[:2]), biggest, -1, 255, thickness=cv2.FILLED)
    # Smooth contours and trim shadows
    global smooth
    smooth = median_filter(fill.astype('uint8'), size=4)
    trim_contours, _ = cv2.findContours(smooth, 0, 1)
    cv2.drawContours(smooth, trim_contours, -1, color=0, thickness=1)


def split():
    contours, _ = cv2.findContours(smooth, 0, 1)
    global tiles 
    tiles = []
    global tile_centers 
    tile_centers = []

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        shape, tile = np.zeros(puzzle.shape[:2]), np.zeros((300,300,4), 'uint8')
        cv2.drawContours(shape, [contours[i]], -1, color=1, thickness=-1)
        shape = (puzzle * shape[:,:,None])[y:y+h,x:x+w,:]
        tile[(300-h)//2:(300-h)//2+h,(300-w)//2:(300-w)//2+w] = shape
        tiles.append(tile)
        tile_centers.append((h//2+y, w//2+x))

    canvas_tiles = []
    for i in range(len(tiles)):
        canvas_tile = np.zeros((1400,1400,4), 'uint8')
        canvas_tile[550:850, 550:850] = tiles[i].copy()
        canvas_tiles.append(canvas_tile)
