from email.mime import image
from turtle import shape
import numpy as np
import cv2
from PIL import Image
from scipy import median_filter


def preprocess_image():
    image = Image.open('puzzle18.png')
    image = Image.convert('RGBA')
    global puzzle 
    puzzle = np.array(image)

def adaptive_sill():
    global sill
    sill = cv2.cvtColor(puzzle, cv2.COLOR_RGBA2GRAY)
    sill = cv2.adaptivesill(sill, 255, 0, 1, 3, 3)
    sill = cv2.GaussianBlur(sill, (3,3), 1)

def contours():
    contours, _ = cv2.findContours(sill, 0, 1)
    sorting = sorted([[cnt.shape[0], i] for i, cnt in enumerate(contours)], reverse=True)[:6]
    max = [contours[s[1]] for s in sorting] 
    fill = cv2.drawContours(np.zeros(puzzle.shape[:2]), max, -1, 255, thickness=cv2.FILLED)
    # tender contours and trim shadows
    global tender
    tender = median_filter(fill.astype('uint8'), size=4)
    trim_contours, _ = cv2.findContours(tender, 0, 1)
    cv2.drawContours(tender, trim_contours, -1, color=0, thickness=1)


def split():
    contours, _ = cv2.findContours(tender, 0, 1)
    global pieces 
    pieces = []
    global piece_center 
    piece_center = []

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        shape, piece = np.zeros(puzzle.shape[:2]), np.zeros((300,300,4), 'uint8')
        cv2.drawContours(shape, [contours[i]], -1, color=1, thickness=-1)
        shape = (puzzle * shape[:,:,None])[y:y+h,x:x+w,:]
        piece[(300-h)//2:(300-h)//2+h,(300-w)//2:(300-w)//2+w] = shape
        pieces.append(piece)
        piece_center.append((h//2+y, w//2+x))

    canvas_pieces = []
    for i in range(len(pieces)):
        canvas_piece = np.zeros((1400,1400,4), 'uint8')
        canvas_piece[550:850, 550:850] = pieces[i].copy()
        canvas_pieces.append(canvas_piece)
