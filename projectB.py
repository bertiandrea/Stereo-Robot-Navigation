import matplotlib.pyplot as plt
import argparse
import cv2 as cv
import pandas as pd
import numpy as np
import math
##############################################################################################################
# Axis definition
X_AXIS = 1
Y_AXIS = 0
# Dissimilarity measure
M_SAD = 0
M_SSD = 1
# Chessboard dimensions (in millimeters)
H = 178
W = 125
# Chessboard parameters
CB_INNER_W_CORNERS = 6
CB_INNER_H_CORNERS = 8
# Best blocksize value 
BEST_BLOCKSIZE_VALUE=33
# Alarm trigger limit (in mm)
MINIMUM_DISTANCE =  800
# Camera parameters
FOCAL_LENGHT =  567.2 #in pixels
BASELINE = 92.226  #in mm
# Smoothing curves window size
SMOOTH=10
##############################################################################################################
def cutMatrix(img, imageDim):
    centerY = img.shape[Y_AXIS]//2
    centerX = img.shape[X_AXIS]//2
    return img[centerY-(imageDim//2):centerY+(imageDim//2), centerX-(imageDim//2):centerX+(imageDim//2)]
################################################################################################################
def computeChessboard(imgL, imageDim):
    ret, corners = cv.findChessboardCorners(imgL ,(CB_INNER_H_CORNERS,CB_INNER_W_CORNERS))
    if ret == True:
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 1.0)
        cv.cornerSubPix(imgL, corners, (5, 5), (-1, -1), term)
        cv.drawChessboardCorners(imgL, (CB_INNER_H_CORNERS,CB_INNER_W_CORNERS), corners, ret)
        top_left = corners[0][0]
        top_right = corners[(CB_INNER_W_CORNERS - 1) * CB_INNER_H_CORNERS][0]
        bottom_left = corners[CB_INNER_H_CORNERS - 1][0]
        h = np.linalg.norm(top_left - bottom_left)
        w = np.linalg.norm(top_left - top_right)
        imgL = cutMatrix(imgL, imageDim)
        return imgL, h, w
    else:
        imgL = cutMatrix(imgL, imageDim)
        return imgL, None, None
##############################################################################################################
def computeDisparityMap(imgL, imgR, disparity_range, block_size, imageDim, measure = M_SAD):
    kernel = np.ones([block_size, block_size])
    measure_maps = np.zeros([imageDim, imageDim, len(disparity_range)])
    left_image_f = np.float32(imgL)
    right_image_f = np.float32(imgR)
       
    left_translation = np.float32([[1, 0, -(left_image_f.shape[X_AXIS]-imageDim)//2], [0, 1, -(left_image_f.shape[Y_AXIS]-imageDim)//2]])
    left_ROI = cv.warpAffine(left_image_f, left_translation, (imageDim, imageDim))
    for i in disparity_range:
        right_translation = left_translation + ([[0,0,i],[0,0,0]])
        right_ROI = cv.warpAffine(right_image_f, right_translation, (imageDim, imageDim))
        if measure == M_SSD:
            differences = np.square(left_ROI-right_ROI)
        else: # M_SAD
            differences = np.abs(left_ROI-right_ROI)       
        measure_map = cv.filter2D(src=differences, ddepth=-1, kernel=kernel, borderType=cv.BORDER_ISOLATED)
        measure_maps[:, :, i-disparity_range[0]] = measure_map

    disparity_map = np.argmin(measure_maps, axis=-1)
    disparity_map += disparity_range[0]
    return disparity_map, measure_maps
##############################################################################################################
def generateOutputImage(frame, distance, imgSize, alarm):
    output_image = np.copy(frame)
    if distance <= alarm: # If value <= threshold, the box will be red, otherwise green
        color = [158, 3, 34]
    else:
        color = [0, 184, 31]
    # Highlight the disparity map area
    cv.rectangle(img = output_image, 
                 pt1 = ((frame.shape[X_AXIS]-imgSize)//2, (frame.shape[Y_AXIS]-imgSize)//2),
                 pt2 = ((frame.shape[X_AXIS]+imgSize)//2, (frame.shape[Y_AXIS]+imgSize)//2),
                 color = color, thickness = 5, lineType=cv.LINE_AA)
    # Add the text inside the rectangle
    cv.putText(img = output_image, text = "Distance: {:.2f}m".format(distance/1000), org=(10, 30),
               fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = color, thickness = 5, lineType = cv.LINE_AA)
    return output_image
##############################################################################################################
def computeObstaclesCoords(disparity_map, num_stripes):
    height, width = disparity_map.shape
    stripe_width = width // num_stripes
    # Calcolo della disparità media per ciascuna striscia
    main_disparities = []
    for i in range(0, num_stripes):
        stripe = disparity_map[:, stripe_width * i : stripe_width * (i + 1)]
        stripe_main_disparity = np.average(stripe)
        main_disparities.append(stripe_main_disparity)
    # Calcolo delle coordinate 2D dei centri delle strisce nell'immagine
    frame_points = np.zeros((num_stripes, 2))
    for i in range(num_stripes):
        frame_points[i] = [stripe_width * (i + 0.5), height / 2]
    # Traslazione del sistema di coordinate con origine al centro dell'immagine (u,v)
    image_points = frame_points - [width / 2, height / 2]
    # Conversione delle coordinate immagine (u,v) in coordinate reali (X,Z)
    XZ_coords = np.zeros((num_stripes, 2))
    for i in range(0, num_stripes):
        Z = (BASELINE * FOCAL_LENGHT) / main_disparities[i]
        X = image_points[i, 0] * Z / FOCAL_LENGHT
        XZ_coords[i] =  [X, Z]
    
    # Approssimazione di una retta attraverso i punti XZ e calcolo dell'angolo
    # Utilizziamo np.polyfit per una regressione lineare (Z = m*X + q)
    coef = np.polyfit(XZ_coords[:, 0], XZ_coords[:, 1], 1)
    angle_deg = math.degrees(math.atan(coef[0]))

    # Normalizzazione delle coordinate tra 0 e 1 (Useremo queste coordinate per disegnare la vista planare)
    XZ_coords[:, 0] = 0.5 + XZ_coords[:, 0] / (2 * np.max(XZ_coords[:, 0]))
    XZ_coords[:, 1] = 1 - XZ_coords[:, 1] / np.max(XZ_coords[:, 1])
    return XZ_coords, angle_deg

def drawPlanarView(norm_coords, angle):
    view = np.ones((400, 400, 3), dtype=np.uint8) * 255  # Sfondo bianco
    obst_width = 400 // (len(norm_coords) * 2)  # Larghezza dei rettangoli rappresentanti gli ostacoli
    # Disegno degli assi X e Z
    cv.arrowedLine(view, (200,380), (200,20), (127,127,127), 1, tipLength=0.03)
    cv.arrowedLine(view, (20,360), (380,360), (127,127,127), 1, tipLength=0.03)
    cv.putText(view, "X", (354,380), cv.FONT_HERSHEY_DUPLEX, 0.5, (127,127,127), 1)
    cv.putText(view, "Z", (214,30), cv.FONT_HERSHEY_DUPLEX, 0.5, (127,127,127), 1)
    # Disegno della telecamera
    cv.rectangle(view, (170,350), (230,370), (255,0,0), thickness=cv.FILLED)
    cv.rectangle(view, (178,343), (192,349), (0,0,0), thickness=cv.FILLED)
    cv.rectangle(view, (208,343), (222,349), (0,0,0), thickness=cv.FILLED)
    # Scalatura delle coordinate per la visualizzazione
    scaled_coords = norm_coords * 320 + 40  # scala i valori da [0,1] a [40,360]
    # Disegno degli ostacoli (rettangoli)
    for c in scaled_coords:
        cv.rectangle(view, (int(c[0] - obst_width), int(c[1] - 4)), (int(c[0] + obst_width), int(c[1] + 4)), (0, 0, 0), cv.FILLED)
    # Visualizzazione dell'angolo calcolato
    cv.putText(img=view, text="{:.1f} deg".format(angle), org=(24, 320),
                fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (127, 127, 127), thickness = 5, lineType = cv.LINE_AA)
    # Approssimazione di una retta attraverso gli ostacoli scalati e disegno della stessa
    coef = np.polyfit(scaled_coords[:,0], scaled_coords[:,1], 1)
    poly1d_fn = np.poly1d(coef)
    l_start = (20, int(poly1d_fn(20)))
    l_end = (360, int(poly1d_fn(360)))
    cv.line(view, l_start, l_end, (0, 127, 0), 3)
    return view
##############################################################################################################
def main(numDisparities, blockSize, imageDim, display = False):
    df = pd.DataFrame(columns = ['Z(m)','Hdiff','Wdiff'])
    LCameraView = cv.VideoCapture('robotL.avi')
    RCameraView = cv.VideoCapture('robotR.avi')
    frame_counter = 0
    disparity_range = np.arange(numDisparities)
    try:
        while LCameraView.isOpened() and RCameraView.isOpened():
            frame_counter += 1
            Lret, frameL = LCameraView.read()
            Rret, frameR = RCameraView.read()
            if (not Lret) or (frameL is None) or (not Rret) or (frameR is None):
                LCameraView.release()
                RCameraView.release()
                print("Can't receive frame (stream end?). Exiting ...")
                break
            imgL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
            imgR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
            ######################################################
            disparity_map, _ = computeDisparityMap(imgL, imgR, disparity_range, blockSize, imageDim, M_SAD)
            mainDisparity = np.average(disparity_map)
            z = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            output_image = generateOutputImage(frameL, z, imageDim, MINIMUM_DISTANCE)
            ######################################################
            coords, angle = computeObstaclesCoords(disparity_map, 5)
            planar_view = drawPlanarView(coords, angle)
            ######################################################
            _, h, w = computeChessboard(imgL, imageDim)
            ######################################################
            if (h != None and w != None):
                HComputed = (z * h) / FOCAL_LENGHT
                WComputed = (z * w) / FOCAL_LENGHT
                Hdiff = abs(HComputed - H)
                Wdiff = abs(WComputed - W)
            else:
                Hdiff = None
                Wdiff = None
            ######################################################
            if display:
                plt.figure('Display'); 
                plt.clf()
                plt.subplot(1,2,1)
                plt.imshow(output_image, vmin=output_image.min(), vmax=output_image.max(), cmap='gray')
                plt.subplot(1,2,2)
                plt.imshow(planar_view, vmin=planar_view.min(), vmax=planar_view.max(), cmap='gray')
                plt.pause(0.000001)
            ######################################################
            df.loc[len(df)] = {
                'Z(m)' : z/1000,
                'Hdiff': Hdiff,
                'Wdiff': Wdiff
            }
            ######################################################
        df = df[df['Hdiff'] != None]
        df = df[df['Wdiff'] != None]
        df['Hdiff'] = df.Hdiff.rolling(SMOOTH).mean()
        df['Wdiff'] = df.Wdiff.rolling(SMOOTH).mean()
        plt.figure()
        plt.plot(df['Z(m)'], label='Z(m)', color='blue')
        plt.title("Z(m)")
        plt.figure()
        plt.plot(df['Hdiff'], label='Hdiff', color='blue')
        plt.title("Hdiff")
        plt.legend()
        plt.figure()
        plt.plot(df['Wdiff'], label='Wdiff', color='blue')
        plt.title("Wdiff")
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("Hdiff {}".format(df['Hdiff'].mean()))
        print("Wdiff {}".format(df['Wdiff'].mean()))
    except KeyboardInterrupt:
        LCameraView.release()
        RCameraView.release()
        print("Released Video Resource")
##############################################################################################################
def getParams():
    parser = argparse.ArgumentParser(prog='CVproject', description='Computer Vision Project')
    parser.add_argument('--imageDim',default='200', help='Image box dimension to cut from original frames', type=int)
    parser.add_argument('--numDisparities',default='128', help='Disparities number parameter for disparity map algorithm', type=int)
    parser.add_argument('--blockSize',default=BEST_BLOCKSIZE_VALUE, help='Block size parameter for disparity map algorithm', type=int)
    parser.add_argument('--display',default='False', help='Display the output', type=bool)
    return parser.parse_args()

if __name__ == "__main__":
    args = getParams()
    main(args.numDisparities, args.blockSize, args.imageDim, args.display)


