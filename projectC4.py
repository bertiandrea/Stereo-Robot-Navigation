import matplotlib.pyplot as plt
import argparse
import cv2 as cv
import pandas as pd
import numpy as np

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
def applyCrossCheck(disparity_map_L, disparity_map_R):
    map = np.zeros_like(disparity_map_L)
    j_indices = np.arange(disparity_map_L.shape[X_AXIS])
    x_shifted = j_indices - disparity_map_L
    valid_mask = (x_shifted >= 0) & (x_shifted < disparity_map_R.shape[X_AXIS])
    for i in range(disparity_map_L.shape[Y_AXIS]):
        for j in range(disparity_map_L.shape[X_AXIS]):
            if valid_mask[i,j]:
                map[i,j] = abs(disparity_map_L[i,j] - disparity_map_R[i, x_shifted[i,j]])

    # Normalization
    map = (map - np.min(map[valid_mask])) / (np.max(map[valid_mask]) - np.min(map[valid_mask]))
    # Higher values indicate a better match (i.e., less disparity difference)
    map[valid_mask] = 1 - map[valid_mask]
    return map
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
            disparity_map_L, _ = computeDisparityMap(imgL, imgR, disparity_range, blockSize, imageDim, M_SAD)
            ######################################################
            disparity_range = np.arange(-disparity_range[-1], -disparity_range[0]) # Invert Disparity Range
            disparity_map_R, _ = computeDisparityMap(imgR, imgL, disparity_range, blockSize, imageDim, M_SAD)
            disparity_map_R = np.abs(disparity_map_R) # Invert Disparity values
            map = applyCrossCheck(disparity_map_L, disparity_map_R)
            mask = map >= np.percentile(map, 70)
            mainDisparity = np.average(disparity_map_L[mask])
            z = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            imgChessboard, h, w = computeChessboard(imgL, imageDim)
            ######################################################
            if (h != None and w != None):
                HComputed = (z * h) / FOCAL_LENGHT
                WComputed = (z * w) / FOCAL_LENGHT
                print("HComputed {}".format(HComputed), "WComputed {}".format(WComputed))
                Hdiff = abs(HComputed - H)
                Wdiff = abs(WComputed - W)
                print("Hdiff {}".format(Hdiff), "Wdiff {}".format(Wdiff))
            else:
                Hdiff = None
                Wdiff = None
            ######################################################
            if display:
                plt.figure('Display'); 
                plt.clf()
                plt.subplot(1,3,1)
                plt.imshow(map, vmin=map.min(), vmax=map.max(), cmap='gray')
                plt.subplot(1,3,2)
                plt.imshow(disparity_map_L, vmin=disparity_map_L.min(), vmax=disparity_map_L.max(), cmap='gray')
                plt.subplot(1,3,3)
                plt.imshow(imgChessboard, cmap='gray')
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
    parser.add_argument('--display', help='Display the results', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = getParams()
    main(args.numDisparities, args.blockSize, args.imageDim, args.display)


