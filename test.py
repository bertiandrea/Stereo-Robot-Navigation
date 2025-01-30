import matplotlib.pyplot as plt
import argparse
import cv2 as cv
import pandas as pd
import numpy as np
import math
from IPython.display import clear_output
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
MINIMUM_DISTANCE = 800
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
##############################################################################################################
def computeCVDisparityMap(imgL, imgR, numDisparities, blockSize, imageDim):
    stereoMatcher = cv.StereoBM_create(numDisparities = numDisparities, blockSize = blockSize)
    disparity_map = stereoMatcher.compute(imgL, imgR).astype(np.float32) / 16.0
    disparity_map = cutMatrix(disparity_map, imageDim)
    return disparity_map
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
def moravecOperator(image, imgDim, block_size, measure = M_SAD):
    direction = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    measure_maps = np.zeros((imgDim, imgDim, len(direction)), dtype=int)
    image_f = np.float32(image)
    kernel = np.ones([block_size, block_size])
    
    translation = np.float32([[1, 0, -(image.shape[X_AXIS]-imgDim)//2], [0, 1, -(image.shape[Y_AXIS]-imgDim)//2]])
    ROI = cv.warpAffine(image_f, translation, (imgDim, imgDim))
    for i in range(len(direction)):
        u, v = direction[i]
        dir_translation = translation + ([[0,0,u],[0,0,v]])
        dir_ROI = cv.warpAffine(image_f, dir_translation, (imgDim, imgDim))
        if measure == M_SAD:
            differences = np.square(ROI-dir_ROI)
        else: # M_SAD
            differences = np.abs(ROI-dir_ROI)      
        measure_map = cv.filter2D(src=differences, ddepth=-1, kernel=kernel, borderType=cv.BORDER_ISOLATED)
        measure_maps[:, :, i] = measure_map
    return np.min(measure_maps, axis=-1)
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
def computeDisparityRange(main_disparity):
    if main_disparity < 32:
        return range(0, 64)
    else:
        return range(int(main_disparity)-32, int(main_disparity)+32)
##############################################################################################################    
def secondBestRatio(dissimilarity_maps):
    map = np.zeros_like(dissimilarity_maps[:,:,0])
    for i in range(0, dissimilarity_maps.shape[0]):
        for j in range(0, dissimilarity_maps.shape[1]):
            min = np.min(dissimilarity_maps[i,j,:])
            index_min = np.argmin(dissimilarity_maps[i,j,:])
            temp = np.delete(dissimilarity_maps[i,j,:], index_min)
            second_min = np.min(temp)
            map[i,j] = min/second_min
    return map
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

def computeObstaclesCoords(disparity_map, num_stripes):
    height, width = disparity_map.shape
    stripe_width = width // num_stripes
    # Calcolo della disparità media per ogni striscia
    main_disparities = []
    for i in range(0, num_stripes):
        stripe = disparity_map[:, stripe_width * i : stripe_width * (i + 1)]
        stripe_main_disparity = np.average(stripe)
        main_disparities.append(stripe_main_disparity)
    # Determinazione delle coordinate 2D dei punti centrali delle strisce nell'immagine
    frame_points = np.zeros((num_stripes, 2))
    for i in range(num_stripes):
        frame_points[i] = [stripe_width * (i + 0.5), height / 2]  # Punto centrale della striscia
    # Conversione delle coordinate dell'immagine al sistema con origine al centro dell'immagine
    image_points = frame_points - [width / 2, height / 2]
    # Conversione delle coordinate dall'immagine alle coordinate reali (XZ)
    XZ_coords = np.zeros((num_stripes, 2))
    for i in range(0, num_stripes):
        Z = (BASELINE * FOCAL_LENGHT) / main_disparities[i]
        X = image_points[i, 0] * Z / FOCAL_LENGHT
        XZ_coords[i] =  [X, Z]
    # Approssimazione di una retta attraverso i punti XZ e calcolo dell'angolo
    coef = np.polyfit(XZ_coords[:, 0], XZ_coords[:, 1], 1)
    angle_deg = math.degrees(math.atan(coef[0]))
    # Normalizzazione delle coordinate tra 0 e 1
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
    scaled_coords = norm_coords * 320 + 40  # Scalatura delle coordinate per la visualizzazione
    for c in scaled_coords:
        cv.rectangle(view, (int(c[0] - obst_width), int(c[1] - 4)), (int(c[0] + obst_width), int(c[1] + 4)), (0, 0, 0), cv.FILLED)
    # Disegno dell'angolo
    cv.putText(img=view, text="t={:.1f} deg".format(angle), org=(24, 320),
                fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (127, 127, 127), thickness = 5, lineType = cv.LINE_AA)
    # Disegno della retta che approssima gli ostacoli
    coef = np.polyfit(scaled_coords[:,0], scaled_coords[:,1], 1)
    poly1d_fn = np.poly1d(coef)
    l_start = (20, int(poly1d_fn(20)))
    l_end = (360, int(poly1d_fn(360)))
    cv.line(view, l_start, l_end, (0, 127, 0), 3)

    return view
def main(numDisparities, blockSize, imageDim, display = False):
    df = pd.DataFrame(columns = [
        'Hdiff_cv_disparity','Wdiff_cv_disparity',
        'Hdiff_disparity','Wdiff_disparity',
        'Hdiff_ratio','Wdiff_ratio',
        'Hdiff_moravec','Wdiff_moravec'
        ])
    LCameraView = cv.VideoCapture('robotL.avi')
    RCameraView = cv.VideoCapture('robotR.avi')
    frame_counter = 0
    disparity_range = range(0, numDisparities) #Starting Disparity Range
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
            cv_disparity_map = computeCVDisparityMap(imgL, imgR, numDisparities, blockSize, imageDim)
            mask = cv_disparity_map > 0 # -1 values are not considered (no disparity)
            mainDisparity = np.average(cv_disparity_map[mask])
            z_cv = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            disparity_map, map = computeDisparityMap(imgL, imgR, disparity_range, blockSize, imageDim, M_SAD)
            mainDisparity = np.average(disparity_map)
            z_disparity = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            ratio_map = secondBestRatio(map)
            ratio_mask = ratio_map > np.percentile(ratio_map, 70)
            mainDisparity = np.average(disparity_map[ratio_mask])
            z_ratio = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            moravec_map = moravecOperator(imgL, imageDim, blockSize, M_SAD)
            moravec_mask = moravec_map >= np.percentile(moravec_map, 70)
            mainDisparity = np.average(disparity_map[moravec_mask])
            z_moravec = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            _, h, w = computeChessboard(imgL, imageDim)
            ######################################################
            if (h != None and w != None):
                HComputed = (z_cv * h) / FOCAL_LENGHT
                WComputed = (z_cv * w) / FOCAL_LENGHT
                Hdiff_cv_disparity = abs(HComputed - H)
                Wdiff_cv_disparity = abs(WComputed - W)
                HComputed = (z_disparity * h) / FOCAL_LENGHT
                WComputed = (z_disparity * w) / FOCAL_LENGHT
                Hdiff_disparity = abs(HComputed - H)
                Hdiff_disparity = abs(WComputed - W)
                HComputed = (z_ratio * h) / FOCAL_LENGHT
                WComputed = (z_ratio * w) / FOCAL_LENGHT
                Hdiff_ratio = abs(HComputed - H)
                Wdiff_ratio = abs(WComputed - W)
                HComputed = (z_moravec * h) / FOCAL_LENGHT
                WComputed = (z_moravec * w) / FOCAL_LENGHT
                Hdiff_moravec = abs(HComputed - H)
                Wdiff_moravec = abs(WComputed - W)
            else:
                Hdiff_cv_disparity = None
                Wdiff_cv_disparity = None
                Hdiff_disparity = None
                Wdiff_disparity = None
                Hdiff_ratio = None
                Wdiff_ratio = None
                Hdiff_moravec = None
                Wdiff_moravec = None
            ######################################################
            output_image = generateOutputImage(frameL, z_disparity, imageDim, MINIMUM_DISTANCE)
            coords, angle = computeObstaclesCoords(disparity_map, 5)
            planar_view = drawPlanarView(coords, angle)
            ######################################################
            if(display):
                plt.figure('Display'); 
                plt.clf()
                plt.subplot(2,3,1)
                plt.imshow(output_image)
                plt.subplot(2,3,2)
                plt.imshow(cv_disparity_map, vmin=cv_disparity_map.min(), vmax=cv_disparity_map.max(), cmap='gray')
                plt.subplot(2,3,3)
                plt.imshow(disparity_map, vmin=disparity_map.min(), vmax=disparity_map.max(), cmap='gray')
                plt.subplot(2,3,4)
                plt.imshow(ratio_mask, vmin=ratio_mask.min(), vmax=ratio_mask.max(), cmap='gray')
                plt.subplot(2,3,5)
                plt.imshow(moravec_mask, vmin=moravec_mask.min(), vmax=moravec_mask.max(), cmap='gray')
                plt.subplot(2,3,6)
                plt.imshow(planar_view)
                plt.title('Display')
                plt.pause(0.000001)
            ######################################################
            # Update dataframe with current frame infos
            df.loc[len(df)] = {
                'Hdiff_cv_disparity': Hdiff_cv_disparity,
                'Wdiff_cv_disparity': Wdiff_cv_disparity,
                'Hdiff_disparity': Hdiff_disparity,
                'Wdiff_disparity': Wdiff_disparity,
                'Hdiff_ratio': Hdiff_ratio,
                'Wdiff_ratio': Wdiff_ratio,
                'Hdiff_moravec': Hdiff_moravec,
                'Wdiff_moravec': Wdiff_moravec
            }
            ######################################################
            print("Main Disparity: {}".format(mainDisparity))
            disparity_range = computeDisparityRange(mainDisparity)
            print("Disparity Range: {} - {}".format(disparity_range[0],disparity_range[-1]))
            ######################################################
        df= df[df['Hdiff_cv_disparity'] != None]
        df= df[df['Wdiff_cv_disparity'] != None]
        df= df[df['Hdiff_disparity'] != None]
        df= df[df['Wdiff_disparity'] != None]
        df= df[df['Hdiff_ratio'] != None]
        df= df[df['Wdiff_ratio'] != None]
        df= df[df['Hdiff_moravec'] != None]
        df= df[df['Wdiff_moravec'] != None]
        df['Hdiff_cv_disparity'] = df.Hdiff_cv_disparity.rolling(SMOOTH).mean()
        df['Wdiff_cv_disparity'] = df.Wdiff_cv_disparity.rolling(SMOOTH).mean()
        df['Hdiff_disparity'] = df.Hdiff_disparity.rolling(SMOOTH).mean()
        df['Wdiff_disparity'] = df.Wdiff_disparity.rolling(SMOOTH).mean()
        df['Hdiff_ratio'] = df.Hdiff_ratio.rolling(SMOOTH).mean()
        df['Wdiff_ratio'] = df.Wdiff_ratio.rolling(SMOOTH).mean()
        df['Hdiff_moravec'] = df.Hdiff_moravec.rolling(SMOOTH).mean()
        df['Wdiff_moravec'] = df.Wdiff_moravec.rolling(SMOOTH).mean()
        # Plotting dataframe values
        df.plot(subplots=True)
        plt.tight_layout()
        plt.show()
        # Print global values
        print("GlobalHdiffErrorCVDisparity {}".format(df['Hdiff_cv_disparity'].mean()))
        print("GlobalWdiffErrorCVDisparity {}".format(df['Wdiff_cv_disparity'].mean()))
        print("GlobalHdiffErrorDisparity {}".format(df['Hdiff_disparity'].mean()))
        print("GlobalWdiffErrorDisparity {}".format(df['Wdiff_disparity'].mean()))
        print("GlobalHdiffErrorRatio {}".format(df['Hdiff_ratio'].mean()))
        print("GlobalWdiffErrorRatio {}".format(df['Wdiff_ratio'].mean()))
        print("GlobalHdiffErrorMoravec {}".format(df['Hdiff_moravec'].mean()))
        print("GlobalWdiffErrorMoravec {}".format(df['Wdiff_moravec'].mean()))
    except KeyboardInterrupt:
        LCameraView.release()
        RCameraView.release()
        print("Released Video Resource")

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


