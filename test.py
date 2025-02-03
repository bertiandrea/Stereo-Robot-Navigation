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
    map = np.min(measure_maps, axis=-1)
    
    # Normalization
    map = (map - np.min(map)) / (np.max(map) - np.min(map))
    return map
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
    for i in range(0, dissimilarity_maps.shape[Y_AXIS]):
        for j in range(0, dissimilarity_maps.shape[X_AXIS]):
            min = np.min(dissimilarity_maps[i,j,:])
            index_min = np.argmin(dissimilarity_maps[i,j,:])
            temp = np.delete(dissimilarity_maps[i,j,:], index_min)
            second_min = np.min(temp)
            map[i,j] = min / second_min
            
    # Normalization
    map = (map - np.min(map)) / (np.max(map) - np.min(map))
    # Higher values indicate a better match (Second Best is really far from the best -> min/second_min is close to 0)
    # 1 = min, 100 = second_min -> 1/100 = 0.01 -> 1 - 0.01 = 0.99 -> Good match
    map = 1 - map
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
def applyLocalFilter(disparity_map, block_size=25):
    mean_map = cv.boxFilter(disparity_map.astype(np.float32), ddepth=-1, ksize=(block_size, block_size)) # E[X]
    mean_squared_map = cv.boxFilter(np.square(disparity_map.astype(np.float32)), ddepth=-1, ksize=(block_size, block_size)) # E[X^2]
    variance_map = mean_squared_map - np.square(mean_map) # Local Variance: E[X^2] - (E[X])^2

    # Normalization
    variance_map = (variance_map - np.min(variance_map)) / (np.max(variance_map) - np.min(variance_map))
    # Higher values indicate a better match (i.e., less variance)
    variance_map = 1 - variance_map
    return variance_map
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
################################################################################################################
def main(numDisparities, blockSize, imageDim, display = False):
    df = pd.DataFrame(columns = [
        'Hdiff_CV','Wdiff_CV',
        'Hdiff','Wdiff',
        'Hdiff_RATIO','Wdiff_RATIO',
        'Hdiff_MORAVEC','Wdiff_MORAVEC'
        'Hdiff_LOCAL','Wdiff_LOCAL'
        'Hdiff_CROSS','Wdiff_CROSS'
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
            mask = cv_disparity_map >= 0 # -1 values are not considered (no disparity)
            mainDisparity = np.average(cv_disparity_map[mask])
            z_cv = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            disparity_map, map = computeDisparityMap(imgL, imgR, disparity_range, blockSize, imageDim, M_SAD)
            mainDisparity = np.average(disparity_map)
            z = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            ratio_map = secondBestRatio(map)
            ratio_mask = ratio_map >= np.percentile(ratio_map, 70)
            mainDisparity = np.average(disparity_map[ratio_mask])
            z_ratio = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            moravec_map = moravecOperator(imgL, imageDim, blockSize, M_SAD)
            moravec_mask = moravec_map >= np.percentile(moravec_map, 70)
            mainDisparity = np.average(disparity_map[moravec_mask])
            z_moravec = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            local_map = applyLocalFilter(disparity_map, blockSize)
            local_mask = local_map >= np.percentile(local_map, 70)
            mainDisparity = np.average(disparity_map[local_mask])
            z_local = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            disparity_map_R, _ = computeDisparityMap(imgR, imgL, disparity_range, blockSize, imageDim, M_SAD)
            cross_map = applyCrossCheck(disparity_map, disparity_map_R)
            cross_mask = cross_map >= np.percentile(cross_map, 70)
            mainDisparity = np.average(disparity_map[cross_mask])
            z_cross = (FOCAL_LENGHT * BASELINE) / mainDisparity
            ######################################################
            output_image = generateOutputImage(frameL, z, imageDim, MINIMUM_DISTANCE)
            ######################################################
            coords, angle = computeObstaclesCoords(disparity_map, 5)
            planar_view = drawPlanarView(coords, angle)
            ######################################################
            _, h, w = computeChessboard(imgL, imageDim)
            ######################################################
            if (h != None and w != None):
                HComputed = (z_cv * h) / FOCAL_LENGHT
                WComputed = (z_cv * w) / FOCAL_LENGHT
                Hdiff_CV = abs(HComputed - H)
                Wdiff_CV = abs(WComputed - W)
                HComputed = (z * h) / FOCAL_LENGHT
                WComputed = (z * w) / FOCAL_LENGHT
                Hdiff = abs(HComputed - H)
                Wdiff = abs(WComputed - W)
                HComputed = (z_ratio * h) / FOCAL_LENGHT
                WComputed = (z_ratio * w) / FOCAL_LENGHT
                Hdiff_RATIO = abs(HComputed - H)
                Wdiff_RATIO = abs(WComputed - W)
                HComputed = (z_moravec * h) / FOCAL_LENGHT
                WComputed = (z_moravec * w) / FOCAL_LENGHT
                Hdiff_MORAVEC = abs(HComputed - H)
                Wdiff_MORAVEC = abs(WComputed - W)
                HComputed = (z_local * h) / FOCAL_LENGHT
                WComputed = (z_local * w) / FOCAL_LENGHT
                Hdiff_LOCAL = abs(HComputed - H)
                Wdiff_LOCAL = abs(WComputed - W)
                HComputed = (z_cross * h) / FOCAL_LENGHT
                WComputed = (z_cross * w) / FOCAL_LENGHT
                Hdiff_CROSS = abs(HComputed - H)
                Wdiff_CROSS = abs(WComputed - W)
            else:
                Hdiff_CV = None
                Wdiff_CV = None
                Hdiff = None
                Wdiff = None
                Hdiff_RATIO = None
                Wdiff_RATIO = None
                Hdiff_MORAVEC = None
                Wdiff_MORAVEC = None
                Hdiff_LOCAL = None
                Wdiff_LOCAL = None
                Hdiff_CROSS = None
                Wdiff_CROSS = None
            ######################################################
            if(display):
                plt.figure('Display'); 
                plt.clf()
                plt.subplot(2,4,1)
                plt.imshow(output_image)
                plt.subplot(2,4,2)
                plt.imshow(cv_disparity_map, vmin=cv_disparity_map.min(), vmax=cv_disparity_map.max(), cmap='gray')
                plt.subplot(2,4,3)
                plt.imshow(disparity_map, vmin=disparity_map.min(), vmax=disparity_map.max(), cmap='gray')
                plt.subplot(2,4,4)
                plt.imshow(planar_view)
                plt.subplot(2,4,5)
                plt.imshow(ratio_map, vmin=ratio_map.min(), vmax=ratio_map.max(), cmap='gray')
                plt.subplot(2,4,6)
                plt.imshow(moravec_map, vmin=moravec_map.min(), vmax=moravec_map.max(), cmap='gray')
                plt.subplot(2,4,7)
                plt.imshow(local_map, vmin=local_map.min(), vmax=local_map.max(), cmap='gray')
                plt.subplot(2,4,8)
                plt.imshow(cross_map, vmin=cross_map.min(), vmax=cross_map.max(), cmap='gray')
                plt.pause(0.000001)
            ######################################################
            # Update dataframe with current frame infos
            df.loc[len(df)] = {
                'Hdiff_CV': Hdiff_CV,
                'Wdiff_CV': Wdiff_CV,
                'Hdiff': Hdiff,
                'Wdiff': Wdiff,
                'Hdiff_RATIO': Hdiff_RATIO,
                'Wdiff_RATIO': Wdiff_RATIO,
                'Hdiff_MORAVEC': Hdiff_MORAVEC,
                'Wdiff_MORAVEC': Wdiff_MORAVEC,
                'Hdiff_LOCAL': Hdiff_LOCAL,
                'Wdiff_LOCAL': Wdiff_LOCAL,
                'Hdiff_CROSS': Hdiff_CROSS,
                'Wdiff_CROSS': Wdiff_CROSS
            }
            ######################################################
            print("Main Disparity: {}".format(mainDisparity))
            disparity_range = computeDisparityRange(mainDisparity)
            print("Disparity Range: {} - {}".format(disparity_range[0],disparity_range[-1]))
            ######################################################
        df= df[df['Hdiff_CV'] != None]
        df= df[df['Wdiff_CV'] != None]
        df= df[df['Hdiff'] != None]
        df= df[df['Wdiff'] != None]
        df= df[df['Hdiff_RATIO'] != None]
        df= df[df['Wdiff_RATIO'] != None]
        df= df[df['Hdiff_MORAVEC'] != None]
        df= df[df['Wdiff_MORAVEC'] != None]
        df= df[df['Hdiff_LOCAL'] != None]
        df= df[df['Wdiff_LOCAL'] != None]
        df= df[df['Hdiff_CROSS'] != None]
        df= df[df['Wdiff_CROSS'] != None]
        df['Hdiff_CV'] = df.Hdiff_CV.rolling(SMOOTH).mean()
        df['Wdiff_CV'] = df.Wdiff_CV.rolling(SMOOTH).mean()
        df['Hdiff'] = df.Hdiff.rolling(SMOOTH).mean()
        df['Wdiff'] = df.Wdiff.rolling(SMOOTH).mean()
        df['Hdiff_RATIO'] = df.Hdiff_RATIO.rolling(SMOOTH).mean()
        df['Wdiff_RATIO'] = df.Wdiff_RATIO.rolling(SMOOTH).mean()
        df['Hdiff_MORAVEC'] = df.Hdiff_MORAVEC.rolling(SMOOTH).mean()
        df['Wdiff_MORAVEC'] = df.Wdiff_MORAVEC.rolling(SMOOTH).mean()
        df['Hdiff_LOCAL'] = df.Hdiff_LOCAL.rolling(SMOOTH).mean()
        df['Wdiff_LOCAL'] = df.Wdiff_LOCAL.rolling(SMOOTH).mean()
        df['Hdiff_CROSS'] = df.Hdiff_CROSS.rolling(SMOOTH).mean()
        df['Wdiff_CROSS'] = df.Wdiff_CROSS.rolling(SMOOTH).mean()
        plt.figure()
        plt.plot(df['Hdiff_CV'], label='Hdiff_CV', color='orange')
        plt.plot(df['Hdiff'], label='Hdiff', color='blue')
        plt.plot(df['Hdiff_RATIO'], label='Hdiff_RATIO', color='green')
        plt.plot(df['Hdiff_MORAVEC'], label='Hdiff_MORAVEC', color='red')
        plt.plot(df['Hdiff_LOCAL'], label='Hdiff_LOCAL', color='purple')
        plt.plot(df['Hdiff_CROSS'], label='Hdiff_CROSS', color='black')
        plt.title("Hdiff")
        plt.legend()
        plt.figure()
        plt.plot(df['Wdiff_CV'], label='Wdiff_CV', color='orange')
        plt.plot(df['Wdiff'], label='Wdiff', color='blue')
        plt.plot(df['Wdiff_RATIO'], label='Wdiff_RATIO', color='green')
        plt.plot(df['Wdiff_MORAVEC'], label='Wdiff_MORAVEC', color='red')
        plt.plot(df['Wdiff_LOCAL'], label='Wdiff_LOCAL', color='purple')
        plt.plot(df['Wdiff_CROSS'], label='Wdiff_CROSS', color='black')
        plt.title("Wdiff")
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("Hdiff_CV {}".format(df['Hdiff_CV'].mean()))
        print("Hdiff {}".format(df['Hdiff'].mean()))
        print("Hdiff_RATIO {}".format(df['Hdiff_RATIO'].mean()))
        print("Hdiff_MORAVEC {}".format(df['Hdiff_MORAVEC'].mean()))
        print("Hdiff_LOCAL {}".format(df['Hdiff_LOCAL'].mean()))
        print("Hdiff_CROSS {}".format(df['Hdiff_CROSS'].mean()))
        print("Wdiff_CV {}".format(df['Wdiff_CV'].mean()))
        print("Wdiff {}".format(df['Wdiff'].mean()))
        print("Wdiff_RATIO {}".format(df['Wdiff_RATIO'].mean()))
        print("Wdiff_MORAVEC {}".format(df['Wdiff_MORAVEC'].mean()))
        print("Wdiff_LOCAL {}".format(df['Wdiff_LOCAL'].mean()))
        print("Wdiff_CROSS {}".format(df['Wdiff_CROSS'].mean()))
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


