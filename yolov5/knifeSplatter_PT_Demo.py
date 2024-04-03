

import numpy as np
import cv2
#import pandas as pd
import os
import time
import torch
#from line_profiler import LineProfiler
#from vidstab import VidStab
import argparse

#for application, camera will need to be positioned so the knives/walls are horizontal and not angled
#walls can't be inside the splatter measurement region. When knives are low, there is a differential between knives and walls
#method 1: train yolov5 on horizontal edges and use to redefine splatter measurement area
#method 2: set splatter measurement area a few pixels lower if underside is not present. If underside is present, line is already good.

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a video file.")
    parser.add_argument('--model', type=str, required=True, help='Path to the weights file.')
    parser.add_argument('--source', type=str, required=True, help='Path to the video source.')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    video_path = args.source
    model_path = args.model
    
    model = torch.hub.load(model_path, source='local', force_reload=True, device='cuda:0')
    cap = cv2.VideoCapture(video_path)
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Create the rotation matrix
    angle_degrees = 3
    center = (frame_width // 2, frame_height // 2)
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)


    
    i = 1
    infNum = 1
    
    fgbgC = cv2.bgsegm.createBackgroundSubtractorCNT() #GOOD
    fgbgC.setMinPixelStability(1)
    fgbgC.setMaxPixelStability(10)
    fgbgC.setIsParallel(True)
    fgbgC.setUseHistory(True)

 #   resultsTable = pd.DataFrame(columns=['frameNumber','splatterAmount(px)', 'splatterWidth(px)', 'splatterSeverity', 'frameTime(s)'])
    
    frameNumber = 0
    frameNo = 0
    previousContours = []
    similarityCounter = []
    airKnifePoss = []
    airKnifePosCount = 0
    scalingFactors = []
    scalingFactorCount = 0
  #  stabilizer = VidStab()
    stabCount = 0
    while(1):
        
        startTime = time.time()
        
        ret, frame = cap.read()
        
        if ret == False:
            break
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_width, frame_height))
        frame = rotated_frame
        scale_percent = 100 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
          
        # resize image
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        
        
        
        results = model(frame)
        #results.save(infNum=infNum, save_dir='C:/Users/Callum/Anaconda3/envs/splatterValidation/topNewYOLO')
        pandasResults = results.pandas().xyxy[0]
        #print("pandasResults:", pandasResults)
    

        pandasResultsOrdered = pandasResults.sort_values(['confidence'], ascending=False)
        #print("pandasResultsOrdered: ", pandasResultsOrdered)
        lowConfList = pandasResultsOrdered.index[pandasResultsOrdered['confidence']<0.8].tolist()
        #print("lcl: ",lowConfList)
        pandasResultsOrdered.drop(lowConfList,axis=0)
        #print("pandasResultsOrderedDropped: ", pandasResultsOrdered)
        existingClasses = pandasResultsOrdered.loc[:,'class']
        existingClasses = existingClasses.to_numpy()
        #print("existing", existingClasses)
        if 1 in existingClasses:
            undersideExists = 1
            #print("undersideexists")
        else:
            undersideExists = 0
        
        
        #print(pandasResultsOrdered)
        if undersideExists == 1:
            #print("error C")
            pandasResultsOrderedKnives = pandasResultsOrdered.iloc[:4]
        else:
            pandasResultsOrderedKnives = pandasResultsOrdered.iloc[:2]
            
        airKnifePos = max(pandasResultsOrderedKnives.loc[:,'ymax'])
        

        #print("airknifepos: ", airKnifePos)
        airKnifePoss.append(airKnifePos)
        airKnifePosCount += 1
        if 0 < airKnifePosCount < 10:
            maAirKnifePos = airKnifePoss[0]
        elif airKnifePosCount % 10 == 0:
            maAirKnifePos = sum(airKnifePoss[-10:])/10
        else:
            maAirKnifePos = maAirKnifePos
        
        knifeFace1 = pandasResultsOrderedKnives.iloc[0]
        knifeFace2 = pandasResultsOrderedKnives.iloc[1]
        knifeFace1_xmin = knifeFace1[0]
        knifeFace1_ymin = knifeFace1[1]
        knifeFace1_xmax = knifeFace1[2]
        knifeFace1_ymax = knifeFace1[3]
        
        knifeFace2_xmin = knifeFace2[0]
        knifeFace2_ymin = knifeFace2[1]
        knifeFace2_xmax = knifeFace2[2]
        knifeFace2_ymax = knifeFace2[3]

        if undersideExists == 1:
            try:
                #print("error A")
                knifeUnderside1 = pandasResultsOrderedKnives.iloc[2]
                knifeUnderside2 = pandasResultsOrderedKnives.iloc[3]
                knifeUnderside1_xmin = knifeUnderside1[0]
                knifeUnderside1_ymin = knifeUnderside1[1]
                knifeUnderside1_xmax = knifeUnderside1[2]
                knifeUnderside1_ymax = knifeUnderside1[3]
                
                knifeUnderside2_xmin = knifeUnderside2[0]
                knifeUnderside2_ymin = knifeUnderside2[1]
                knifeUnderside2_xmax = knifeUnderside2[2]
                knifeUnderside2_ymax = knifeUnderside2[3]
            except:
                #print("error B")
                continue
        
        bboxCombArea = ((knifeFace1_xmax - knifeFace1_xmin) * (knifeFace1_ymax - knifeFace1_ymin)) + ((knifeFace2_xmax - knifeFace2_xmin) * (knifeFace2_ymax - knifeFace2_ymin))
        
        scalingFactor = round(bboxCombArea/(80000*(scale_percent/100)*(scale_percent/100)),1)
        scalingFactors.append(scalingFactor)
        scalingFactorCount += 1
        if 0 < scalingFactorCount < 25000:
            maScalingFactor = scalingFactors[0]
        elif scalingFactorCount % 25000 == 0:
            maScalingFactor = sum(scalingFactors[-25000:])/25000
        else:
            maScalingFactor = maScalingFactor
        
        
        #print("Frame: ", frameNo)
        frameNo += 1
        infNum += 1
        
        #print("frameNo: ", frameNo)
        #every time line moves, algorithms reset meaning whole frame is background for 1 frame
        
        #print("NEW FRAME --------------------------------------------------------------------------------------")
        
        #fgmask = fgbg.apply(frame[int(maAirKnifePos):,:])
        fgmaskC = fgbgC.apply(frame)

        #ERODE SMALL CONTOURS TO REMOVE NOISE FOR CNT SUBTRACTION
        kernel = np.ones((2, 2), np.uint8)
        #fgmaskM = cv2.erode(fgmaskM, kernel) 
        fgmaskC = cv2.erode(fgmaskC, kernel) 
        fgmask2 = fgmaskC.copy()
        fgmask2 = np.zeros_like(fgmask2)
        fgmask3 = fgmask2.copy()
    
    
        pContourSize = []
        contourSize = []
        pContourPosition = []
        contourPosition = []
        pM = []
        M = []
        pcX = []
        pcY = []
        cX = []
        cY = []
        Q= 0
        #print(frame.shape[0],frame.shape[1])
        #i1 = (0, 400)
        
        leftMostSide = np.min([knifeFace1_xmin,knifeFace1_xmax, knifeFace2_xmin, knifeFace2_xmax])
        rightMostSide = np.max([knifeFace1_xmin,knifeFace1_xmax, knifeFace2_xmin, knifeFace2_xmax])
        #print(leftMostSide, rightMostSide)
        #print(fgmask2.shape)
        if undersideExists == 0:         
            i1 = (0, int(maAirKnifePos+(28*maScalingFactor)))
            i2 = (leftMostSide+25, int(maAirKnifePos+(28*maScalingFactor)))
            i3 = (leftMostSide+25, int(maAirKnifePos))
            i4 = (rightMostSide-25, int(maAirKnifePos))
            i5 = (rightMostSide-25, int(maAirKnifePos+(28*maScalingFactor)))
            i6 = (frame.shape[1], int(maAirKnifePos+(28*maScalingFactor)))
            i7 = (frame.shape[1], frame.shape[0])
            i8 = (0, frame.shape[0])
            
            
            improvedContour = np.array([i1,i2,i3,i4,i5,i6,i7,i8], dtype=np.int32)
            
            

        else:
            i1 = (0, int(maAirKnifePos+(23*maScalingFactor)))
            i2 = (frame.shape[1], int(maAirKnifePos+(23*maScalingFactor)))
            i3 = (frame.shape[1], frame.shape[0])
            i4 = (0, frame.shape[0])     
            improvedContour = np.array([i1,i2,i3,i4])
        

        #print(improvedContour)
        #print(len(improvedContour))
        
        improvedContourFlipped = [[y, x] for x, y in improvedContour]
        
        
        cv2.drawContours(fgmask2, [improvedContour], -1, 255, -3)
        splatterArea = np.where(fgmask2 == 255)

        #print("III: ", i1, i2, i3, i4)
        #COUNT WHITE PIXELS
        whitePixels = np.sum(fgmaskC[splatterArea] == 255)
        #print("WHITEPIXELS: ", whitePixels)
        splatterWidth = 0
        splatterSeverity = 0
        if whitePixels < (1500000 * (scale_percent / 100)):
            
            #CALCULATE SPLATTER WIDTH
            maxX = 0
            maxY = 0
            minX = 0
            minY = 0
            leftSplatter = 0
            rightSplatter = 0

            fgmaskC = cv2.bitwise_and(fgmaskC, fgmask2)
            contours, hierarchy = cv2.findContours(fgmaskC,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                 
            #mask = np.ones(fgmask.shape[:2], dtype="uint8") * 255
            #LOOP THROUGH ALL CONTOURS, FIND LEFTMOST AND RIGHTMOST PIXELS TO CALCULATE SPLATTER WIDTH
            for c in contours:

                if len(c[:,0,0]) > 75*(scale_percent/100):
                #cv2.drawContours(fgmask[300:,:], c, -1, color=(155), thickness=2)
                    maxXc = np.max(c[:,0,0])
                    minXc = np.min(c[:,0,0])
                    if maxXc > maxX:
                        maxX = maxXc
                    if (minX == 0 and minXc != 0) or minXc < minX:
                        minX = minXc
                    leftSplatter = (frame.shape[1]/2) - minX
                    rightSplatter =  maxX - (frame.shape[1]/2)
                    
    
                    
                    if leftSplatter < 0:
                        splatterWidth = rightSplatter
                    if rightSplatter < 0:
                        splatterWidth = leftSplatter
                    else:
                        splatterWidth = leftSplatter+rightSplatter
                    Q += 1
                    previousContours = contours     
        
        else:
            whitePixels = 0
            
        knifeCoverage = maAirKnifePos/frame.shape[0]
        
        amount0 = 7500 * maScalingFactor * (scale_percent / 100) * (scale_percent / 100)# * 0.75# * knifeCoverage
        amount1 = 15000 * maScalingFactor * (scale_percent / 100) * (scale_percent / 100)# * 0.75# * knifeCoverage
        amount2 = 22500 * maScalingFactor * (scale_percent / 100) * (scale_percent / 100)# * 0.75# * knifeCoverage
        amount3 = 30000 * maScalingFactor * (scale_percent / 100) * (scale_percent / 100)# * 0.75# * knifeCoverage
        width0 = 200 * maScalingFactor * (scale_percent / 100) * (scale_percent / 100)# * 0.75# * knifeCoverage
        width1 = 400 * maScalingFactor * (scale_percent / 100) * (scale_percent / 100)# * 0.75# * knifeCoverage
        width2 = 600 * maScalingFactor * (scale_percent / 100) * (scale_percent / 100)# * 0.75# * knifeCoverage
        width3 = 800 * maScalingFactor * (scale_percent / 100) * (scale_percent / 100)# * 0.75# * knifeCoverage
        if width3 > 1920 * (scale_percent / 100):
            width3 = 1920 * (scale_percent / 100)
        
        #print("MA SCALING FACTOR: ", maScalingFactor)
        fgmask3 = cv2.putText(fgmask3, ("Scaling Factor: %.1f" %scalingFactor), (int(1400 * (scale_percent / 100)), int(180 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
        fgmask3 = cv2.putText(fgmask3, ("MA Scaling Factor: %.1f" %maScalingFactor), (int(1400 * (scale_percent / 100)), int(220 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)  
        if ((0 <= whitePixels <= amount0) and (0 <= splatterWidth <= width0)):
                splatterSeverity = 0
                fgmask3 = cv2.putText(fgmask3, ("Splatter Severity: 0"), (int(120 * (scale_percent / 100)),int(180 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
                fgmask3 = cv2.putText(fgmask3, ("Splatter quantity: %d" %whitePixels), (int(120 * (scale_percent / 100)),int(220 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
                fgmask3 = cv2.putText(fgmask3, ("Splatter width: %d" %splatterWidth), (int(120 * (scale_percent / 100)),int(260 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
        elif ((amount0 <= whitePixels <= amount1) and (width0 <= splatterWidth <= width1)) or ((amount0 <= whitePixels <= amount1) and (0 <= splatterWidth <= width0)) or ((0 <= whitePixels <= amount0) and (width0 <= splatterWidth <= width1)):
                splatterSeverity = 1
                fgmask3 = cv2.putText(fgmask3, ("Splatter Severity: 1"), (int(120 * (scale_percent / 100)),int(180 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
                fgmask3 = cv2.putText(fgmask3, ("Splatter quantity: %d" %whitePixels), (int(120 * (scale_percent / 100)),int(220 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
                fgmask3 = cv2.putText(fgmask3, ("Splatter width: %d" %splatterWidth), (int(120 * (scale_percent / 100)),int(260 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA) 
        elif ((amount1 <= whitePixels <= amount2) and (width1 <= splatterWidth <= width2)) or ((amount0 <= whitePixels <= amount1) and (width1 <= splatterWidth <= width2)) or ((0 <= whitePixels <= amount0) and (width1 <= splatterWidth <= width2)) or ((amount1 <= whitePixels <= amount2) and (width0 <= splatterWidth <= width1)) or ((amount1 <= whitePixels <= amount2) and (0 <= splatterWidth <= width0)):
                splatterSeverity = 2
                fgmask3 = cv2.putText(fgmask3, ("Splatter Severity: 2"), (int(120 * (scale_percent / 100)),int(180 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
                fgmask3 = cv2.putText(fgmask3, ("Splatter quantity: %d" %whitePixels), (int(120 * (scale_percent / 100)),int(220 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
                fgmask3 = cv2.putText(fgmask3, ("Splatter width: %d" %splatterWidth), (int(120 * (scale_percent / 100)),int(260 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
        elif ((amount2 <= whitePixels <= amount3) and (width2 <= splatterWidth <= width3)) or ((amount1 <= whitePixels <= amount2) and (width2 <= splatterWidth <= width3)) or ((amount0 <= whitePixels <= amount1) and (width2 <= splatterWidth <= width3)) or ((0 <= whitePixels <= amount0) and (width2 <= splatterWidth <= width3)) or ((amount2 <= whitePixels <= amount3) and (width1 <= splatterWidth <= width2)) or ((amount2 <= whitePixels <= amount3) and (width0 <= splatterWidth <= width1)) or ((amount2 <= whitePixels <= amount3) and (0 <= splatterWidth <= width0)):
                splatterSeverity = 3
                fgmask3 = cv2.putText(fgmask3, ("Splatter Severity: 3"), (int(120 * (scale_percent / 100)),int(180 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
                fgmask3 = cv2.putText(fgmask3, ("Splatter quantity: %d" %whitePixels), (int(120 * (scale_percent / 100)),int(220 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
                fgmask3 = cv2.putText(fgmask3, ("Splatter width: %d" %splatterWidth), (int(120 * (scale_percent / 100)),int(260 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
        else:
                splatterSeverity = 4
                fgmask3 = cv2.putText(fgmask3, ("Splatter Severity: 4"), (int(120 * (scale_percent / 100)),int(180 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
                fgmask3 = cv2.putText(fgmask3, ("Splatter quantity: %d" %whitePixels), (int(120 * (scale_percent / 100)),int(220 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
                fgmask3 = cv2.putText(fgmask3, ("Splatter width: %d" %splatterWidth), (int(120 * (scale_percent / 100)),int(260 * (scale_percent / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scale_percent/100), (255,0,0), 1, cv2.LINE_AA)
        
        #DRAW SEARCH AREA
        cv2.drawContours(fgmask3, [improvedContour], 0, 255, 3)
        
    
        #ADD DATA TO TABLE
        endTime = time.time()
        frameTime = endTime - startTime
        # resultsTable.loc[frameNumber,'frameNumber'] = frameNumber
        # resultsTable.loc[frameNumber,'splatterAmount(px)'] = whitePixels
        # resultsTable.loc[frameNumber,'splatterWidth(px)'] = splatterWidth
        # resultsTable.loc[frameNumber,'splatterSeverity'] = splatterSeverity
        # resultsTable.loc[frameNumber,'frameTime(s)'] = frameTime
        # print(resultsTable.loc[[frameNumber]])
    
    
    
    
        frameNumber +=1
        fgmaskCRGB = cv2.cvtColor(fgmaskC,cv2.COLOR_GRAY2RGB)
        #fgmaskMRGB = cv2.cvtColor(fgmaskM,cv2.COLOR_GRAY2RGB)
        fgmask3 = cv2.cvtColor(fgmask3, cv2.COLOR_GRAY2RGB)
        #fgmaskTRGB = cv2.cvtColor(fgmaskT, cv2.COLOR_GRAY2RGB)
        
        cv2.rectangle(fgmask3, (int(knifeFace1_xmin), int(knifeFace1_ymin)), (int(knifeFace1_xmax), int(knifeFace1_ymax)), (0,0,255), 1)
        fgmask3 = cv2.putText(fgmask3, ("Knife Face"), (int(knifeFace1_xmin) + 10,int(knifeFace1_ymin) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scale_percent/100), (0,0,255), 1, cv2.LINE_AA)
        cv2.rectangle(fgmask3, (int(knifeFace2_xmin), int(knifeFace2_ymin)), (int(knifeFace2_xmax), int(knifeFace2_ymax)), (0,0,255), 1)
        fgmask3 = cv2.putText(fgmask3, ("Knife Face"), (int(knifeFace2_xmin) + 10,int(knifeFace2_ymin) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scale_percent/100), (0,0,255), 1, cv2.LINE_AA)
        
        if undersideExists == 1:
            cv2.rectangle(fgmask3, (int(knifeUnderside1_xmin), int(knifeUnderside1_ymin)), (int(knifeUnderside1_xmax), int(knifeUnderside1_ymax)), (0,0,255), 1)
            fgmask3 = cv2.putText(fgmask3, ("Knife Face"), (int(knifeUnderside1_xmin) + 10,int(knifeUnderside1_ymin) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scale_percent/100), (0,0,255), 1, cv2.LINE_AA)        
            cv2.rectangle(fgmask3, (int(knifeUnderside2_xmin), int(knifeUnderside2_ymin)), (int(knifeUnderside2_xmax), int(knifeUnderside2_ymax)), (0,0,255), 1)
            fgmask3 = cv2.putText(fgmask3, ("Knife Face"), (int(knifeUnderside2_xmin) + 10,int(knifeUnderside2_ymin) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scale_percent/100), (0,0,255), 1, cv2.LINE_AA)    
        
        
        fgmaskCRGB[np.where(fgmaskCRGB[:,:,0] == 255)] = (255,0,255)
        fgmaskCRGB[np.where(fgmaskCRGB[:,:,0] == 155)] = (0,255,255)
        fgmaskCRGB[np.where(fgmaskCRGB[:,:,0] == 200)] = (0,255,0)
        #fgmaskMRGB[np.where(fgmaskMRGB[:,:,0] == 255)] = (255,0,255)
        #fgmaskMRGB[np.where(fgmaskMRGB[:,:,0] == 155)] = (0,255,255)
        #fgmaskMRGB[np.where(fgmaskMRGB[:,:,0] == 200)] = (0,255,0)
        fgmask3[np.where(fgmask3[:,:,0] == 255)] = (0, 0, 255)
        #print(fgmaskRGB.shape, fgmaskTRGB.shape)
#show fgmaskCRGB below maAirPosKnife and fgmaskMRGB above maAirPosKnife
#overlay fgmask3 on top for text and bboxes
        #print(fgmaskRGB.shape)
        src1 = frame
        src2 = fgmaskCRGB
        #src2 = fgmaskCRGB[:int(maAirKnifePos),:] #top CNT (not used)
        #src3 = fgmaskCRGB[int(maAirKnifePos):,:]
        
        #src4 = fgmaskMRGB[:int(maAirKnifePos),:]
        #src5 = fgmaskMRGB[int(maAirKnifePos):,:] #bottom MOG2 (not used)
        
        #src6 = np.vstack((src4, src3))
        
        src7 = fgmask3
        
        dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
        dst = cv2.addWeighted(dst, 0.5, src7, 0.5, 0)
        dstResized = cv2.resize(dst, (960, 540))                # Resize image
        cv2.imshow('sFrame',dstResized)
        #cv2.imshow('sFrame',fgmask)
        #cv2.imshow('sFrame',fgmask2)
        #cv2.imshow('sFrame',fgmaskRGB)
        endTime = time.time()
        frameTime = endTime - startTime
        #print("frame time: ", frameTime)
        print("FPS: ", 1/frameTime)
        # resultsTable.loc[frameNumber,'frameNumber'] = frameNumber
        # resultsTable.loc[frameNumber,'splatterAmount(px)'] = whitePixels
        # resultsTable.loc[frameNumber,'splatterWidth(px)'] = splatterWidth
        # resultsTable.loc[frameNumber,'splatterSeverity'] = splatterSeverity
        # resultsTable.loc[frameNumber,'frameTime(s)'] = frameTime
        
        k = cv2.waitKey(1) & 0xff
        if k == 27:
           break
       
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/bottomOld/Frame%d.jpg' %frameNumber, dst)
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/bottomNew/Frame%d.jpg' %frameNumber, dst)
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/topNew/Frame%d.jpg' %frameNumber, dst)
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/originalFrames/%dFrame%d.jpg' %(vidCounter, frameNumber), frame)
    
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideos/1/Frame%d.jpg' %frameNumber, dst)
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideos/2/Frame%d.jpg' %frameNumber, dst) 
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideos/3_stable/Frame%d.jpg' %frameNumber, dst) 
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideos/4/Frame%d.jpg' %frameNumber, dst) 
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideos/5/Frame%d.jpg' %frameNumber, dst) 
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideos/6/Frame%d.jpg' %frameNumber, dst) 
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideos/5_bbox/Frame%d.jpg' %frameNumber, dst)  
      
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideosImproved/1/Frame%d.jpg' %frameNumber, dst)
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideosImproved/2/Frame%d.jpg' %frameNumber, dst) 
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideosImproved/3/Frame%d.jpg' %frameNumber, dst) 
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideosImproved/4/Frame%d.jpg' %frameNumber, dst) 
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideosImproved/5/Frame%d.jpg' %frameNumber, dst) 
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatterVideosImproved/6/Frame%d.jpg' %frameNumber, dst)    
        #cv2.imwrite(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/splatterWithYOLO/splatter1_tst/Frame%d.jpg' %frameNumber, dst) 

  
 
    #resultsTable.to_csv(r'C:/Users/Callum/Anaconda3/envs/splatterValidation/csv/splatterResults.csv', index=False)
    cap.release()
    cv2.destroyAllWindows()
 #   infTimes = np.array(resultsTable['frameTime(s)'])  
    
    avgInfTime = np.mean(infTimes)
    avgFPS = 1/avgInfTime
    print("%d FRAMES PER SECOND" %avgFPS)
    

#print("TESTING")
overallStartTime = time.time()
# lp = LineProfiler()
# lp_wrapper = lp(main)
# lp_wrapper()
# lp.print_stats()    
overallTime = time.time() - overallStartTime
print("OVERALL TIME TAKEN: ", overallTime)
if __name__ == "__main__":
    main()