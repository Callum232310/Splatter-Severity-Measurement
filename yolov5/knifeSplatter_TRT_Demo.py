# -*- coding: utf-8 -*-
"""
Script Name: knifeSplatter_Final_Deployment.py

Description: This script is the YOLOv5 + background subtraction splatter model optimised with TensorRT. This script will be utilised on the inference device by the API when the client-side streams video across the network.

Original Author: Callum O'Donovan

Original Creation Date: December 6th 2023

Email: callumodonovan2310@gmail.com
  
Disclaimer: This script is part of a project focusing on practical application in engineering.
            For full code quality and contribution guidelines, see the README file. 
            
"""

import numpy as np
import cv2
import time
import tensorrt as trt
import pycuda.driver as cuda
import tensorflow as tf
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

cuda.init()






def profile_section(start_time, section_name=""): # This function is used for timing each step
    end_time = time.time()
    print(f"{section_name} took {(end_time - start_time) * 1000:.2f} ms")



def tf_nms(detections_filtered, iou_threshold=0.5, score_threshold=0.0000001, max_detections=200): # Non-maximum suppression using tensorflow (uses GPU)

    #start = time.time()
    
    # Convert numpy input to tensors
    boxes = tf.constant(detections_filtered[:, [1, 0, 3, 2]], dtype=tf.float32)  # Adjusted slicing for y_min, x_min, y_max, x_max
    scores = tf.constant(detections_filtered[:, 4], dtype=tf.float32)  # Confidence scores
    
    # Apply NMS
    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size=max_detections, iou_threshold=iou_threshold, score_threshold=score_threshold
    )
    
    # Gather selected detections
    selected_boxes = tf.gather(boxes, selected_indices).numpy()
    selected_scores = tf.gather(scores, selected_indices).numpy()
    
    # Convert back to numpy array
    selected_detections = np.array([
        [
            selected_boxes[idx][1],  # x_min
            selected_boxes[idx][0],  # y_min
            selected_boxes[idx][3],  # x_max
            selected_boxes[idx][2],  # y_max
            selected_scores[idx],    # confidence
            detections_filtered[selected_indices.numpy()[idx], 5],  # class1_prob
            detections_filtered[selected_indices.numpy()[idx], 6]   # class2_prob
        ] for idx in range(len(selected_indices))
    ])

    #profile_section(start, "TF NMS")
    
    return selected_detections


def load_engine(engine_path):
    
    #start = time.time()
    #print("TESTING LOAD ENGINE")
    
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        
        #profile_section(start, "LOAD ENGINE")
        
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine): # Allocate host and device buffers  
 
    #start = time.time()
    
    #print("TESTING BUFFERS")
    
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings
        bindings.append(int(device_mem))
        
        # Append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
            
        #profile_section(start, "ALLOCATE BUFFERS")
        
    return inputs, outputs, bindings, stream


def preprocess(frame, scaleParameter):
    
    #start = time.time()
    
    #print("TESTING PREPROCESS")
    
    startTime = time.time()
    
    frame_height, frame_width = frame.shape[:2]
    
    #center = (frame_width // 2, frame_height // 2)
    
    # Rotate frame if needed
    #rotation_matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)
    #rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame_width, frame_height))
    
    # Resize frame if needed
    # width = int(rotated_frame.shape[1] * scaleParameter / 100)
    # height = int(rotated_frame.shape[0] * scaleParameter / 100)
    
    width = int(frame.shape[1] * scaleParameter / 100)
    height = int(frame.shape[0] * scaleParameter / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    #profile_section(start, "PREPROCESS")
    
    return resized_frame, startTime


def infer(context, bindings, inputs, outputs, stream):
    
    #start = time.time()
    
    #print("TESTING INFER")
    
    # Transfer input data to the GPU
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    
    # Run inference
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    
    # Transfer predictions back from the GPU
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    
    # Synchronize the stream
    stream.synchronize()
    
    # Return only the host outputs
    all_outputs = [out['host'] for out in outputs]
    
    for output in all_outputs:
        print(output.shape)

    #profile_section(start, "INFER")
    
    return outputs[0]['host']

def postprocess(inferenceResults, airKnifePosCount, scaleParameter, airKnifePoss, scalingFactors, scalingFactorCount, fgbgC, frame, eKernelParameter, cThreshParameter, startTime):
    
    #start = time.time()
    
    
    detections = inferenceResults.reshape(-1, 7)  # Reshape from 176400 to 29400,6
    
    
    class_probability_threshold = 0.9
    mask = np.maximum(detections[:, 5], detections[:, 6]) > class_probability_threshold
    filtered_detections = detections[mask]
    
    # Sort by confidence
    indices_sorted = np.argsort(-filtered_detections[:, 4])  # Negating for descending order
    detections_filtered = filtered_detections[indices_sorted]
    
    # Perform tensorflow-gpu NMS
    detections_nms = tf_nms(detections_filtered, iou_threshold=0.5, score_threshold=0.000001)

    # Initialize variables to None to avoid errors
    knifeFace1 = knifeFace2 = knifeUnderside1 = knifeUnderside2 = None
  
    # Check for the existence of the underside
    undersideExists = np.any(detections_nms[:, 6] > detections_nms[:, 5])
    
    # Check if underside of knives are visible
    if undersideExists:
        relevant_detections = detections_nms
    else:
        # If underside not present, ignore underside detections because they will definitely be wrong
        relevant_detections = detections_nms[detections_nms[:, 5] > detections_nms[:, 6]]
 
      # Relevant detections to calculate the air knives' positions
    if len(relevant_detections) > 0:
        airKnifePos = np.max(relevant_detections[:, 3])  # y_max
        airKnifePoss.append(airKnifePos)
        airKnifePosCount += 1
        if 0 < airKnifePosCount < 10:
            maAirKnifePos = airKnifePoss[0]
        elif airKnifePosCount % 10 == 0:
            maAirKnifePos = sum(airKnifePoss[-10:]) / 10
        else:
            maAirKnifePos = maAirKnifePos
            
    # Select best 2 detections of each class
    knife_faces = relevant_detections[relevant_detections[:, 5] > relevant_detections[:, 6]][:2]
    undersides = relevant_detections[relevant_detections[:, 6] > relevant_detections[:, 5]][:2]

    # Check if knifeface detections exist
    if knife_faces is not None and len(knife_faces) > 0:
        knifeFace1 = knife_faces[0]  # Isolate desired detections
        knifeFace2 = knife_faces[1] if len(knife_faces) > 1 else None
        
    # Check if underside detections exist
    if undersideExists and undersides is not None and len(undersides) > 0:
        knifeUnderside1 = undersides[0]  
        knifeUnderside2 = undersides[1] if len(undersides) > 1 else None

    # Extract co-ordinates from chosen detections
    if knifeFace1 is not None:
        knifeFace1_xmin = knifeFace1[0]  
        knifeFace1_ymin = knifeFace1[1]
        knifeFace1_xmax = knifeFace1[2]
        knifeFace1_ymax = knifeFace1[3]  
        
    if knifeFace2 is not None:
        knifeFace2_xmin = knifeFace2[0]  
        knifeFace2_ymin = knifeFace2[1]
        knifeFace2_xmax = knifeFace2[2]
        knifeFace2_ymax = knifeFace2[3] 
        

     # Calculate combined area of both knifeFace bounding boxes in pixels
    if knifeFace1 is not None and knifeFace2 is not None:
        bboxCombArea = ((knifeFace1[2] - knifeFace1[0]) * (knifeFace1[3] - knifeFace1[1])) + \
                       ((knifeFace2[2] - knifeFace2[0]) * (knifeFace2[3] - knifeFace2[1]))
     
        # Calculate scaling factor from bboxCombArea
        scalingFactor = round(bboxCombArea / (80000 * (scaleParameter / 100) * (scaleParameter / 100)), 1)
        scalingFactors.append(scalingFactor)
        scalingFactorCount += 1
        if 0 < scalingFactorCount < 25000:
            maScalingFactor = scalingFactors[0]
        elif scalingFactorCount % 25000 == 0:
            maScalingFactor = sum(scalingFactors[-25000:])/25000
        else:
            maScalingFactor = maScalingFactor
            
    # Apply BGS
    fgmaskC = fgbgC.apply(frame)
    
    kernel = np.ones((int(eKernelParameter),int(eKernelParameter)), np.uint8)
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
    
    # Find outer and inner edges of bounding boxes to create refined splatter region boundary
    if knifeFace1 is not None and knifeFace2 is not None:
        leftMostSide = np.min([knifeFace1_xmin,knifeFace1_xmax, knifeFace2_xmin, knifeFace2_xmax])
        rightMostSide = np.max([knifeFace1_xmin,knifeFace1_xmax, knifeFace2_xmin, knifeFace2_xmax])
        
    else:
        print("NO DETECTIONS")
            
    # Define splatter region points
    if undersideExists == 0:         
        i1 = (0, int(maAirKnifePos+(28*maScalingFactor)))
        i2 = (leftMostSide+25, int(maAirKnifePos+(25*maScalingFactor)))
        i3 = (leftMostSide+25, int(maAirKnifePos))
        i4 = (rightMostSide-25, int(maAirKnifePos))
        i5 = (rightMostSide-25, int(maAirKnifePos+(25*maScalingFactor)))
        i6 = (frame.shape[1], int(maAirKnifePos+(25*maScalingFactor)))
        i7 = (frame.shape[1], frame.shape[0])
        i8 = (0, frame.shape[0])
        
        
        improvedContour = np.array([i1,i2,i3,i4,i5,i6,i7,i8], dtype=np.int32)
        
        

    else:
        i1 = (0, int(maAirKnifePos+(25*maScalingFactor)))
        i2 = (frame.shape[1], int(maAirKnifePos+(25*maScalingFactor)))
        i3 = (frame.shape[1], frame.shape[0])
        i4 = (0, frame.shape[0])     
        improvedContour = np.array([i1,i2,i3,i4])
    
    improvedContourFlipped = [[y, x] for x, y in improvedContour]
    
    
    # Define splatterArea (splatter region)
    cv2.drawContours(fgmask2, [improvedContour], -1, 255, -3)
    splatterArea = np.where(fgmask2 == 255)

    # Count white pixels in splatterArea
    whitePixels = np.sum(fgmaskC[splatterArea] == 255)
    splatterWidth = 0
    splatterSeverity = 0
    
   
    # Check that it is not one of the initial frames (CNT takes 10-20 frames to initialise where it segments the whole image)
    if whitePixels < (1500000 * (scaleParameter / 100)):
        
        # Calculate splatter width
        maxX = 0
        maxY = 0
        minX = 0
        minY = 0
        leftSplatter = 0
        rightSplatter = 0
        
        
        fgmaskC = cv2.bitwise_and(fgmaskC, fgmask2)
        contours, hierarchy = cv2.findContours(fgmaskC,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
             
        # Find leftmost and rightmost points of all contours to calculate splatter width from leftmost point and rightmost point of segmentation mask
        for c in contours:

            if len(c[:,0,0]) > cThreshParameter*(scaleParameter/100):
                
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
        

    
    # Set splatter severity level boundaries
    amount0 = 7500 * maScalingFactor * (scaleParameter / 100) * (scaleParameter / 100)
    amount1 = 15000 * maScalingFactor * (scaleParameter / 100) * (scaleParameter / 100)
    amount2 = 22500 * maScalingFactor * (scaleParameter / 100) * (scaleParameter / 100)
    amount3 = 30000 * maScalingFactor * (scaleParameter / 100) * (scaleParameter / 100)
    width0 = 200 * maScalingFactor * (scaleParameter / 100) * (scaleParameter / 100)
    width1 = 400 * maScalingFactor * (scaleParameter / 100) * (scaleParameter / 100)
    width2 = 600 * maScalingFactor * (scaleParameter / 100) * (scaleParameter / 100)
    width3 = 800 * maScalingFactor * (scaleParameter / 100) * (scaleParameter / 100)
    if width3 > 1920 * (scaleParameter / 100):
        width3 = 1920 * (scaleParameter / 100)
    
    fgmaskC = cv2.putText(fgmaskC, ("Scaling Factor: %.1f" %scalingFactor), (int(1400 * (scaleParameter / 100)), int(180 * (scaleParameter / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scaleParameter/100), (255,0,0), 1, cv2.LINE_AA)
    fgmaskC = cv2.putText(fgmaskC, ("MA Scaling Factor: %.1f" %maScalingFactor), (int(1400 * (scaleParameter / 100)), int(220 * (scaleParameter / 100))), cv2.FONT_HERSHEY_SIMPLEX, (scaleParameter/100), (255,0,0), 1, cv2.LINE_AA)  
    
    
    # Final severity levels. putText is used for when there is visual output which is not suitable for efficient deployment
    if ((0 <= whitePixels <= amount0) and (0 <= splatterWidth <= width0)):
            splatterSeverity = 0
            fgmaskC = cv2.putText(fgmaskC, ("Splatter Severity: 0"), (120,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            fgmaskC = cv2.putText(fgmaskC, ("Splatter quantity: %d" %whitePixels), (120,220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            fgmaskC = cv2.putText(fgmaskC, ("Splatter width: %d" %splatterWidth), (120,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
    elif ((amount0 <= whitePixels <= amount1) and (width0 <= splatterWidth <= width1)) or ((amount0 <= whitePixels <= amount1) and (0 <= splatterWidth <= width0)) or ((0 <= whitePixels <= amount0) and (width0 <= splatterWidth <= width1)):
            splatterSeverity = 1
            fgmaskC = cv2.putText(fgmaskC, ("Splatter Severity: 1"), (120,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            fgmaskC = cv2.putText(fgmaskC, ("Splatter quantity: %d" %whitePixels), (120,220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            fgmaskC = cv2.putText(fgmaskC, ("Splatter width: %d" %splatterWidth), (120,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA) 
    elif ((amount1 <= whitePixels <= amount2) and (width1 <= splatterWidth <= width2)) or ((amount0 <= whitePixels <= amount1) and (width1 <= splatterWidth <= width2)) or ((0 <= whitePixels <= amount0) and (width1 <= splatterWidth <= width2)) or ((amount1 <= whitePixels <= amount2) and (width0 <= splatterWidth <= width1)) or ((amount1 <= whitePixels <= amount2) and (0 <= splatterWidth <= width0)):
            splatterSeverity = 2
            fgmaskC = cv2.putText(fgmaskC, ("Splatter Severity: 2"), (120,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            fgmaskC = cv2.putText(fgmaskC, ("Splatter quantity: %d" %whitePixels), (120,220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            fgmaskC = cv2.putText(fgmaskC, ("Splatter width: %d" %splatterWidth), (120,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
    elif ((amount2 <= whitePixels <= amount3) and (width2 <= splatterWidth <= width3)) or ((amount1 <= whitePixels <= amount2) and (width2 <= splatterWidth <= width3)) or ((amount0 <= whitePixels <= amount1) and (width2 <= splatterWidth <= width3)) or ((0 <= whitePixels <= amount0) and (width2 <= splatterWidth <= width3)) or ((amount2 <= whitePixels <= amount3) and (width1 <= splatterWidth <= width2)) or ((amount2 <= whitePixels <= amount3) and (width0 <= splatterWidth <= width1)) or ((amount2 <= whitePixels <= amount3) and (0 <= splatterWidth <= width0)):
            splatterSeverity = 3
            fgmaskC = cv2.putText(fgmaskC, ("Splatter Severity: 3"), (120,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            fgmaskC = cv2.putText(fgmaskC, ("Splatter quantity: %d" %whitePixels), (120,220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            fgmaskC = cv2.putText(fgmaskC, ("Splatter width: %d" %splatterWidth), (120,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
    else:
            splatterSeverity = 4
            fgmaskC = cv2.putText(fgmaskC, ("Splatter Severity: 4"), (120,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            fgmaskC = cv2.putText(fgmaskC, ("Splatter quantity: %d" %whitePixels), (120,220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            fgmaskC = cv2.putText(fgmaskC, ("Splatter width: %d" %splatterWidth), (120,260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            

    # Draw search area
    cv2.drawContours(fgmaskC, [improvedContour], 0, 155, 3)
    
    # Convert mask to RGB
    fgmaskRGB = cv2.cvtColor(fgmaskC,cv2.COLOR_GRAY2RGB)
    
    # Draw bounding boxes on visual output
    cv2.rectangle(fgmaskRGB, (int(knifeFace1_xmin), int(knifeFace1_ymin)), (int(knifeFace1_xmax), int(knifeFace1_ymax)), (0,0,255), 1)
    fgmask = cv2.putText(fgmaskRGB, ("Knife Face"), (int(knifeFace1_xmin) + 10,int(knifeFace1_ymin) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scaleParameter/100), (0,0,255), 1, cv2.LINE_AA)
    cv2.rectangle(fgmaskRGB, (int(knifeFace2_xmin), int(knifeFace2_ymin)), (int(knifeFace2_xmax), int(knifeFace2_ymax)), (0,0,255), 1)
    fgmask = cv2.putText(fgmaskRGB, ("Knife Face"), (int(knifeFace2_xmin) + 10,int(knifeFace2_ymin) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scaleParameter/100), (0,0,255), 1, cv2.LINE_AA)
    
    
    # Draw underside bounding boxes on visual output
    if undersideExists == 1:
        cv2.rectangle(fgmaskRGB, (int(knifeUnderside1['x_min']), int(knifeUnderside1['y_min'])), (int(knifeUnderside1['x_max']), int(knifeUnderside1['y_max'])), (0,0,255), 1)
        fgmask = cv2.putText(fgmaskRGB, ("Knife Underside"), (int(knifeUnderside1['x_min']) + 10,int(knifeUnderside1['y_min']) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scaleParameter/100), (0,0,255), 1, cv2.LINE_AA)        
        cv2.rectangle(fgmaskRGB, (int(knifeUnderside2['x_min']), int(knifeUnderside2['y_min'])), (int(knifeUnderside2['x_max']), int(knifeUnderside2['y_max'])), (0,0,255), 1)
        fgmask = cv2.putText(fgmaskRGB, ("Knife Underside"), (int(knifeUnderside2['x_min']) + 10,int(knifeUnderside2['y_min']) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scaleParameter/100), (0,0,255), 1, cv2.LINE_AA)    
    
    # Colour the mask for clarity and superimpose it on the original frame
    fgmaskRGB[np.where(fgmaskRGB[:,:,0] == 255)] = (255,0,255)
    fgmaskRGB[np.where(fgmaskRGB[:,:,0] == 155)] = (0,0,255)
    fgmaskRGB[np.where(fgmaskRGB[:,:,0] == 200)] = (0,255,0)
 
    src1 = fgmaskRGB
    src2 = frame
    dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

    endTime = time.time()
    frameTime = endTime - startTime
 
    
    #avgInfTime = np.mean(infTimes)
    
    #avgFPS = 1/avgInfTime
    
    #print("%.2f seconds per frame" %avgInfTime)
    
    #print("%.2f frames per second" %avgFPS)
    
    #cv2.imshow('Result', dst)
    #cv2.waitKey(1)
    
    print("splatterSeverity: %d, frameTime: %.2f" %(splatterSeverity, frameTime))

    #profile_section(start, "POSTPROCESS END")
    
    return splatterSeverity, dst


def process_frame(frame, model, fgbgC, context, bindings, inputs, outputs, stream,
                  scaleParameter, eKernelParameter, cThreshParameter, results, airKnifePosCount,
                  airKnifePoss, scalingFactors, scalingFactorCount):
    
    #start = time.time()
    
    resized_frame, startTime = preprocess(frame, scaleParameter)
    
    np.copyto(inputs[0]['host'], np.ravel(resized_frame))
    
    inferenceResults = infer(context, bindings, inputs, outputs, stream)
    
    splatterSeverity, dst = postprocess(inferenceResults, airKnifePosCount, scaleParameter, airKnifePoss, 
                                        scalingFactors, scalingFactorCount, fgbgC, frame, eKernelParameter,
                                        cThreshParameter, startTime)
    
    
    #profile_section(start, "INFERENCE END")
    
    return splatterSeverity, dst 


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a video file with a TensorRT optimized model.")
    parser.add_argument('--model', type=str, required=True, help='Path to the TensorRT engine file.')
    parser.add_argument('--source', type=str, required=True, help='Path to the video file to process.')
    args = parser.parse_args()
    return args


def main():
    # Inputs
    args = parse_args()
    video_path = args.source
    engine_path = args.model

    # Parameters
    scaleParameter = 100
    eKernelParameter = 2
    cThreshParameter = 100

    # Load the TensorRT engine
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Initialize the background subtractor
    fgbgC = cv2.createBackgroundSubtractorMOG2()

    # Video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    airKnifePosCount = 0
    airKnifePoss = []
    scalingFactors = []
    scalingFactorCount = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # No more frames or error

            # Preprocess the frame
            resized_frame, startTime = preprocess(frame, scaleParameter)
            
            # Copy preprocessed frame to input buffer
            np.copyto(inputs[0]['host'], np.ravel(resized_frame))

            # Inference
            inferenceResults = infer(context, bindings, inputs, outputs, stream, resized_frame.shape)

            # Postprocess and get results
            splatterSeverity, dst = postprocess(inferenceResults, airKnifePosCount, scaleParameter, airKnifePoss, scalingFactors, scalingFactorCount, fgbgC, frame, eKernelParameter, cThreshParameter, startTime)

            # Show result
            cv2.imshow('Result', dst)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
