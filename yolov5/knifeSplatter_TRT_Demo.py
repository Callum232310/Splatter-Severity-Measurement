# -*- coding: utf-8 -*-
"""
Script Name: knifeSplatter_TRT_Demo.py

Description: This script is for demonstration of the YOLOv5 + background subtraction splatter model optimised with TensorRT.

Original Author: Callum O'Donovan

Original Creation Date: March 28th 2024

Email: callumodonovan2310@gmail.com
  
Disclaimer: This script is part of a project focusing on practical application in engineering.
            For full code quality and contribution guidelines, see the README file. 
            
"""

import numpy as np
import cv2
import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import tensorflow as tf
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

cuda.init()

def profile_section(start_time, section_name=""): # Time each step
    end_time = time.time()
    print(f"{section_name} took {(end_time - start_time) * 1000:.2f} ms")


def convert_to_corners(detections): # Converts detections from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
    converted = []
    for det in detections:
        x_center, y_center, width, height = det[:4]
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)
        converted.append([x_min, y_min, x_max, y_max] + list(det[4:]))
    return np.array(converted)
    
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

def load_engine(engine_path): #Load the TensorRT engine file and deserialise it

    runtime = trt.Runtime(TRT_LOGGER)
    
    # Open the file and read it
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
        
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    return engine


def allocate_buffers(engine): # Allocate host and device buffers  
 
    #start = time.time()
       
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


def infer(context, bindings, inputs, outputs, stream): #Perform YOLOv5 inference
    
    #start = time.time()
    
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
    
    #for output in all_outputs:
        #print(output.shape)

    #profile_section(start, "INFER")
    
    return outputs[0]['host']
   
def preprocess(frame, target_size=(1088, 1920), scaleParameter=100): #Preprocess data ready for inference

    startTime = time.time()
    
    # Resize frame to target width, maintaining aspect ratio
    h, w = frame.shape[:2]
    scaling_factor = target_size[1] / w
    new_size = (int(w * scaling_factor * scaleParameter / 100), int(h * scaling_factor * scaleParameter / 100))
    frame_resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
    
    # Pad the resized frame to the target height if necessary
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    frame_padded = cv2.copyMakeBorder(frame_resized, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)

    # Normalise pixel values
    frame_normalized = frame_rgb / 255.0

    # HWC to CHW format for TensorRT model
    frame_chw = np.transpose(frame_normalized, (2, 0, 1))

    # Add batch dimension
    frame_batch = frame_chw[np.newaxis, ...].astype(np.float32)

    return frame_batch, startTime

def postprocess(inferenceResults, airKnifePosCount, scaleParameter, airKnifePoss, scalingFactors, scalingFactorCount, fgbgC, frame, eKernelParameter, cThreshParameter, startTime, maAirKnifePos, maScalingFactor): #Postprocess YOLOv5 inference and apply BGS
    
    #start = time.time()
    
    #Initialise some variables
    knifeFace1 = knifeFace2 = knifeUnderside1 = knifeUnderside2 = None
    knifeFace1_xmin = knifeFace1_ymin = knifeFace1_xmax = knifeFace1_ymax = None
    knifeFace2_xmin = knifeFace2_ymin = knifeFace2_xmax = knifeFace2_ymax = None
    knifeUnderside1_xmin = knifeUnderside1_ymin = knifeUnderside1_xmax = knifeUnderside1_ymax = None
    knifeUnderside2_xmin = knifeUnderside2_ymin = knifeUnderside2_xmax = knifeUnderside2_ymax = None

    detections = inferenceResults.reshape(-1, 7)  # Reshape so each row is one detection
    
    #Filter out detections by confidence first
    confidence_threshold = 0.01
    confidence_mask = detections[:, 4] > confidence_threshold
    filtered_detections = detections[confidence_mask]
    filtered_detections = convert_to_corners(filtered_detections)
    
    # Separate detections by class probability
    class1_mask = filtered_detections[:, 5] > filtered_detections[:, 6]
    class2_mask = ~class1_mask

    class1_detections = filtered_detections[class1_mask]
    class2_detections = filtered_detections[class2_mask]
    
    # Apply tf_nms separately for each class
    class1_nms_detections = tf_nms(class1_detections, iou_threshold=0.5, score_threshold=0.001, max_detections=200)
    class2_nms_detections = tf_nms(class2_detections, iou_threshold=0.5, score_threshold=0.3, max_detections=200)

    # Combining the NMS results of both classes
    if len(class2_nms_detections) > 0:
        detections_nms = np.vstack([class1_nms_detections, class2_nms_detections])
    else:
        detections_nms = class1_nms_detections
    
    # Check for the existence of the underside
    undersideExists = np.any(detections_nms[:, 6] > detections_nms[:, 5])
    
    # Check if underside of knives are visible
    if undersideExists:
        relevant_detections = detections_nms
    else:
        relevant_detections = detections_nms[detections_nms[:, 5] > detections_nms[:, 6]]
      
    # Select best 2 detections of each class
    knife_faces = relevant_detections[relevant_detections[:, 5] > relevant_detections[:, 6]][:2]
    undersides = relevant_detections[relevant_detections[:, 5] < relevant_detections[:, 6]][:2]
    
    # Relevant detections to calculate the air knives' positions
    if len(relevant_detections) > 0:
        airKnifePos = np.max(relevant_detections[:, 3])  # y_max
        airKnifePoss.append(airKnifePos)
        airKnifePosCount += 1
        
        # Calculate moving average of the last 10 airKnifePos values
        if 0 < airKnifePosCount < 10:
            maAirKnifePos = airKnifePoss[0]
        
        elif airKnifePosCount % 10 == 0:
            maAirKnifePos = sum(airKnifePoss[-10:]) / 10            
        else:
            maAirKnifePos = maAirKnifePos


    # Isolate individual detections
    if knife_faces is not None and len(knife_faces) > 0:
        knifeFace1 = knife_faces[0]  
        knifeFace2 = knife_faces[1] if len(knife_faces) > 1 else None
        
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
    
    # Apply erosion
    kernel = np.ones((int(eKernelParameter),int(eKernelParameter)), np.uint8)
    fgmaskC = cv2.erode(fgmaskC, kernel) 
    
    
    fgmask2 = fgmaskC.copy()
    fgmask2 = np.zeros_like(fgmask2)
    
    Q= 0
    
    
    # Create a list of the variables
    values = [knifeFace1_xmin, knifeFace1_xmax, knifeFace2_xmin, knifeFace2_xmax]

    # Filter out None values
    filtered_values = [value for value in values if value is not None]

    # Calcualte min and max x values
    if filtered_values:
        leftMostSide = np.min(filtered_values)
        rightMostSide = np.max(filtered_values)
    else:
        leftMostSide = 0
        rightMostSide = frame.shape[1]
   
    # Define splatter region points
    if undersideExists == 0:         
        i1 = (0, int(maAirKnifePos+(28*maScalingFactor))+40)
        i2 = (leftMostSide+45, int(maAirKnifePos+(28*maScalingFactor))+40)
        i3 = (leftMostSide+45, int(maAirKnifePos)+40)
        i4 = (rightMostSide-45, int(maAirKnifePos)+40)
        i5 = (rightMostSide-45, int(maAirKnifePos+(28*maScalingFactor))+40)
        i6 = (frame.shape[1], int(maAirKnifePos+(28*maScalingFactor))+40)
        i7 = (frame.shape[1], frame.shape[0])
        i8 = (0, frame.shape[0])
        
        
        improvedContour = np.array([i1,i2,i3,i4,i5,i6,i7,i8], dtype=np.int32)

    else:
        i1 = (0, int(maAirKnifePos+(23*maScalingFactor))+40)
        i2 = (frame.shape[1], int(maAirKnifePos+(23*maScalingFactor))+40)
        i3 = (frame.shape[1], frame.shape[0])
        i4 = (0, frame.shape[0])     
        improvedContour = np.array([i1,i2,i3,i4])
    
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
                
            if len(c[:,0,0]) > 100*(scaleParameter/100):
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
                     
                
        # Draw lines showing splatter width. Used for demonstratin purposes but not suitable for efficient deployment
        if undersideExists == 0:
            
            if (leftMostSide + 45) < minX < (rightMostSide -45):
                cv2.line(fgmaskC, (int(minX), int(maAirKnifePos)+40), (minX, frame.shape[0]), (155), 2)
            else:
                cv2.line(fgmaskC, (int(minX), int(maAirKnifePos+(28*maScalingFactor))+40), (minX, frame.shape[0]), (155), 2)
                
            if (leftMostSide + 45) < maxX < (rightMostSide -45):
                cv2.line(fgmaskC, (int(maxX), int(maAirKnifePos)+40), (maxX, frame.shape[0]), (155), 2)   
            else:
                cv2.line(fgmaskC, (int(maxX), int(maAirKnifePos+(28*maScalingFactor))+40), (maxX, frame.shape[0]), (155), 2) 
                
        else:
            cv2.line(fgmaskC, (int(minX), int(maAirKnifePos+(23*maScalingFactor))+40), (minX, frame.shape[0]), (155), 2)
            cv2.line(fgmaskC, (int(maxX), int(maAirKnifePos+(23*maScalingFactor))+40), (maxX, frame.shape[0]), (155), 2)              
                         
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

    # Draw bounding boxes on visual output. As mentioned, this is for demonstration and not deployment
    if knifeFace1 is not None:
        cv2.rectangle(fgmaskRGB, (int(knifeFace1_xmin), int(knifeFace1_ymin)), (int(knifeFace1_xmax), int(knifeFace1_ymax)), (0,0,255), 1)
        fgmask = cv2.putText(fgmaskRGB, ("Knife Face"), (int(knifeFace1_xmin) + 10,int(knifeFace1_ymin) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scaleParameter/100), (0,0,255), 1, cv2.LINE_AA)
        
    if knifeFace2 is not None:
        cv2.rectangle(fgmaskRGB, (int(knifeFace2_xmin), int(knifeFace2_ymin)), (int(knifeFace2_xmax), int(knifeFace2_ymax)), (0,0,255), 1)
        fgmask = cv2.putText(fgmaskRGB, ("Knife Face"), (int(knifeFace2_xmin) + 10,int(knifeFace2_ymin) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scaleParameter/100), (0,0,255), 1, cv2.LINE_AA)
        
    if knifeUnderside1 is not None:
         cv2.rectangle(fgmaskRGB, (int(knifeUnderside1[0]), int(knifeUnderside1[1])), (int(knifeUnderside1[2]), int(knifeUnderside1[3])), (0,0,255), 1)
         fgmask = cv2.putText(fgmaskRGB, ("Knife Underside"), (int(knifeUnderside1[0]) + 10,int(knifeUnderside1[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scaleParameter/100), (0,0,255), 1, cv2.LINE_AA)

    if knifeUnderside2 is not None:
         cv2.rectangle(fgmaskRGB, (int(knifeUnderside2[0]), int(knifeUnderside2[1])), (int(knifeUnderside2[2]), int(knifeUnderside2[3])), (0,0,255), 1)
         fgmask = cv2.putText(fgmaskRGB, ("Knife Underside"), (int(knifeUnderside2[0]) + 10,int(knifeUnderside2[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scaleParameter/100), (0,0,255), 1, cv2.LINE_AA)

    # Colour the mask for clarity and superimpose it on the original frame
    fgmaskRGB[np.where(fgmaskRGB[:,:,0] == 255)] = (255,0,255)
    fgmaskRGB[np.where(fgmaskRGB[:,:,0] == 155)] = (0,0,255)
    fgmaskRGB[np.where(fgmaskRGB[:,:,0] == 200)] = (0,255,0)
 
    src1 = fgmaskRGB
    src2 = frame
    dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

    endTime = time.time()
    frameTime = endTime - startTime
     
    print("splatterSeverity: %d, frameTime: %.2f" %(splatterSeverity, frameTime))

    #profile_section(start, "POSTPROCESS END")
    
    return splatterSeverity, dst, airKnifePosCount, airKnifePoss, maAirKnifePos, scalingFactorCount, scalingFactors, maScalingFactor

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a video file with a TensorRT optimised model.")
    parser.add_argument('--model', type=str, required=True, help='Path to the TensorRT engine file.')
    parser.add_argument('--source', type=str, required=True, help='Path to the video file to process.')
    args = parser.parse_args()
    return args


def main():
    
    # Inputs
    args = parse_args()
    video_path = args.source
    engine_path = args.model
    
    # Initialise some variables
    airKnifePosCount = 0
    airKnifePoss = []
    maAirKnifePos = 0
    
    scalingFactorCount = 0
    scalingFactors = []
    maScalingFactor = 1.0
    
    frameNo = 0

    # Parameters
    scaleParameter = 100
    eKernelParameter = 2
    cThreshParameter = 100

    # Load the TensorRT engine
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    
    # Allocate buffers
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Initialise the background subtractor
    fgbgC = cv2.createBackgroundSubtractorMOG2()

    # Video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
                
	        # Rotate frame if necessary (slow)
            angle_degrees = 4
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)
            rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h))
            frame = rotated_frame
	    
	        # Preprocessing to ensure correct input size
            resized_frame, startTime = preprocess(frame, target_size=(1088, 1920), scaleParameter=100)
            
            # Copy preprocessed frame to input buffer
            np.copyto(inputs[0]['host'], np.ravel(resized_frame))

            # Inference
            inferenceResults = infer(context, bindings, inputs, outputs, stream)

            # Postprocess and get results
            splatterSeverity, dst, airKnifePosCount, airKnifePoss, maAirKnifePos, scalingFactorCount, scalingFactors, maScalingFactor = postprocess(inferenceResults, airKnifePosCount, scaleParameter, airKnifePoss, scalingFactors, scalingFactorCount, fgbgC, frame, eKernelParameter, cThreshParameter, startTime, maAirKnifePos, maScalingFactor)
            
            frameNo +=1
            print(frameNo)

            # Show result
            cv2.imshow('Result', dst)
            cv2.waitKey(1)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
