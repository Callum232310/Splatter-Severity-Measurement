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

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


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
    

    # Final severity levels
    if ((0 <= whitePixels <= amount0) and (0 <= splatterWidth <= width0)):
            splatterSeverity = 0

    elif ((amount0 <= whitePixels <= amount1) and (width0 <= splatterWidth <= width1)) or ((amount0 <= whitePixels <= amount1) and (0 <= splatterWidth <= width0)) or ((0 <= whitePixels <= amount0) and (width0 <= splatterWidth <= width1)):
            splatterSeverity = 1
            
    elif ((amount1 <= whitePixels <= amount2) and (width1 <= splatterWidth <= width2)) or ((amount0 <= whitePixels <= amount1) and (width1 <= splatterWidth <= width2)) or ((0 <= whitePixels <= amount0) and (width1 <= splatterWidth <= width2)) or ((amount1 <= whitePixels <= amount2) and (width0 <= splatterWidth <= width1)) or ((amount1 <= whitePixels <= amount2) and (0 <= splatterWidth <= width0)):
            splatterSeverity = 2
            
    elif ((amount2 <= whitePixels <= amount3) and (width2 <= splatterWidth <= width3)) or ((amount1 <= whitePixels <= amount2) and (width2 <= splatterWidth <= width3)) or ((amount0 <= whitePixels <= amount1) and (width2 <= splatterWidth <= width3)) or ((0 <= whitePixels <= amount0) and (width2 <= splatterWidth <= width3)) or ((amount2 <= whitePixels <= amount3) and (width1 <= splatterWidth <= width2)) or ((amount2 <= whitePixels <= amount3) and (width0 <= splatterWidth <= width1)) or ((amount2 <= whitePixels <= amount3) and (0 <= splatterWidth <= width0)):
            splatterSeverity = 3
            
    else:
            splatterSeverity = 4


    endTime = time.time()
    frameTime = endTime - startTime
     
    print("splatterSeverity: %d, frameTime: %.2f" %(splatterSeverity, frameTime))

    #profile_section(start, "POSTPROCESS END")
    
    return splatterSeverity, airKnifePosCount, airKnifePoss, maAirKnifePos, scalingFactorCount, scalingFactors, maScalingFactor



