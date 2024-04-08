# -*- coding: utf-8 -*-
"""
Script Name: fastAPI.py

Description: This script is for setting up the Uvicorn server on the Jetson Orin Nano so that the client-side device can connect to it. Upon successful connection, video can be streamed to this device, processed by the TensorRT splatter model, and the results are sent back.

Original Author: Callum O'Donovan

Original Creation Date: January 31st 2024

Email: callumodonovan2310@gmail.com

Disclaimer: This script is part of a project focusing on practical application in engineering.
            For full code quality and contribution guidelines, see the README file. 
            
"""

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import os
import tensorrt as trt
import numpy as np
from knifeSplatter_Final_Deployment import load_engine, allocate_buffers, preprocess, infer, postprocess
import time
import json

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
IMAGEDIR = "images/"
files = os.listdir(IMAGEDIR)

app = FastAPI()

# Initialise global variables
context, bindings, inputs, outputs, stream = None, None, None, None, None

def decode_frame(frame_data):
    
    # Convert the bytes data to a numpy array
    nparr = np.frombuffer(frame_data, np.uint8)
    
    # Decode the numpy array to an image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return frame


def profile_api_section(start_time, section_name=""): #This function is used for timing each step
    end_time = time.time()
    print(f"{section_name} took {(end_time - start_time) * 1000:.2f} ms")


@app.on_event("startup") #Initialise TensorRT
async def load_engine_context_buffers():
    print("TESTING CONTEXT")
    global context, bindings, inputs, outputs, stream
    
    # Load the TensorRT engine
    engine_path = '/usr/src/app/models/bestcw.engine'
    engine = load_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)


@app.websocket("/video") #Create an endpoint for the video from the client-side device to be streamed to
async def video_stream(websocket: WebSocket):
    start = time.time()
    print("TESTING VIDEO STREAM")
    
    await websocket.accept()
    await websocket.send_text("CONNECTION MADE")
    profile_api_section(start, "CONNECTION")
    try:
        start = time.time()
        setup_data = await websocket.receive_text()  # The first message should be the setup parameters in JSON format

        setup_params = json.loads(setup_data)
        profile_api_section(start, "SETUP PARAMS")

        scaleParameter = setup_params.get("scaleParameter", 100)  # Provide default values for customisable parameters
        eKernelParameter = setup_params.get("eKernelParameter", 2)
        cThreshParameter = setup_params.get("cThreshParameter", 100)
        print(f"Received scaleParameter: {scaleParameter}")
        start = time.time()
        
        # Initialize the background subtractor
        fgbgC = cv2.bgsegm.createBackgroundSubtractorCNT() #GOOD
        
        profile_api_section(start, "BGSEGM")
               
        # Initialise some variables
        airKnifePosCount = 0
        airKnifePoss = []
        maAirKnifePos = 0
        
        scalingFactorCount = 0
        scalingFactors = []
        maScalingFactor = 1.0

        startTime = time.time()
  
        while True:
            try:
                start = time.time()
                
                # Receive frame data from WebSocket
                print("RECEIVING DATA")
                data = await websocket.receive_bytes()
                print("RECEIVED DATA")
                profile_api_section(start, "RECEIVE FRAME")
                
                
                
                start = time.time()
                
                # Decode the frame data to an image
                frame = decode_frame(data)          
        
                profile_api_section(start, "DECODE FRAME")
                
                
                
                start = time.time()
                
                # Resize the frame
                resized_frame, startTime = preprocess(frame, scaleParameter)
                
                profile_api_section(start, "PREPROCESS FRAME")
                
                
                
                start = time.time()
                
                # Make a YOLOv5 inference on the frame
                inferenceResults = infer(context, bindings, inputs, outputs, stream)
                
                profile_api_section(start, "INFER FRAME")
                
                
                
                start = time.time()
                
                # Finish the rest of the processing such as BGS and calculating results
                splatterSeverity, airKnifePosCount, airKnifePoss, maAirKnifePos, scalingFactorCount, scalingFactors, maScalingFactor = postprocess(inferenceResults, airKnifePosCount, scaleParameter, airKnifePoss, scalingFactors, scalingFactorCount, fgbgC, frame, eKernelParameter, cThreshParameter, startTime, maAirKnifePos, maScalingFactor)
                
                profile_api_section(start, "POSTPROCESS FRAME")
                
                start = time.time()
                
                #Send the splatter severity value back to the client-side device
                await websocket.send_text(f"Splatter Severity: {splatterSeverity}")
                
                profile_api_section(start, "SEND SEVERITY")
                
                # Wait for an acknowledgment from the client
                ack = await websocket.receive_text()
            except Exception as e:
                print(f"Error during frame processing: {e}")
                break
    except WebSocketDisconnect as disconnect_error:
        print(f"WebSocket disconnected, code: {disconnect_error.code}")
    except Exception as e:
        print(f"Unhandled exception: {e}")    
  

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) #Host the server on the device that executes this code, through port 8000
