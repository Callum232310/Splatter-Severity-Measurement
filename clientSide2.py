# -*- coding: utf-8 -*-
"""
Script Name: clientSide2.py

Description: This script is for the client-side device to connect to the Uvicorn server over wi-fi. The device that executes this code should have the video source.

Original Author: Callum O'Donovan

Original Creation Date: February 13th 2024

Email: callumodonovan2310@gmail.com
    
Disclaimer: This script is part of a project focusing on practical application in engineering.
            For full code quality and contribution guidelines, see the README file. 
            
"""

import asyncio
import cv2
import json
import websockets
import logging
import numpy as np
import time


#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)


def profile_client_section(start_time, section_name=""):
    end_time = time.time()
    print(f"{section_name} took {(end_time - start_time) * 1000:.2f} ms")


async def send_frame(websocket, frame, frame_number):
    start_time = time.time()
    _, frame_encoded = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
    #logging.debug("DEBUG 1")
    await websocket.send(frame_encoded.tobytes())  # Send the binary data
    #logging.debug("DEBUG 2")
    logging.info(f"Frame {frame_number} sent.")  # Log after sending
    #logging.debug("DEBUG 3")
    result = await websocket.recv()  # Receive the result from the server (splatter severity value)
    #logging.debug("DEBUG 4")
    logging.info(f"Received result for frame {frame_number}: {result}")
    
    
    end_time = time.time()  # End timing after receiving the result
    total_rtt = end_time - start_time  # Calculate the total round-trip time
    logging.info(f"Total RTT for frame {frame_number}: {total_rtt} seconds.")
    
    
    # Send acknowledgment to the server indicating readiness for the next frame
    await websocket.send("ACK")
    logging.info(f"Acknowledgment sent for frame {frame_number}.")
    
    
async def video_capture_thread(websocket):
    logging.info("Starting video capture thread.")
    cap = cv2.VideoCapture(r'C:\Users\Callum\Downloads\splatter2023\splatter1.mp4')  # Change to 0 for live camera feed
    previous_frame = None
    frame_number = 0
    
    if not cap.isOpened():
        logging.error("Error opening video source.")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Attempting to send frame {frame_number}")
                
                if previous_frame is None: # Check if the current frame is identical to the previous frame to ensure dataflow is functioning properly
                    print("PREVIOUS FRAME IS NONE")
                elif previous_frame is not None and np.array_equal(frame, previous_frame):
                    logging.warning(f"Duplicate frame detected at frame number {frame_number}.")
                else:
                    logging.debug("DEBUG 1")
                    await send_frame(websocket, frame, frame_number)
                    logging.debug("DEBUG 2")
                    logging.debug(f"Frame {frame_number} sent.")
                
                # Update the previous frame and increase the frame counter
                previous_frame = frame
                frame_number += 1
            else:
                logging.warning("Failed to read frame from capture source.")
                break
            #logging.debug("DEBUG 5")
            await asyncio.sleep(1/25)  # Assuming 25 fps
            #logging.debug("DEBUG 6")
    except Exception as e:
        logging.error(f"Exception occurred in video_capture_thread: {e}")
    finally:
        cap.release()
        logging.info("Video capture thread ended.")



async def main():
    uri = "ws://192.168.1.113:8000/video" #Set the IP address to that of the Jetson Orin Nano (192.168.1.113), the port exposed by fastAPI (8000) and the API endpoint (/video)
    try:
        async with websockets.connect(uri, ping_interval=None, ping_timeout=None, close_timeout=None, read_limit=1000000, write_limit=1000000) as websocket:
            setup_data = {
                "scaleParameter": 100, #Custom parameters for optimising the model
                "eKernelParameter": 2,
                "cThreshParameter": 100
            }
            #logging.debug("DEBUG 7")
            await websocket.send(json.dumps(setup_data))
            #logging.debug("DEBUG 8")
            await video_capture_thread(websocket)
            #logging.debug("DEBUG 9")
    except Exception as e:
        logging.error(f"WebSocket connection error: {e}")
        await asyncio.sleep(5)  # Wait for 5 seconds before retrying
        await main()  # Attempt to reconnect


if __name__ == "__main__":
    asyncio.get_event_loop().set_debug(True)
    asyncio.run(main())
    



