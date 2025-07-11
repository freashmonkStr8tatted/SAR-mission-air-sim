# SAR-mission-air-sim
### airsim_llm_sar_drone/main.py

import airsim
import openai
import numpy as np
import cv2
from ultralytics import YOLO
import time

# --- Configuration ---
openai.api_key = 'your-openai-api-key'  # Replace with your actual key
model = YOLO("yolov8n.pt")

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# --- Helper Functions ---
def capture_frame():
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
    return img_rgb

def detect_objects(frame):
    results = model(frame)
    results.show()
    return results

def prompt_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def fly_grid_search(area_size=100, step=20, altitude=-10):
    for x in range(0, area_size, step):
        for y in range(0, area_size, step):
            print(f"Flying to ({x}, {y})")
            client.moveToPositionAsync(x, y, altitude, 3).join()
            frame = capture_frame()
            detect_objects(frame)
            time.sleep(1)

# --- Main Script ---
print("Taking off...")
client.takeoffAsync().join()

mission_prompt = "Plan a 100x100 meter SAR mission using grid search. Avoid obstacles and return to origin if a person is found."
plan = prompt_gpt(mission_prompt)
print("GPT Mission Plan:\n", plan)

# Simple static grid pattern
fly_grid_search()

print("Returning home...")
client.moveToPositionAsync(0, 0, -10, 3).join()
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("Mission complete.")
