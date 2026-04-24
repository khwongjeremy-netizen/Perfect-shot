import cv2
import numpy as np
import math
from collections import deque

def run_analysis(VIDEO_SOURCE, mode="Initial Launch"):
    # --- 1. CONFIGURATION ---
    STRIKE_APPROACH_FRAME = 120  
    BALL_COLOR = (0, 0, 255)   # Red
    FOOT_COLOR = (255, 255, 0) # Cyan/Yellow
    TEXT_COLOR = (0,0,0) #BLACK
    TRAJECTORY_LIMIT = 60      # Length of the visible trail
    LOOKBACK = 10 

    shot_category = "Analysis Incomplete" 
    ball_points = deque(maxlen=TRAJECTORY_LIMIT)
    foot_points = deque(maxlen=TRAJECTORY_LIMIT)

    video_stream = cv2.VideoCapture(VIDEO_SOURCE)
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, STRIKE_APPROACH_FRAME)

    is_loaded, current_frame = video_stream.read()
    if not is_loaded: return "Error: Video Failed"

    # --- 2. TRACKER SETUP ---
    analytics_tracker = cv2.legacy.MultiTracker_create()
    # --- 2. MULTI-OBJECT TRACKING SETUP ---
    analytics_tracker = cv2.legacy.MultiTracker_create()

    for i in range(2):
        # Create a fresh copy of the frame so previous text is removed
        display_frame = current_frame.copy()
        
        target = "BALL" if i == 0 else "FOOT"
        color = (0, 0, 255) if i == 0 else (255, 255, 0) # Red for ball, Cyan for foot
        
        # Draw the instruction on the COPY, not the original
        cv2.putText(display_frame, f"SELECT {target} (ENTER to confirm)", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Show the display_frame with the specific instruction
        roi = cv2.selectROI("Selection Window", display_frame, False)
        
        # Add to tracker using the original current_frame (clean)
        analytics_tracker.add(cv2.legacy.TrackerCSRT_create(), current_frame, roi)

    cv2.destroyWindow("Selection Window")

    # --- 3. MAIN LOOP ---
    while True:
        is_loaded, current_frame = video_stream.read()
        if not is_loaded: break

        success, boxes = analytics_tracker.update(current_frame)

        if success:
            for i, box in enumerate(boxes):
                x, y, w, h = [int(v) for v in box]
                center = (x + w // 2, y + h // 2)
                if i == 0: 
                    ball_points.appendleft(center)
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), BALL_COLOR, 2)
                else: 
                    foot_points.appendleft(center)
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), FOOT_COLOR, 2)

        # --- 4. DRAWING TRACERS (The Tracer Logic) ---
        # Draw Ball Tracer (Tapered effect)
        for i in range(1, len(ball_points)):
            if ball_points[i - 1] is None or ball_points[i] is None: continue
            # Calculate thickness so the tail fades out
            thickness = int(np.sqrt(TRAJECTORY_LIMIT / float(i + 1)) * 2.5)
            cv2.line(current_frame, ball_points[i - 1], ball_points[i], BALL_COLOR, thickness)

        # Draw Foot Tracer (Solid line)
        for i in range(1, len(foot_points)):
            if foot_points[i - 1] is None or foot_points[i] is None: continue
            cv2.line(current_frame, foot_points[i - 1], foot_points[i], FOOT_COLOR, 2)

        # --- 5. PHYSICS & MODE LOGIC ---
        if len(ball_points) >= LOOKBACK + 1:
            if mode == "Initial Launch" and shot_category != "Analysis Incomplete":
                pass 
            else:
                dx = ball_points[0][0] - ball_points[LOOKBACK][0]
                dy = ball_points[LOOKBACK][1] - ball_points[0][1]
                angle = math.degrees(math.atan2(dy, dx))
                
                if dx > 2:
                    if angle < 15: shot_category = "Low Driven / Power"
                    elif 15 <= angle <= 35: shot_category = "Floating Cross / Lifted"
                    else: shot_category = "Chip / Lob"

                # Overlay current stats on the video
                cv2.putText(current_frame, f"Angle: {int(angle)} Deg", (50, 80), 1, 1.5, (255, 255, 255), 2)
                cv2.putText(current_frame, f"Shot: {shot_category}", (50, 110), 1, 1.5, (0, 255, 255), 2)

        # Display UI elements on video
        cv2.putText(current_frame, f"MODE: {mode}", (50, 40), 1, 1.2, (0, 255, 0), 2)
        
        cv2.imshow("Striker Analytics v1.0", current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video_stream.release()
    cv2.destroyAllWindows()
    return shot_category