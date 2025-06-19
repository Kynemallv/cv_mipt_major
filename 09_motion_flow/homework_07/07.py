import cv2
import numpy as np
from collections import defaultdict


def main():
    video_path = 'original.mp4'
    cap = cv2.VideoCapture(video_path)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=1000,
        varThreshold=32,
        detectShadows=False
    )
    
    tracked_objects = defaultdict(dict)
    object_id = 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1.0
    
    MIN_CONTOUR_AREA = 2000
    MIN_WIDTH = 200
    MIN_HEIGHT = 200
    PADDING_RATIO = 0.08
    
    MAX_DISTANCE = 700
    MAX_FRAMES_SKIP = 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        fgmask = fgbg.apply(blurred)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                continue
            
            padding_w = int(w * PADDING_RATIO)
            padding_h = int(h * PADDING_RATIO)
            
            x += padding_w
            y += padding_h
            w -= 2 * padding_w
            h -= 2 * padding_h
            
            x = max(0, x)
            y = max(0, y)
            w = max(5, w)
            h = max(5, h)
            
            current_boxes.append({
                'box': (x, y, x+w, y+h),
                'center': ((x + x+w)/2, (y + y+h)/2)
            })
        
        updated_ids = []
        objects_to_remove = set()
        
        for obj_id, obj_data in tracked_objects.items():
            if 'last_box' in obj_data:
                best_match = None
                min_dist = MAX_DISTANCE
                
                for i, curr_box in enumerate(current_boxes):
                    dist = np.linalg.norm(np.array(obj_data['center']) - np.array(curr_box['center']))
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match = i
                
                if best_match is not None:
                    tracked_objects[obj_id]['last_box'] = current_boxes[best_match]['box']
                    tracked_objects[obj_id]['center'] = current_boxes[best_match]['center']
                    tracked_objects[obj_id]['speed'] = min_dist * fps * scale
                    tracked_objects[obj_id]['frames_skipped'] = 0
                    
                    updated_ids.append(obj_id)
                    del current_boxes[best_match]
                else:
                    tracked_objects[obj_id]['frames_skipped'] += 1
                    
                    if tracked_objects[obj_id]['frames_skipped'] > MAX_FRAMES_SKIP:
                        objects_to_remove.add(obj_id)
        
        for obj_id in objects_to_remove:
            del tracked_objects[obj_id]
        
        for curr_box in current_boxes:
            tracked_objects[object_id] = {
                'last_box': curr_box['box'],
                'center': curr_box['center'],
                'speed': 0,
                'frames_skipped': 0
            }
            updated_ids.append(object_id)
            object_id += 1
        
        for obj_id, obj_data in tracked_objects.items():
            if 'last_box' in obj_data:
                x1, y1, x2, y2 = obj_data['last_box']
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.putText(frame, f"ID: {obj_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Speed: {obj_data['speed']:.2f}", (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
