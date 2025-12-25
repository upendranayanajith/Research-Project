import math
import cv2
import numpy as np

def calculate_angle(center, point):
    """
    Calculates the angle of a point relative to the center.
    0 degrees is 12 o'clock (Up). Rotation is Clockwise.
    """
    cx, cy = center
    tx, ty = point
    dx = tx - cx
    dy = ty - cy 
    
    # atan2(y, x) normally gives CCW from X-axis.
    # We use atan2(dx, -dy) to swap axes: 0 is Up, Positive is CW.
    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad)
    
    if angle_deg < 0:
        angle_deg += 360
        
    return angle_deg

def get_aligned_crop(img, center, angle_to_vertical, box_size=128):
    """
    Rotates the entire image so the hand points UP (0 deg), then crops it.
    Used for Component 3 (Angle Regression).
    """
    h, w = img.shape[:2]
    cx, cy = center
    
    # Rotate image around the center point
    # We rotate by 'angle_to_vertical' because we want to undo the rotation.
    M = cv2.getRotationMatrix2D((cx, cy), angle_to_vertical, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
    
    # Calculate crop coordinates
    half = box_size // 2
    x1 = int(cx - half)
    y1 = int(cy - half)
    
    # Add padding if the crop goes out of bounds
    pad_w, pad_h = 0, 0
    if x1 < 0: pad_w = -x1
    if y1 < 0: pad_h = -y1
    if x1 + box_size > w: pad_w = max(pad_w, (x1 + box_size) - w)
    if y1 + box_size > h: pad_h = max(pad_h, (y1 + box_size) - h)
    
    if pad_w > 0 or pad_h > 0:
        rotated_img = cv2.copyMakeBorder(
            rotated_img, pad_h, pad_h, pad_w, pad_w, 
            cv2.BORDER_CONSTANT, value=(255,255,255)
        )
        x1 += pad_w
        y1 += pad_h
        
    crop = rotated_img[y1:y1+box_size, x1:x1+box_size]
    return crop