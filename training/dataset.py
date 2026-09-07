import os
import cv2
import random
import numpy as np
from typing import Tuple, List, Optional
try:
    import albumentations as A
    _HAS_ALBUMENTATIONS = True
except Exception:
    _HAS_ALBUMENTATIONS = False

class SyntheticCricketBallDataset:
    """
    Generates synthetic broadcast cricket match frames with high-speed (>140km/h)
    motion-blurred balls, stadium clutter, and produces YOLO-formatted training data.
    """
    def __init__(self, output_dir: str = "dataset"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels", "val"), exist_ok=True)

        if _HAS_ALBUMENTATIONS:
            self.transform = A.Compose([
                A.Affine(scale=(0.98, 1.02), translate_percent=(-0.02, 0.02), rotate=(-3, 3), p=0.3, border_mode=cv2.BORDER_CONSTANT),
                A.MotionBlur(blur_limit=15, p=0.8),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.4),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = None

    def generate_synthetic_frame(
        self,
        width: int = 1280,
        height: int = 720
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Synthesizes a realistic broadcast pitch frame with an elongated/motion-blurred cricket ball.
        Returns:
            frame: (H, W, 3) BGR image
            bbox_yolo: (center_x_norm, center_y_norm, width_norm, height_norm)
        """
        # 1. Pitch / Turf Background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Green turf
        frame[:, :] = (30 + random.randint(-5, 5), 140 + random.randint(-10, 10), 30 + random.randint(-5, 5))
        
        # Pitch Clay Strip in Center
        pitch_left = int(width * 0.28)
        pitch_right = int(width * 0.72)
        frame[:, pitch_left:pitch_right] = (120 + random.randint(-8, 8), 160 + random.randint(-8, 8), 180 + random.randint(-8, 8))

        # Crease markings
        cv2.line(frame, (pitch_left + 20, 180), (pitch_right - 20, 180), (255, 255, 255), 3)
        cv2.line(frame, (pitch_left + 20, 580), (pitch_right - 20, 580), (255, 255, 255), 3)

        # Stumps
        for sx in [-20, 0, 20]:
            cx = width // 2 + sx
            cv2.line(frame, (cx, 180), (cx, 120), (200, 200, 200), 2)
            cv2.line(frame, (cx, 580), (cx, 640), (200, 200, 200), 2)

        # 2. Add Red / White Cricket Ball with High-Speed Motion Streak
        bx = random.randint(int(width * 0.35), int(width * 0.65))
        by = random.randint(int(height * 0.25), int(height * 0.75))
        
        # Motion streak parameters
        streak_len = random.randint(12, 35)
        streak_angle = random.uniform(-0.4, 0.4)
        ball_color = (30, 30, 220) if random.random() > 0.3 else (240, 240, 240)  # Red ball or White ball

        # Draw motion streak line
        p1 = (int(bx - streak_len * np.cos(streak_angle)), int(by - streak_len * np.sin(streak_angle)))
        p2 = (int(bx + streak_len * np.cos(streak_angle)), int(by + streak_len * np.sin(streak_angle)))
        cv2.line(frame, p1, p2, ball_color, thickness=random.randint(6, 12), lineType=cv2.LINE_AA)
        cv2.circle(frame, (bx, by), random.randint(5, 9), ball_color, -1, lineType=cv2.LINE_AA)

        # Bounding Box calculation
        min_x = max(0, min(p1[0], p2[0]) - 8)
        max_x = min(width - 1, max(p1[0], p2[0]) + 8)
        min_y = max(0, min(p1[1], p2[1]) - 8)
        max_y = min(height - 1, max(p1[1], p2[1]) + 8)

        bw = max_x - min_x
        bh = max_y - min_y
        cx_norm = (min_x + bw / 2.0) / width
        cy_norm = (min_y + bh / 2.0) / height
        bw_norm = bw / width
        bh_norm = bh / height

        bbox_yolo = (cx_norm, cy_norm, bw_norm, bh_norm)

        # 3. Apply Albumentations augmentations if available
        if self.transform is not None:
            try:
                augmented = self.transform(
                    image=frame,
                    bboxes=[list(bbox_yolo)],
                    class_labels=[0]
                )
                if augmented['bboxes']:
                    frame = augmented['image']
                    bbox_yolo = tuple(augmented['bboxes'][0])
            except Exception:
                pass

        return frame, bbox_yolo

    def build_dataset(self, n_train: int = 50, n_val: int = 10) -> str:
        """
        Generates sample synthetic dataset and writes dataset.yaml.
        """
        splits = [("train", n_train), ("val", n_val)]
        for split_name, count in splits:
            for i in range(count):
                img, bbox = self.generate_synthetic_frame()
                img_path = os.path.join(self.output_dir, "images", split_name, f"frame_{i:04d}.jpg")
                lbl_path = os.path.join(self.output_dir, "labels", split_name, f"frame_{i:04d}.txt")

                cv2.imwrite(img_path, img)
                with open(lbl_path, "w") as f:
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        # Create dataset.yaml
        yaml_path = os.path.join(self.output_dir, "dataset.yaml")
        yaml_content = f"""path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val
names:
  0: cricket_ball
"""
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        return yaml_path
