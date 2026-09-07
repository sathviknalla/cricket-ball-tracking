import os
import argparse
from typing import Optional

def train_ball_detector(
    data_yaml: str = "dataset/dataset.yaml",
    model_weights: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 1280,
    batch: int = 8,
    project: str = "runs/train",
    name: str = "cricket_ball_detector"
):
    """
    Fine-tunes YOLOv8 specifically for high-speed small-object cricket ball detection.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Training] Ultralytics is required for training. Run: pip install ultralytics")
        return None

    print(f"[Training] Initializing YOLOv8 model from {model_weights}...")
    model = YOLO(model_weights)

    # Train with high resolution (1280px) and small-object augmentation
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        cos_lr=True,
        close_mosaic=5,
        fl_gamma=1.5,  # Focal loss gamma for hard background false-positive mining
        verbose=True
    )
    print(f"[Training] Model fine-tuning completed.")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Cricket Ball Dataset")
    parser.add_argument("--data", type=str, default="dataset/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size")
    args = parser.parse_args()

    train_ball_detector(data_yaml=args.data, epochs=args.epochs, imgsz=args.imgsz)
