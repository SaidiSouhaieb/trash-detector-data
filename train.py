import argparse
from ultralytics import YOLO
import sys
import ruamel.yaml
import os

if __name__ == "__main__":
    model_path = "model/base/yolo11x.pt"
    yaml_path = "config.yaml"

    model = YOLO(model_path)
    model.nc = 1
    model.train(
        data=yaml_path,
        epochs=10,
        batch=6,
    )
