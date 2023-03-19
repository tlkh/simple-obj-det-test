from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection, AutoImageProcessor

print("Downloading dataset")

ds = load_dataset("keremberke/plane-detection", 'full')

print("Downloading model")

model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
aip = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

print("Done!")
