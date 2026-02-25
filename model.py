# Load model from hugging face
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="microsoft/deberta-v3-base")
