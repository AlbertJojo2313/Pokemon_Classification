import sys
import os

path = os.path.join(os.path.dirname(__file__), "scripts")
sys.path.insert(0, path)
print(f"Added to path: {path}")
print(f"Full sys.path: {sys.path}")
