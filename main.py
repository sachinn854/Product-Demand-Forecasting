# main.py

import os
import sys

# Add current directory and 'src' to the Python path
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, 'src')
sys.path.append(current_dir)
sys.path.append(src_dir)

from src.pipeline import run_full_pipeline  # absolute import

if __name__ == "__main__":
    data_path = "C:\\Users\\sachi\\OneDrive\\Product Demand Forecasting\\6_Merge File\\Master_file.csv"
    run_full_pipeline(data_path)
