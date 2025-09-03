import argparse
import os
import sys
import time

import requests
import csv
from datetime import datetime  # Optional: for recording time (add if needed)

# Define CSV file path
CSV_FILE = "prediction_results.csv"

def download_pdf_from_server(server_url, filename, save_dir="downloaded"):
    try:
        os.makedirs(save_dir, exist_ok=True)
        params = {'filename': filename}
        response = requests.get(server_url, params=params, stream=True)
        if response.status_code == 200:
            base_name, ext = os.path.splitext(filename)
            save_path = os.path.join(save_dir, base_name)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter empty chunks
                        f.write(chunk)
            print(f"File downloaded successfully, saved to: {save_path}")
            return save_path
        else:
            try:
                error_info = response.json().get('info', 'Unknown error')
            except:
                error_info = response.text
            print(f"Download failed (status code: {response.status_code}): {error_info}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Network request error: {str(e)}")
    except IOError as e:
        print(f"File write error: {str(e)}")
    except Exception as e:
        print(f"Download failed: {str(e)}")
    return None

def init_csv():
    """Initialize CSV file, create header on first run"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "status", "message"])  # Header: status, message

def write_to_csv(status, message):
    """Append results to CSV file"""
    init_csv()
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, message])

def process_image(image_path, male):
    try:
        if not os.path.exists(image_path):
            error_msg = f"Image file does not exist: {image_path}"
            write_to_csv("Failed", error_msg)
            print(error_msg)
            return
        # 1. Open image
        with open(image_path, 'rb') as image_file:
            filename = os.path.split(image_path)[-1]
            data = {
                "male": male,
                "filename": filename
            }
            # 2. Upload image
            response1 = requests.post(
                "http://localhost:5001/upload",
                files={"image": image_file},
                data=data
            )
            if response1.status_code == 200:
                result = response1.json()
                write_to_csv("Success" if result['success'] else "Failed", str(result['info']))
                print(f"Image upload: {result}")
                # 3. Predict
                response = requests.post(
                    "http://localhost:5001/predict",
                    data=data
                )
                # Process response result
                if response.status_code == 200:
                    try:
                        result = response.json()
                        write_to_csv("Success" if result['success'] else "Failed", str(result['info']))
                        print(f"Prediction successful: {result}")
                        # 4. Download PDF report
                        pdf_filename = str(result['pdfpath'])
                        if len(pdf_filename) <= 0:
                            print(f"No valid PDF obtained")
                        else:  # Download PDF
                            server_url = "http://localhost:5001/download/pdf"
                            download_pdf_from_server(server_url, pdf_filename)
                            # print(f"Report saved in: downloaded/{pdf_filename}")

                    except ValueError:
                        error_msg = f"Response format error, not JSON: {response.text}"
                        write_to_csv("Failed", error_msg)
                        print(error_msg)
                else:
                    # Backend returns error status code (e.g., 400, 500, etc.)
                    error_msg = f"Backend error (status code: {response.status_code}): {response.text}"
                    write_to_csv("Failed", response.json()['info'])
                    print(error_msg)
            else:
                error_msg = f"Backend error (status code: {response1.status_code}): {response1.text}"
                write_to_csv("Failed", response1.json()['info'])
                print(error_msg)
    except requests.exceptions.RequestException as e:
        # Network request related exceptions (e.g., connection timeout, unreachable, etc.)
        error_msg = f"Request exception: {str(e)}"
        write_to_csv("Failed", error_msg)
        print(error_msg)
    except Exception as e:
        error_msg = f"Unknown error: {str(e)}"
        write_to_csv("Failed", error_msg)
        print(error_msg)

# if __name__ == "__main__":
#     # Test example
#     # path = os.path.join("/media/dzy/deep1/train_data2/guling", "boneage-training-dataset", "7256.png")
#     path = os.path.join("test", "1959.png")  # Note: If it's an image, it's recommended to confirm the file format is correct (e.g., .png/.jpg)
#     process_image(path, male=1)