import argparse
from predictor_cli import predict_image
import os
import requests

def main():
    def download_pdf_from_server(server_url, filename, save_dir="downloaded"):
        try:
            os.makedirs(save_dir, exist_ok=True)
            response = requests.get(server_url, params={"filename": filename}, stream=True)
            if response.status_code == 200:
                save_path = os.path.join(save_dir, filename)
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"[INFO] PDF downloaded successfully → {save_path}")
            else:
                print(f"[ERROR] Failed to download PDF: {response.text}")
        except Exception as e:
            print(f"[EXCEPTION] PDF download error: {str(e)}")

    parser = argparse.ArgumentParser(description="Bone age prediction CLI tool")
    parser.add_argument("--image", required=True, help="file_path")
    parser.add_argument("--male", required=True, type=int, choices=[0, 1], help="Gender: Female 0 Male 1")

    args = parser.parse_args()
    pdf_filename = predict_image(args.image, args.male)
    if pdf_filename:  
        server_url = "http://localhost:5001/download/pdf"
        download_pdf_from_server(server_url, pdf_filename)
    else:
        print("[ERROR] Download failed.")

if __name__ == "__main__":
    main()