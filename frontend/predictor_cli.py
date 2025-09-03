import os
import requests

def predict_image(image_path, male):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file does not exist: {image_path}")
        return None

    try:
        with open(image_path, 'rb') as image_file:
            filename = os.path.basename(image_path)
            data = {"male": male, "filename": filename}
            files = {"image": image_file}

            print(f"[INFO] Uploading: {filename}")
            upload_resp = requests.post("http://localhost:5001/upload", data=data, files=files)

        if upload_resp.status_code != 200:
            print(f"[UPLOAD ERROR] {upload_resp.status_code} - {upload_resp.text}")
            return None

        print("[INFO] Uploaded successfully. Start predicting...")
        predict_resp = requests.post("http://localhost:5001/predict", data=data)

        if predict_resp.status_code == 200:
            result = predict_resp.json()
            print("[SUCCESS] prediction result:", result)
            pdf_filename = result.get("pdfpath")

            if not pdf_filename:
                print("[WARN] No PDF file path found in response.")
                return None
            return pdf_filename
        else:
            print(f"[PREDICT ERROR] {predict_resp.status_code} - {predict_resp.text}")
            return None

    except Exception as e:
        print(f"[EXCEPTION] Request failed: {e}")
        return None