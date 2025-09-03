import os
import threading
import time
from datetime import datetime
from flask import Flask, send_from_directory, request, jsonify
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

from config import exceededTime, supportFormat
from controller import BoneAgePredictor
from data_process.data_config import rootdata
from utils.generateReport import generate_bone_age_report

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load model
predictor = BoneAgePredictor(ckp_path="output/debug/version_0/ckp/best_model.ckpt",)

class PredictionThread(threading.Thread):
    """Class for executing prediction tasks in subthreads, storing prediction results and exceptions"""
    def __init__(self, image_path, male):
        super().__init__()
        self.image_path = image_path
        self.male = male
        self.result = None  # Store prediction results
        self.error = None   # Store exception information

    def run(self):
        """Execute prediction task in subthread"""
        try:
            self.result = predictor.predict_single_image(self.image_path, self.male)
        except Exception as e:
            self.error = str(e)  # Capture exceptions

@app.route("/upload", methods=["POST"])
def uploadpic():
    if 'image' not in request.files:
        return jsonify({"success": False,"info": "No image file provided"}), 400

    image_file = request.files['image']
    filename = request.form.get('filename', '')
    appendfix = filename.split(".")[-1]
    if appendfix not in supportFormat:
        allfor = ",".join(supportFormat)
        return jsonify({"success": False,"info": f"Unsupported image format, currently supported: { allfor }"}), 400
    try:
        # Save uploaded image
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        save_path = os.path.join("upload", filename)
        image.save(save_path)
        return jsonify({"success": True,"info": f"picture upload success"}), 200
    except Exception as e:
        return jsonify({"success": False,"info": f"Server error: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    male = request.form.get('male', 0)
    filename = request.form.get('filename', '')

    try:
        # Save uploaded image
        save_path = os.path.join("upload", filename)
        if not os.path.exists(save_path):
            return jsonify({"success": False,"info": f"pic not exists "}), 400

        # Create and start prediction subthread
        pred_thread = PredictionThread(
            image_path=save_path,
            male=int(male)
        )
        pred_thread.start()
        # Main thread waits for subthread, maximum wait of exceededTime seconds
        pred_thread.join(timeout=exceededTime)  # Timeout in seconds: exceededTime
        # Determine result
        if pred_thread.is_alive():
            # Subthread is still running
            return jsonify({"success": False,"info": f"Prediction timed out (exceeded {exceededTime} seconds)"}), 504  # 504 is gateway timeout status code
        if pred_thread.error:
            # Error occurred in subthread execution
            return jsonify({"success": False,"info": f"Prediction failed: {pred_thread.error}"}), 500
        # Prediction successful, save pdf
        result = pred_thread.result
        result['pdfpath'] =''
        if pred_thread.result['success']:
            print("age:" , pred_thread.result['info'])
            times = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            name = filename.split(".")[0]
            generate_bone_age_report(
                records=[ (
                    f"./upload/{filename}",
                    f"./upload/heat_{filename}",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    pred_thread.result['info'] ) ],
                filename_time=f"{times}_{name}",
                report_title="Bone Age Prediction Report",
                font_path="./utils/SimSun.ttf"  # Replace with actual font path
            )
            result['pdfpath'] = f"{times}_{name}_BoneAgePredictionReport.pdf"

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"success": False,"info": f"Server error: {str(e)}"}), 500

@app.route('/download/pdf', methods=['GET'])
def download_pdf():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"success": False, "info": "Please provide filename parameter (filename)"}), 400
    # Concatenate full path
    pdf_path = os.path.join("./upload", filename)
    print(pdf_path)
    # Check if file exists
    if not os.path.exists(pdf_path):
        return jsonify({"success": False, "info": f"File {filename} does not exist"}), 404
    # Check if it's a PDF file
    if not filename.endswith('.pdf'):
        return jsonify({"success": False, "info": "Only PDF files are supported for download"}), 400
    try:
        return send_from_directory(
            "./upload",
            filename,
            as_attachment=True,
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({"success": False, "info": f"Download failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)