import os
import logging.config
import pandas as pd
import torch
import yaml
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import warnings

from data_process.data_config import rootdata
from heatmap import GradCAM
from lib.datasets import HandDatamodule

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BoneAgePredictor:
    def __init__(self, ckp_path="output/debug/version_0/ckp/best_model.ckpt",
                 input_size=[1, 512, 512], mean=127.3207517246848,
                 std=41.182021399396326, device="cuda" if torch.cuda.is_available() else "cpu"):

        self.ckp_path = ckp_path
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.device = device
        self.model = None
        self.transform = self._create_transform()

        # Load model
        self.load_model()
        self.data_augmentation = HandDatamodule.get_inference_augmentation(512,512,
                                                                           flip=False,
                                                                           rotation_angle=0,
                                                                           )

        self.target_layer = "backbone.base._conv_head"


    def _create_transform(self):
        """Create image preprocessing transformations"""
        return transforms.Compose([
            transforms.Resize((self.input_size[1], self.input_size[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming standardization is used
        ])

    def load_model(self):
        """Load bone age prediction model"""
        try:
            from lib.models import BoneAgeModel
            logger.info(f"Loading model from {self.ckp_path}...")
            self.model = BoneAgeModel.load_from_checkpoint(
                self.ckp_path,
                weights_only=False
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

    def preprocess_image(self, image_path):
        image = cv2.imread(os.path.abspath(image_path), cv2.IMREAD_ANYDEPTH)
        origin_img = cv2.imread(os.path.abspath(image_path))
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
        image = self.data_augmentation(image=image)["image"]
        image = image.to(torch.float32)
        m = image.mean()
        sd = image.std()
        image = (image - m) / sd
        return image,origin_img,image_path.split(os.sep)[-1]

    def predict_single_image(self, image, male=1):
        img_tensor,origin_img,img_name = self.preprocess_image(image)
        img_tensor = img_tensor[None].to(self.device)
        try:
            male_tensor = torch.tensor([[male]], dtype=torch.float32).to(self.device)

            # Generate heatmap
            grad_cam = GradCAM(self.model, self.target_layer)  # Re-instantiate for each new image to ensure proper hook registration
            heatmap, pred_normalized = grad_cam(img_tensor, male_tensor)
            heatmap_resized = cv2.resize(heatmap, (origin_img.shape[1], origin_img.shape[0]))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            # heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # Convert to RGB format
            superimposed = cv2.addWeighted(origin_img, 0.6, heatmap_colored, 0.4, 0)  # Weighted fusion
            cv2.imwrite(f"./upload/heat_{img_name}",superimposed)

            with torch.no_grad():
                outputs = self.model(img_tensor , male_tensor)
                y_hat = outputs[0][0][0]
                # Denormalize to get actual bone age (months)
                if isinstance(y_hat, torch.Tensor):
                    y_hat = y_hat.cpu().numpy()
                predicted_age = y_hat * self.std + self.mean
            logger.info(f"Predicted bone age: {predicted_age:.2f} months")
            return {
                "success": True,
                "info": f"{ round(float(predicted_age),2)}",

            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "success": False,
                "info": str(e),

            }

    def predict_batch(self, image_paths, males=None):
        if self.model is None:
            logger.error("Model not loaded, please load the model first")
            return None

        # Create custom dataset
        class BatchDataset(Dataset):
            def __init__(self, image_paths, males, transform):
                self.image_paths = image_paths
                self.males = males if males is not None else [0.5]*len(image_paths)
                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img = Image.open(self.image_paths[idx]).convert('L')
                img = self.transform(img)
                male = torch.tensor(self.males[idx], dtype=torch.float32)
                return {"x": img, "male": male, "image_name": self.image_paths[idx]}

        # Create data loader
        dataset = BatchDataset(image_paths, males, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2
        )

        # Batch prediction
        results = []
        try:
            with torch.no_grad():
                for batch in loader:
                    images = batch["x"].to(self.device)
                    males = batch["male"].unsqueeze(1).to(self.device)
                    outputs = self.model(images, males)

                    y_hat = outputs["y_hat"] if isinstance(outputs, dict) else outputs
                    y_hat = y_hat.cpu().numpy()

                    # Denormalize
                    predicted_ages = y_hat * self.std + self.mean

                    # Save results
                    for i, path in enumerate(batch["image_name"]):
                        results.append({
                            "image_path": path,
                            "predicted_age": float(predicted_ages[i]),
                            "success": True
                        })

            logger.info(f"Batch prediction completed, processed {len(results)} images in total")
            return results
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            return [{
                "error": str(e),
                "success": False
            }]


# Frontend API wrapper
class BoneAgeAPI:
    def __init__(self, model_path="output/debug/version_0/ckp/best_model.ckpt"):
        self.predictor = BoneAgePredictor(ckp_path=model_path)
    def predict_from_image_file(self, file_path, male=None):
        if not os.path.exists(file_path):
            return {"error": "File does not exist", "success": False}
        return self.predictor.predict_single_image(file_path, male)

    def predict_from_base64(self, base64_str, male=None):

        try:
            import base64
            from io import BytesIO

            # Decode base64 string
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data))

            return self.predictor.predict_single_image(img, male)
        except Exception as e:
            return {"error": f"Base64 decoding or prediction failed: {str(e)}", "success": False}

    def predict_batch_images(self, image_paths, males=None):

        return self.predictor.predict_batch(image_paths, males)


# Usage example
if __name__ == "__main__":
    predictor = BoneAgePredictor()
    test_image_path = os.path.join(rootdata,"boneage-training-dataset","7256.png")
    result = predictor.predict_single_image(test_image_path, male=1)
    print(f"Single image prediction result: {result}")

    # api = BoneAgeAPI()
    # api_result = api.predict_from_image_file(test_image_path, male=0)
    # print(f"API prediction result: {api_result}")