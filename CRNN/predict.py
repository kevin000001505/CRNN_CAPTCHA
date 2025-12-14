import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By
import os
import time
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from model import CRNNModel


def captcha_extract():
    # Screenshot CAPTCHA
    captcha_element = driver.find_element(By.ID, "captchaImage")
    captcha_screenshot = captcha_element.screenshot_as_png

    # Save temporarily
    temp_path = "temp_captcha.png"
    with open(temp_path, "wb") as f:
        f.write(captcha_screenshot)

    return temp_path


def captcha_refresh():
    # Click refresh button
    try:
        refresh_button = driver.find_element(By.ID, "refreshChptcha")
        refresh_button.click()
        time.sleep(2)  # Wait for new captcha to load
    except Exception as e:
        print(f"Error refreshing captcha: {e}")


class CaptchaPredictor:
    def __init__(self, model_filepath):
        # Load model inside the class
        self.model = CRNNModel()
        self.model.load_model(model_filepath)
        self.model.model.eval()  # Set to evaluation mode
        self.model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict_with_probabilities(self, image):
        """Load image and return both predictions and probabilities"""

        image_tensor = self.transform(image).unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            # Get logits from model
            logits = self.model.model(image_tensor)  # Shape: [B, T, C]

            # Convert to probabilities
            probabilities = F.softmax(logits, dim=2)  # Shape: [B, T, C]

            # Get predictions (most likely class at each timestep)
            _, predictions = torch.max(logits, dim=2)  # Shape: [B, T]

            # Decode using your existing method
            decoded_pred = self.model.decode_predictions(logits)
            pred_string = self.model.predictions_to_strings(decoded_pred)[0]

        return {
            "prediction": pred_string,
            "logits": logits.cpu().numpy(),
            "probabilities": probabilities.cpu().numpy(),
            "raw_predictions": predictions.cpu().numpy(),
        }

    def get_prediction_confidence(self, result):
        """Calculate confidence score for the prediction"""

        probabilities = result["probabilities"][0]  # [T, C]
        prediction = result["prediction"]

        # Method 1: Average max probability across timesteps
        max_probs_per_timestep = probabilities.max(axis=1)
        avg_confidence = max_probs_per_timestep.mean()

        # Method 2: Minimum confidence across predicted characters
        min_confidence = max_probs_per_timestep.min()

        # Method 3: Entropy-based confidence (lower entropy = higher confidence)
        entropy_per_timestep = -(probabilities * np.log(probabilities + 1e-8)).sum(
            axis=1
        )
        avg_entropy = entropy_per_timestep.mean()

        return {
            "avg_confidence": avg_confidence,
            "min_confidence": min_confidence,
            "avg_entropy": avg_entropy,
            "prediction": prediction,
        }

    def predict(self, image_path):
        """Predict the captcha text from the image"""
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((128, 32))
            result = self.predict_with_probabilities(img)
            confidence = self.get_prediction_confidence(result)

            return result["prediction"], confidence
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None


if __name__ == "__main__":
    predictor = CaptchaPredictor("checkpoints/best_model.pth")

    url = "https://service.taipower.com.tw/hvcs"
    driver = webdriver.Chrome()
    driver.get(url)
    confidence = {"avg_confidence": 0}

    while confidence["avg_confidence"] < 0.95:
        captcha_refresh()
        image_path = captcha_extract()
        prediction, confidence = predictor.predict(image_path)
        os.remove(image_path)  # Clean up temp image

    if confidence["avg_confidence"] > 0.95:
        print(f"Captcha Prediction: {prediction}")

    time.sleep(5)  # Allow time for the prediction to be displayed
    driver.quit()
