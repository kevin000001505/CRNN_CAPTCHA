import time
import os
import csv
import requests
import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import matplotlib.pyplot as plt


def get_next_captcha_number():
    """Get the next captcha number based on existing files"""
    data_dir = "data/unprocessed"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    existing_files = [
        f
        for f in os.listdir(data_dir)
        if f.startswith("captcha_") and f.endswith(".png")
    ]
    if not existing_files:
        return 1

    numbers = []
    for f in existing_files:
        try:
            num = int(f.replace("captcha_", "").replace(".png", ""))
            numbers.append(num)
        except ValueError:
            continue

    return max(numbers) + 1 if numbers else 1


def display_image(image_path):
    """Display the captcha image for recognition"""
    try:
        img = Image.open(image_path)
        plt.figure(figsize=(6, 3))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Captcha Image: {os.path.basename(image_path)}")
        plt.show(block=False)  # Non-blocking display
        plt.draw()
        plt.pause(0.1)  # Small pause to ensure image displays
    except Exception as e:
        print(f"Error displaying image: {e}")


def save_captcha_image(driver, captcha_number):
    """Save the captcha image to data/unprocessed/"""
    try:
        captcha_element = driver.find_element(By.ID, "captchaImage")
        captcha_src = captcha_element.get_attribute("src")

        # Download the image
        response = requests.get(captcha_src)
        if response.status_code == 200:
            filename = f"captcha_{captcha_number:03d}.png"
            filepath = os.path.join("data/unprocessed", filename)

            with open(filepath, "wb") as f:
                f.write(response.content)

            return filepath
    except Exception as e:
        print(f"Error saving captcha image: {e}")
        return None


def append_to_csv(image_path, label):
    """Append new data to captcha_labels.csv"""
    csv_path = "data/captcha_labels.csv"

    # Create the CSV file with headers if it doesn't exist
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])

    # Append the new data
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow([image_path, label])


url = "https://service.taipower.com.tw/hvcs"
driver = webdriver.Chrome()
driver.get(url)

captcha_counter = get_next_captcha_number()

while True:
    try:
        # Wait for captcha image to load
        captcha_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "captchaImage"))
        )

        # Save the current captcha image
        image_path = save_captcha_image(driver, captcha_counter)

        if image_path:
            print(f"Saved captcha image: {image_path}")

            display_image(image_path)
            # Get user input for the label
            label = str(input("Please enter the captcha: ")).strip()

            if label:
                # Close the current plot before proceeding
                plt.close()

                # Append to CSV
                append_to_csv(image_path, label)
                print(f"Added to CSV: {image_path} -> {label}")

                # Increment counter for next image
                captcha_counter += 1

                # Refresh captcha for next iteration
                refresh_button = driver.find_element(By.ID, "refreshChptcha")
                refresh_button.click()

                # Wait a bit for the new captcha to load
                time.sleep(1)
            else:
                plt.close()  # Close plot if no label entered
                print("No label entered, skipping...")
        else:
            print("Failed to save captcha image")

    except KeyboardInterrupt:
        print("\nStopping captcha collection...")
        plt.close("all")  # Close all plots before exiting
        break
    except Exception as e:
        print(f"Error: {e}")
        # Try to refresh captcha and continue
        try:
            driver.find_element(By.ID, "refreshChptcha").click()
            time.sleep(1)
        except:
            pass

plt.close("all")  # Ensure all plots are closed
driver.quit()
