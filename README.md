# CRNN_CAPTCHA

CRNN-based solver for 4-digit numeric CAPTCHAs from https://service.taipower.com.tw/hvcs. The model couples a ResNet-50 visual backbone with bi-directional LSTMs and CTC decoding, with helper scripts to build a labeled dataset and run automated solving in a browser session.

## Project Structure
- `generate_captcha.py`: Collect and label CAPTCHA images with Selenium; saves PNGs to `data/unprocessed/` and appends labels to `data/captcha_labels.csv`.
- `CRNN/model.py`: CRNN architecture using pretrained ResNet-50 features → two bidirectional LSTM layers → linear classifier (11 classes: blank + digits 0–9) trained with CTC loss.
- `CRNN/predict.py`: Loads the trained checkpoint, opens the Taipower site in Chrome, refreshes CAPTCHAs until the model is confident, and prints the predicted 4-digit code.
- `data/captcha_train.csv`, `data/captcha_val.csv`, `data/captcha_test.csv`: Split label files (≈478 train / 160 val / 160 test rows plus headers) pointing at the collected PNGs.
- `CRNN/checkpoints/best_model.pth`: Saved weights used by the predictor.

## How Data Collection Works
1. Install Chrome and chromedriver on your PATH.
2. Run `python generate_captcha.py` from the repo root.
3. The script opens the Taipower page, downloads the current CAPTCHA, shows it with matplotlib, and prompts for the 4-digit label.
4. Each labeled image is saved as `data/unprocessed/captcha_###.png` and appended to `data/captcha_labels.csv`; the CAPTCHA is refreshed and the loop repeats until you stop with Ctrl+C.

## Model Details
- Backbone: torchvision `resnet50` with default ImageNet weights, truncated before the final pooling/classifier layers.
- Sequence head: two-layer bidirectional LSTM (`hidden_size=256`) feeding a linear layer to predict 11 classes per timestep.
- Loss/decoding: CTC with blank index 0; greedy decoding drops blanks, subtracts 1 to map class IDs to digits 0–9, and keeps the first four symbols for the CAPTCHA string.
- Input preprocessing: resize to `128×32`, `ToTensor`, and ImageNet mean/std normalization to match the pretrained backbone.

## Running the Solver
1. Ensure `CRNN/checkpoints/best_model.pth` exists (provided).
2. From the `CRNN` directory, run `python predict.py`.
3. The script loads the checkpoint on CPU/GPU, opens Chrome to the CAPTCHA page, captures each image, predicts it, and keeps refreshing until the average character confidence exceeds ~0.95. Successful predictions are printed to stdout.

## Requirements
- Python deps: `torch`, `torchvision`, `selenium`, `Pillow`, `numpy`, `requests`, `matplotlib`.
- Chrome + chromedriver available to Selenium.
