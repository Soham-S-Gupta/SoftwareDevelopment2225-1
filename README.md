# ASL Alphabet Translator

This project is a webcam-based ASL alphabet translator made for the TSA Software Development event. It recognizes ASL fingerspelling one letter at a time, builds words on screen, stores completed output in a history view, and includes a letter guide and practice area. The project is deliberately focused on alphabet recognition instead of full ASL sentence translation so the final result stays accurate, realistic, and manageable.

## Software Requirements

- Python `3.11.9`
- A webcam
- Windows PowerShell

Python `3.11.9` is the recommended version for this project because it works well with `mediapipe` and the other libraries used here.

## File Layout

```text
SoftwareDevelopment2225-1/
  README.md
  requirements.txt
  app.py
  tsa_work_log.html
  project_runtime_code_bundle.txt
  configs/
    project_config.yaml
  data/
    raw/
      asl_alphabet/
      custom_letters/
  models/
    checkpoints/
      letter_classifier.pt
      letter_classifier_custom.pt
    labels/
      letter_labels.json
  scripts/
    train_letters.py
    capture_custom_letters.py
    capture_all_custom_letters.py
    fine_tune_letters.py
  src/
    asl_translator/
      __init__.py
      config.py
      hand_tracking.py
      letter_model.py
      pipeline.py
      speech.py
      ui.py
```

## How To Run It

1. Open PowerShell in the `asl_translator_project` folder.
2. Create a virtual environment if needed:

```powershell
python -m venv .venv
```

3. Install the required libraries:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. Run the app:

```powershell
.\.venv\Scripts\python.exe app.py
```

Running the app does not retrain the model. It uses the existing checkpoint files in `models/checkpoints/` and the label file in `models/labels/`.

## How To Use It

1. Launch the app and press `Start`.
2. Hold your hand in front of the webcam and sign ASL alphabet letters one at a time.
3. The system tracks the hand, predicts the current letter, and adds stable letters into the current word.
4. Use the `space` sign or lower your hand briefly to finish a word.
5. Use the `del` sign to remove the most recent letter.
6. Watch the `Current word` and `Sentence` sections as the text builds live.
7. Open the history page to review completed words.
8. Use the letter guide on the right side to click a letter and view a sample image.
9. Use the practice box to type a letter or word and step through the matching signs.

## Credits

### Dataset

This project uses the **ASL Alphabet** dataset from Kaggle:  
[ASL Alphabet on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

The dataset was downloaded from Kaggle and used as the base image dataset for alphabet training. The dataset is not being claimed as original work by this project. It is being credited clearly because that is the correct way to acknowledge outside material used in a school software project.

Giving credit does not mean claiming ownership of the dataset. It means identifying the original source and being honest about where the training images came from. For a TSA project, citing the dataset is appropriate and important because the project depends on outside data for model training. The Kaggle page should still be treated as the main source to reference in documentation and presentation materials.

### Libraries Used

- [Python](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [tqdm](https://tqdm.github.io/)
- [Pillow](https://python-pillow.org/)
- [pyttsx3](https://pyttsx3.readthedocs.io/)

These libraries are credited because they were used to build the project's computer vision, machine learning, interface, configuration, image handling, and speech features.

## What Each Library Was Used For

- `Python` was the main programming language used for the whole project.
- `NumPy` was used for array math and numeric operations in image and landmark processing.
- `OpenCV` was used for webcam input, frame handling, drawing boxes, drawing the interface, and showing the app window.
- `MediaPipe` was used for hand detection and hand landmark tracking.
- `PyTorch` was used to load the trained model and run live predictions.
- `Torchvision` was used for image transforms such as resizing, tensor conversion, and normalization.
- `timm` was used to create the EfficientNet model architecture used by the classifier.
- `PyYAML` was used to read the project configuration file.
- `pandas` was used in the training workflow for data handling support.
- `scikit-learn` was used in the training workflow for machine learning support and evaluation utilities.
- `tqdm` was used to show progress bars in the training scripts.
- `Pillow` was used as part of the image preprocessing pipeline.
- `pyttsx3` was used for the text-to-speech feature that speaks completed words.

## Notes

- The app is designed around ASL alphabet recognition, not full ASL sentence translation.
- The most important runtime files are `models/checkpoints/letter_classifier.pt`, `models/checkpoints/letter_classifier_custom.pt`, and `models/labels/letter_labels.json`.
- If those trained files are removed, the project would need retraining before it could run again.
