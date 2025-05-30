# Brain Tumor Segmentation from DICOM Files Using U-Net

## Overview

This project is a deep learning pipeline for brain tumor segmentation using DICOM images. It includes the following key components:

* Data loading and preprocessing from DICOM files
* Augmentation and batch generation
* U-Net model for segmentation
* Training and validation with appropriate metrics
* Inference on unseen data

---

## Requirements

* Python
* TensorFlow
* OpenCV
* NumPy
* Pandas
* Matplotlib
* seaborn
* albumentations
* pydicom
* tqdm

---

## Directory Structure

```
/your/project/path/
│
├── images/         # DICOM image files
├── masks/          # DICOM mask files
├── output/         # Save predicted masks
└── best_brain_tumor_model.h5  # Saved model
```

---

## Modules

### `DICOMDataLoader`

Handles loading and preprocessing of DICOM files (images and masks):

* Normalizes pixel values
* Resizes to target shape
* Matches images and masks by filename

### `DataGenerator`

Keras `Sequence` generator to provide augmented training batches:

* Augmentations (flip, rotate, contrast, blur)
* Normalization and reshaping

### `build_unet()`

Constructs a U-Net architecture with:

* Encoder-decoder structure
* Dropout for regularization
* Conv2DTranspose for upsampling

### Custom Losses and Metrics

* `dice_coefficient()`: Measures overlap between predicted and true masks.
* `dice_loss()`: 1 - Dice coefficient.
* `combined_loss()`: Binary crossentropy + Dice loss.
* `iou_score()`: Measures Intersection over Union.

### `visualize_results()`

Plots the original image, true mask, and predicted mask side-by-side.

---

## Training Pipeline

### `main()`

1. **Prepare Data**

   * Load and normalize images & masks
   * Train/validation split

2. **Train Model**

   * Model is compiled with custom loss and metrics
   * Training includes callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)

3. **Evaluate Model**

   * Final evaluation on validation data
   * Metrics plotted over training epochs

4. **Visualize Predictions**

   * Predictions on validation images

---

## Inference Pipeline

### `inference_pipeline(model_path, image_dir, output_dir)`

* Loads trained model
* Scans and preprocesses new DICOM images
* Predicts tumor masks
* Saves and visualizes results

---

## Execution

### Train the Model

```python
if __name__ == "__main__":
    main()
```

### Inference Example

```python
predictions = inference_pipeline(
    model_path='best_brain_tumor_model.h5',
    image_dir='/path/to/new/images',
    output_dir='/path/to/save/results'
)
```

---

## Notes

* Set `image_dir` and `mask_dir` properly before training.
* Dataset must have corresponding masks for segmentation training.
* The model expects grayscale (single-channel) 256x256 images.
* Ensure sufficient samples to avoid underfitting.

---

## License

This project is for educational and research purposes only. All rights reserved.
