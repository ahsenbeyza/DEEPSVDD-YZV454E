# One-Class Classification for Anomaly Detection on CIFAR-10

## Overview
This repository contains the implementation of a one-class classification project for anomaly detection on the CIFAR-10 dataset, as part of the BLG 454E course at Istanbul Technical University. The project focuses on identifying anomalies using airplane images as the normal class and other classes (e.g., cars, dogs) as anomalies. We implemented **Deep Support Vector Data Description (Deep SVDD)** with a ResNet18 encoder and compared it against classical models: **One-Class SVM** and **Isolation Forest**.

The project includes a Jupyter notebook with the implementation, a project report, a presentation, and a representation video summarizing the work.

## Team
- **Ahsen Beyza Ozkul** (ozkula21@itu.edu.tr)
  - Data preprocessing, literature survey, Deep SVDD implementation, experiments, and analysis.
- **Tesnime Jemmazi** (jemmazi22@itu.edu.tr)
  - Baseline model implementation (One-Class SVM, Isolation Forest), report writing, Deep SVDD implementation, experiments, and analysis.

## Project Structure
- **`implementation.ipynb`**: Jupyter notebook containing the full implementation of Deep SVDD, baseline models, data preprocessing, training, evaluation, and visualization.
- **`report.pdf`**: Detailed project report covering the problem statement, hypothesis, methodology, results, discussion, and future work.
- **`presentation.pdf`**: Slides summarizing the project motivation, methodology, dataset, results, and discussion.
- **`representation_video.mp4`**: Video presentation explaining the project, methodology, and findings (link or file to be added).
- **`README.md`**: This file, providing an overview and instructions for the repository.

## Requirements
To run the code in `implementation.ipynb`, install the required Python packages:

```bash
pip install torch torchvision numpy matplotlib sklearn seaborn tqdm
```

- Python version: 3.x
- GPU recommended for faster training (CUDA-enabled PyTorch required if using GPU).

## Dataset
The project uses the **CIFAR-10** dataset, which is automatically downloaded via `torchvision` in the notebook. It consists of 60,000 32x32 color images across 10 classes, with 5,000 airplane images used for training (normal class), 1,000 for testing, and 9,000 images from other classes as anomalies.

## Methodology
- **Deep SVDD**: Uses a pretrained ResNet18 encoder (modified to output a 128-dimensional latent space) to map normal data near a center in feature space. Trained for 100 epochs with Adam optimizer (learning rate: 0.001, weight decay: 1e-6).
- **One-Class SVM**: Uses an RBF kernel with pre-extracted ResNet18 features, standardized with `StandardScaler`.
- **Isolation Forest**: Configured with a contamination parameter of 0.1, using the same pre-extracted features.
- **Evaluation**: Models are compared using AUC-ROC, precision, and recall on a test set of 1,000 normal and 9,000 anomaly samples.

## Results
- **One-Class SVM**: AUC-ROC = 0.7145, Precision = 0.93, Recall = 0.81
- **Isolation Forest**: AUC-ROC = 0.7057, Precision = 0.93, Recall = 0.73
- **Deep SVDD**: AUC-ROC = 0.4715, Precision = 0.61, Recall = 0.10

Contrary to our hypothesis, Deep SVDD underperformed compared to classical models, likely due to insufficient hyperparameter tuning or overfitting to pretrained ResNet18 weights.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ahsenozkul/DEEPSVDD-YZV454E.git
   ```
2. Install dependencies (see Requirements).
3. Open `implementation.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the notebook cells sequentially. The code will:
   - Download CIFAR-10 dataset.
   - Train Deep SVDD and baseline models.
   - Evaluate performance and generate visualizations.
   - Save the trained Deep SVDD model as `deep_svdd_model.pth`.
5. View `report.pdf` and `presentation.pdf` for detailed documentation.
6. Watch `representation_video.mp4` for a summary of the project (ensure the video file is included or linked).

## Future Work
- Tune Deep SVDD hyperparameters (e.g., learning rate, $\nu$, regularization).
- Fine-tune or train ResNet18 from scratch for better feature extraction.
- Explore multi-center SVDD or semi-supervised approaches with synthetic outliers.
- Test more expressive models or anomaly-aware loss functions.

## References
1. Ruff, L., et al. (2018). "Deep One-Class Classification." ICML.
2. Schölkopf, B., et al. (2001). "Estimating the Support of a High-Dimensional Distribution." Neural Computation.
3. Liu, F. T., et al. (2008). "Isolation Forest." ICDM.
4. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details (if applicable).

## Contact
For questions or feedback, contact:
- Ahsen Beyza Ozkul: ozkula21@itu.edu.tr
- Tesnime Jemmazi: jemmazi22@itu.edu.tr
