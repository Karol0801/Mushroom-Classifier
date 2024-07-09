# Mushtroom Genus Classifier

## Overview
The project aimed to create a mushroom classifier, which identifies mushrooms from one of 12 classes based on the provided image. The dataset used is available [here].

Before building the model, data augmentation was performed. The model is based on the ResNet101 architecture. We conducted five experiments to compare different optimizers and regularization techniques. Each model was trained for 50 epochs, and evaluation was done using standard metrics: Accuracy, Precision, Recall, and F1-score.

### The experiments conducted:
- Adam
- Adam + Dropout + L2
- SGD + L2
- SGDM + L2 + Early Stopping (5)
- SGDM + L2 + Early Stopping (10)

We also used the Weights and Biases (wandb) library in the project. This tool helped in tracking and visualizing machine learning experiments. It was used to log the training progress, model performance metrics, and hyperparameters, facilitating experiment tracking and model comparison.
Thanks to live training progress tracking, we decided to interrupt one of the experiments because the regularization applied was too restrictive. 

## Results

<div style="text-align:center">
  <img src="README_files/train_accuracy.png" style="display:block; margin: 0 auto;">
</div>

We observe that as training progressed, the accuracy either consistently increased or remained relatively stable. The highest accuracy in the training set was achieved by the model using only the Adam optimizer (96.5%). Slightly lower results were obtained for models using SGDM (90.6%), while the SGD model showed slower learning and lower effectiveness (82.1%). Additionally, we noticed that after 22 epochs, the model utilizing the Adam optimizer, dropout, and L2 regularization still maintained an accuracy of 42.6%. Therefore, we decided to stop its training.

<div style="text-align:center">
  <img src="README_files/test_accuracy.png" style="display:block; margin: 0 auto;">
</div>

The accuracy on the test dataset shows some differences. Models using SGDM clearly perform the best, achieving 92.8% accuracy. The model with the Adam optimizer is less stable compared to the others, with its accuracy decreasing towards the end to 73.5%, indicating potential overfitting. This overfitting is evidenced by the correlation between the training and test accuracies. The model with SGD achieves a similar accuracy of 86.4% on the test dataset (excluding the final epoch drop in the Adam model), but its progress is more stable.

<div style="text-align:center">
  <img src="README_files/precision.png" style="display:block; margin: 0 auto;">
</div>
<div style="text-align:center">
  <img src="README_files/recall.png" style="display:block; margin: 0 auto;">
</div>
<div style="text-align:center">
  <img src="README_files/f1_score.png" style="display:block; margin: 0 auto;">
</div>

These metrics also indicate that SGDM outperforms standard SGD. Implementing early stopping with a threshold of 10 epochs resulted in approximately a 1% improvement across these metrics. While seemingly small, such marginal gains can be crucial in tasks where even slight improvements in accuracy or precision are significant.


Therefore, the recommendation is to avoid terminating training prematurely. However, it’s also cautioned against extending training to the maximum epochs, which may lead to overfitting. This approach not only risks model performance but also consumes valuable time, particularly with complex models that require substantial training durations


## Conclusions

 In our study, models based on SGD showed superior performance. The SGDM model, which incorporates momentum into SGD by accumulating gradients from previous steps, effectively accelerated convergence and yielded better outcomes.
 Conversely, models utilizing the Adam optimizer performed less favorably in our comparison. The non-regularized model exhibited signs of overfitting, which was anticipated given the size and complexity of our dataset. However,
 the introduction of L2 regularization and Dropout noticeably limited training
 effectiveness, likely due to excessively high regularization parameter values. We
 refrained from further fine-tuning as the SGDM model consistently delivered
 satisfactory results.
 We hypothesize that an Adam-based model with appropriately optimized
 parameters could achieve satisfactory but marginally inferior results compared
 to SGD models. The preference for SGD and SGDM stems from their indepen
dence from adaptive learning rates, potentially facilitating better generalization
 on datasets with stable distributions. In our case, our dataset exhibits stability,
 supported by the presence of about 1000 representative images per class.

