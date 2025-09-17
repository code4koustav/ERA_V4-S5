# MNIST CNN — Total Parameter Count Test

## Overview
Dataset: MNIST (28x28 grayscale)
Architecture: 9 Conv layers with transitions, BatchNorm after most convs, Dropout2d throughout, two MaxPool transitions, Global Average Pool (GAP), final Fully Connected layer to 10 classes.
Activation: ReLU
Loss: NLLLoss with log_softmax outputs
Optimizer: SGD (lr=0.01, momentum=0.9) + StepLR(step_size=15, gamma=0.1)
Batch size: 128
Augmentations (train): Random CenterCrop(22) p=0.1, Resize(28,28), RandomRotation(±15°), Normalize

## Model Components
Batch Normalization: Used after conv1, conv2, conv3, conv_t1 (1x1), conv4, conv5, conv_t2 (1x1), conv6.
Dropout: Dropout2d(p=0.05) applied across blocks (8 placements).
GAP / Fully Connected:
GAP: AvgPool2d(7) to reduce to 1×1 per channel.
Fully Connected: Linear(24 → 10) for classification.

## Total Parameter Count
Convolutions (9 layers): 19,140
BatchNorm (8 layers): 288
Fully Connected (24 → 10): 250
Total Trainable Parameters: 19,678

## Training & Test Logs
Below are the key test results per epoch (as captured from the notebook):

Epoch : 1 

Train: Loss=0.2132 Batch_id=468 Accuracy=62.54: 100%|██████████| 469/469 [00:30<00:00, 15.55it/s]
Test: Average loss: 0.1459, Accuracy: 9625/10000 (96.25%)

Epoch : 2 

Train: Loss=0.2153 Batch_id=468 Accuracy=92.69: 100%|██████████| 469/469 [00:29<00:00, 15.74it/s]
Test: Average loss: 0.1151, Accuracy: 9673/10000 (96.73%)

Epoch : 3 

Train: Loss=0.1631 Batch_id=468 Accuracy=94.90: 100%|██████████| 469/469 [00:29<00:00, 15.81it/s]
Test: Average loss: 0.0554, Accuracy: 9844/10000 (98.44%)

Epoch : 4 

Train: Loss=0.0931 Batch_id=468 Accuracy=95.91: 100%|██████████| 469/469 [00:27<00:00, 16.99it/s]
Test: Average loss: 0.0433, Accuracy: 9867/10000 (98.67%)

Epoch : 5 

Train: Loss=0.0811 Batch_id=468 Accuracy=96.44: 100%|██████████| 469/469 [00:25<00:00, 18.07it/s]
Test: Average loss: 0.0481, Accuracy: 9851/10000 (98.51%)

Epoch : 6 

Train: Loss=0.0733 Batch_id=468 Accuracy=96.75: 100%|██████████| 469/469 [00:27<00:00, 16.88it/s]
Test: Average loss: 0.0360, Accuracy: 9894/10000 (98.94%)

Epoch : 7 

Train: Loss=0.0543 Batch_id=468 Accuracy=97.02: 100%|██████████| 469/469 [00:28<00:00, 16.24it/s]
Test: Average loss: 0.0317, Accuracy: 9905/10000 (99.05%)

Epoch : 8 

Train: Loss=0.1696 Batch_id=468 Accuracy=97.07: 100%|██████████| 469/469 [00:25<00:00, 18.07it/s]
Test: Average loss: 0.0334, Accuracy: 9889/10000 (98.89%)

Epoch : 9 

Train: Loss=0.0973 Batch_id=468 Accuracy=97.21: 100%|██████████| 469/469 [00:29<00:00, 16.14it/s]
Test: Average loss: 0.0298, Accuracy: 9909/10000 (99.09%)


Epoch : 10 

Train: Loss=0.0463 Batch_id=468 Accuracy=97.36: 100%|██████████| 469/469 [00:27<00:00, 17.27it/s]
Test: Average loss: 0.0268, Accuracy: 9914/10000 (99.14%)

Epoch : 11 

Train: Loss=0.0509 Batch_id=468 Accuracy=97.48: 100%|██████████| 469/469 [00:28<00:00, 16.75it/s]
Test: Average loss: 0.0285, Accuracy: 9915/10000 (99.15%)

Epoch : 12 

Train: Loss=0.0595 Batch_id=468 Accuracy=97.61: 100%|██████████| 469/469 [00:28<00:00, 16.69it/s]
Test: Average loss: 0.0298, Accuracy: 9907/10000 (99.07%)

Epoch : 13 

Train: Loss=0.0389 Batch_id=468 Accuracy=97.65: 100%|██████████| 469/469 [00:26<00:00, 18.01it/s]
Test: Average loss: 0.0229, Accuracy: 9917/10000 (99.17%)

Epoch : 14 

Train: Loss=0.1121 Batch_id=468 Accuracy=97.74: 100%|██████████| 469/469 [00:26<00:00, 17.87it/s]
Test: Average loss: 0.0219, Accuracy: 9929/10000 (99.29%)

Epoch : 15 

Train: Loss=0.0302 Batch_id=468 Accuracy=97.80: 100%|██████████| 469/469 [00:25<00:00, 18.10it/s]
Test: Average loss: 0.0234, Accuracy: 9924/10000 (99.24%)

Epoch : 16 

Train: Loss=0.1582 Batch_id=468 Accuracy=98.05: 100%|██████████| 469/469 [00:25<00:00, 18.09it/s]
Test: Average loss: 0.0209, Accuracy: 9930/10000 (99.30%)

Epoch : 17 

Train: Loss=0.0290 Batch_id=468 Accuracy=98.18: 100%|██████████| 469/469 [00:26<00:00, 17.56it/s]
Test: Average loss: 0.0205, Accuracy: 9929/10000 (99.29%)

Epoch : 18 

Train: Loss=0.0281 Batch_id=468 Accuracy=98.20: 100%|██████████| 469/469 [00:27<00:00, 17.24it/s]
Test: Average loss: 0.0197, Accuracy: 9936/10000 (99.36%)

Epoch : 19 

Train: Loss=0.0355 Batch_id=468 Accuracy=98.19: 100%|██████████| 469/469 [00:26<00:00, 17.59it/s]
Test: Average loss: 0.0200, Accuracy: 9930/10000 (99.30%)
