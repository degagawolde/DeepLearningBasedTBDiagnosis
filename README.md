# DeepLearningBasedTBDiagnosis

CXR Modality-Specific Pretraining
In the first step, we retrained a selection of MobileNet-V2 and EfficientNet-B0

During this training step, the combined CXR collection, including RSNA CXR, pediatric
pneumonia CXR, and Indiana CXR datasets, is split at the patient-level into 80%
for training and 20% for testing. With a fixed seed value, we allocated 10% of the training
data toward model validation. The models are optimized using stochastic gradient descent
(SGD) algorithm to minimize the categorical cross-entropy loss toward this classification
task. Callbacks are used to check model performance, and the model checkpoints are stored
after each epoch.
