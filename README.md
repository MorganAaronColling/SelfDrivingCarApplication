Evaluating the performance of end to end deep learning
models in a custom car driving simulator:

This report documents the creation of a custom car driving simulator in the Godot game engine and
evaluates the performance of convolutional neural networks (CNNs) within the simulator. The CNN
models were trained using an image of the environment in front of the car and the corresponding
steering angle; images and angles used were gathered from the simulator. The model performance
was evaluated by calculating an autonomy value which represented the average time a model could
control the simulated car without needing to be corrected by the driver. This report found that
trained CNN models can perform tasks such as lane detection, road tracking and obstacle detection
using a small dataset. It was also found that increasing the dataset size resulted in an increased
performance on those tasks. This was evidenced by the increased performance of Model V2
compared to Model V1 (0.0854 interferences per second compared to 0.1135, respectively).
