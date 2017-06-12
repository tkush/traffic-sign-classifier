**Note**: For additional information, please open the Python notebook in this repo.

# Traffic sign classification using Convolution Nets
This document describes the basic thoughts and workflow behind the code for classifying images of traffic signs. Most of the iterations described here are based on online reading, paper reviews and blogs.

#### Basic Idea
Create a convolutional neural network that trains on a set of input images and their labels to be able to predict the correct classification for a “new” image that it has not trained on. 

#### Datasets
Three sets of data are provided for this project: training, test and validation
**Training**: The network learns on this data
**Validation**: The network is tuned based on prediction accuracies on this data
**Test**: The network classification capability is judged based on this data. This data is used only once at the very end.

#### Pre-processing data
Use of different color spaces is attempted as a pre-processing step. HSV was suggested as an alternative to RGB since it is less sensitive to shadows. However, in my experiments, RGB performed the best

The image was mean shifted as this was something that boosted the validation accuracy by a few percentage points for HSV and Grayscale. For RGB, however, histogram equalization proved to be more effective in increasing the validation accuracy, so this was retained. 

#### Data augmentation
In the early iterations of designing the conv net, I observed that if I increased the size of the conv net, the data would need to increase too. Otherwise, I was running into severe overfitting. To counter this, I decided to augment the data by flipping and rotating the original images till I had more than 6 times the original data. This helped bring down overfitting when I increased the size of my convnet

#### Balancing the dataset 
After the conv net architecture was firm, I realized that the input, augmented data was not very well balanced. There were classes with fewer than 2000 instances in them. To boost validation accuracy some more, the data was balanced after being augmented. Balancing was done by adding new data to classes with few instances by rotating the original images a few degrees.

### KushNet architecture
Starting off from the LeNet architecture, KushNet was devised by:
1.	Increasing the number of convolution layers. This helped create more feature maps and helps boost the validation accuracy
2.	Depth of 32 for the hidden layers: Increasing the depth of the network also proved to be beneficial to the overall accuracy
3.	Using ELU instead of RELU for activation of the fully connected layers helped increase the convergence speed

Increasing the network size led to severe overfitting, but this was overcome by using the following techniques

#### Preventing overfitting
The following techniques are used to prevent overfitting:
1.	Data augmentation
2.	Dropout: All but the last fully connected layers have dropout with a keep probability of 0.5. Dropout helps the network learn alternate ways of classifying the same image. Dropout helped reduce overfitting considerably
3.	L2 weight normalization: As the size of the network increased, the validation accuracy got stuck at a certain value and would not change. Adding L2 norms of the weights to the loss operation helped in reducing the absolute values of the weights which reduced saturation and increased the validation accuracy significantly.  

#### Training the model
To train the model, I experiment first with batch size. As the batch size was reduced, I found that the validation accuracy became higher but also fluctuated more. A batch size of 128 was chosen for speed and memory considerations. 

The model is trained for 20 epochs. An **adaptive learning rate** is used that is halved each time the validation accuracy is reduced from one iteration to the next. The training is stopped if the learning rate becomes too small. 

**TensorFlow GPU** libraries have been leveraged in this project. I also made use of the checkpoint feature such that an iteration where the validation accuracy goes down is thrown away and the previous iteration is restored. The learning rate is reduced and the training is continued. This adaptive learning rate scheme helped bump up validation accuracy by about a percent.

The training was done on my computer that has a 2GB GPU. To conserve the GPU memory, I could split ops between my CPU and GPU to achieve the best tradeoff between memory and performance.

#### Solution Approach
The basic solution approach was to monitor the validation accuracy while training is going on. If the validation accuracy is decreasing or stays the same, I would try tuning the hyperparameters, decrease the size of my convnet or reduce dropout. 

If the validation accuracy is considerably lower than the training accuracy, this indicates overfit. This happened several times during this project. The techniques described above were used for this.

#### Testing the model on new images
The model is also tested on 6 images downloaded from the web – one of which is not included as a class in the training data set. The network does well, classifying 5 of the 6 images correctly. The last image is classified as a “priority road” which is a yellow diamond sign, like the merge sign the network is presented.

The network seems to be sure while classifying all the test images except the stop sign. Here, the network is confused between speed limit 60 and the stop sign. It is not clear why this is so immediately, although it may be that the “O” in the STOP sign is being interpreted as the zero in the 60 sign. Another area that could be investigated is the number of images for the 60 vs STOP sign in the training data, i.e. is the STOP sign under-represented w.r.t. the 60 sign?
