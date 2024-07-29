### (GardenPy is current a work-in-progress)

(todo: logo)

# GardenPy #

**GardenPy** is a machine learning library coded from **scratch** (**NumPy** and **Pandas**).
Other than **GardenPy**, this repository includes demos utilizing the machine learning library, including **MNIST**'s handwritten numbers.
While **GardenPy** isn't faster than **PyTorch** and doesn't have GPU utilization, building a model from **GardenPy** follows the mathematical format of a model.

----

## Table of Contents
- [Features](#features)
- [Usage](#usage)
- [Documentation](#documentation)
- [License](#license)
- [Contact](#contact)
- [Status](#status)

----

## Features
- **Automatic differentiation** with vectors, matrices, and tensors.
- **Machine learning algorithms**, including initialization algorithms, activation functions, loss/cost functions, and optimization techniques.
- **DataLoaders** for loading datasets.
- **Visualization functions** for results.
- **Pre-built models** utilizing the **GardenPy** library, including a dense/fully-connected neural network (DNN or FNN) and a convolutional neural network (CNN).
- **Pre-built models** utilizing **PyTorch** following the same format as the models utilizing the **GardenPy** library for comparisons.
- **Pre-written training scripts** for certain datasets, including **MNIST**'s hand-drawn number dataset.
- **Pre-trained models for certain** datasets, including **MNIST**'s hand-drawn number dataset.
- **Drawing scripts** utilizing trained models for recognizing user-drawn numbers utilizing **PyGame**.

----

## Usage

### Demo Usage
This repository includes pre-trained models and pre-written training scripts.

#### Training
To train a model, simply run the *train.py* file included under the folder for each type of model under each dataset.
Data will automatically generate after running the *train.py* script if the data doesn't exist.

(todo: image)

#### Drawing

To run a drawing script, simply run the *draw.py* script located under each dataset's training folder.
It should be noted that not every dataset has a drawing script.

(todo: image)

To change what type of model is used in the drawing script, choose a model from the list of models.
```python
# model name
model_type = 'dnn'
# possible model names
model_types = ['cnn', 'cnn_torch', 'dnn', 'dnn_torch']
```

### Basic Usage
Though **GardenPy**'s pre-built models are simple to use, they allow for significant customization.
Pre-built models include default settings that make utilizing them fast and simple.

Utilizing default settings, a model could be trained in only a few lines of code.

```python
from gardenpy import DNN, DataLoaders

##########

# load dataset
dataset = DataLoaders(
    root=...,
    values=...,
    labels=...
)

##########

# load model
model = DNN()
# initialize model
model.initialize()
# instantiate hyperparameters
model.hyperparameters()
# fit model
model.fit(data=dataset)
# save the trained model (optional)
model.save(location=...)

```

However, every aspect can still be edited in these pre-built models.

```python
from gardenpy import DNN, DataLoaders

##########

# load dataset
dataset = DataLoaders(
    root=...,
    values=...,
    labels=...,
    batch_size=40,
    shuffle=True,
    save_to_memory=True
    
)

##########

# load model
model = DNN(status_bars=True)

# initialize model
model.initialize(
    hidden_layers=[100, 20],
    thetas=[
        # layer 1 initializer
        {'weights': {'algorithm': 'xavier', 'gain': 1.0},
         'biases': {'algorithm': 'zeros'}},
        # layer 2 initializer
        {'weights': {'algorithm': 'xavier', 'gain': 1.0},
         'biases': {'algorithm': 'zeros'}},
        # layer 3 initializer
        {'weights': {'algorithm': 'xavier', 'gain': 1.0},
         'biases': {'algorithm': 'zeros'}}
    ],
    activators=[
        # layer 1 activator
        {'algorithm': 'lrelu', 'beta': 0.1},
        # layer 2 activator
        {'algorithm': 'lrelu', 'beta': 0.1},
        # layer 3 activator
        {'algorithm': 'softmax'}
    ]
)

# instantiate hyperparameters
model.hyperparameters(
    loss={'algorithm': 'centropy'},  # loss function
    solver={
        'algorithm': 'adam',  # optimization algorithm
        'gamma': 1e-2,  # learning rate
        'lambda_d': 0.01,  # L2 term
        'beta1': 0.9,  # beta1 value
        'beta2': 0.999,  # beta2 value
        'epsilon': 1e-10,  # epsilon value
        'ams': False  # AMS variant
    }
)

# fit model
model.fit(
    data=dataset,
    parameters={'epochs': 100_000, 'eval': True, 'eval_rate': 1_000}
)

# print general results
model.results(
    
)

# save the trained model (optional)
model.save(location=...)

```

These pre-built models will take key-word arguments instead of dictionaries if desired.

Refer to [Documentation](#documentation) for in-depth documentation for using **GardenPy**'s pre-built models.

### Advanced Usage
**GardenPy**'s machine-learning library allows building models that, when written, read like mathematical notation for that model.

Here is a comparison between a simple **GardenPy** model and its mathematical notation for a simple model that switches a 1 to a 0.

#### Image
(todo: image)
#### Code
```python
from gardenpy import (
    Tensor,
    nabla,
    chain,
    Initializers,
    Activators,
    Losses,
    Optimizers
)

##########

epochs = 1000

##########

# initializer algorithms
W_init = Initializers(algorithm='gaussian')
B_init = Initializers(algorithm='zeros')

# activation algorithms
act1 = Activators(algorithm='sigmoid')
act2 = Activators(algorithm='sigmoid')

# loss algorithm
loss = Losses(algorithm='ssr')

# optimization algorithm
optim = Optimizers(algorithm='sgd')

##########

# data
X = Tensor([
    [[0, 0]],
    [[0, 1]],
    [[1, 0]],
    [[1, 1]]
])
Y = Tensor([
    [[1, 1]],
    [[1, 0]],
    [[0, 1]],
    [[0, 0]]
])

# weights
W1 = W_init.initialize(2, 3)
W2 = W_init.initialize(3, 2)

# biases
B1 = B_init.initialize(1, 3)
B2 = B_init.initialize(1, 2)

# activation & loss functions
g_1 = act1.activate
g_2 = act2.activate
j = loss.loss

##########

for epoch in range(epochs):
    # forward pass
    X = X
    a2 = g_1(X @ W1 + B1)
    Yhat = g_2(a2 @ W2 + B2)

    L = j(Yhat, Y)

    # gradient calculation

    grad_Yhat = nabla(L, Yhat)

    grad_B2 = chain(grad_Yhat, nabla(Yhat, B2))
    grad_W2 = chain(grad_Yhat, nabla(Yhat, W2))
    grad_a2 = chain(grad_Yhat, nabla(Yhat, a2))

    grad_B1 = chain(grad_a2, nabla(a2, B1))
    grad_W1 = chain(grad_a2, nabla(a2, W1))

    # optimization

    W2 = optim.optimize(W2, grad_W2)
    B2 = optim.optimize(B2, grad_B2)
    W1 = optim.optimize(W1, grad_W1)
    B1 = optim.optimize(B1, grad_B1)

```
Refer to [Documentation](#documentation) for in-depth documentation for using **GardenPy**'s machine learning library.

----

## Documentation
Detailed documentation for **GardenPy** is available at (todo: website).

----

## License
**GardenPy** is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html). See [License](`LICENSE`) for more.

----

## Contact
For questions, recommendations, or feedback on **GardenPy**, contact:
- Christian SW Host-Madsen (chost-madsen25@punahou.edu)
- Mason YY Morales (mmorales25@punahou.edu)
- Isaac P Verbrugge (isaacverbrugge@gmail.com)
- Derek Yee (dyee25@punahou.edu)

Report any issues to [the issues page](https://github.com/personontheinternet1234/NeuralNet/issues).

----

## Status
This repository only updates after significant progress has been made.
See [the working repository](https://github.com/personontheinternet1234/NeuralNet) for **GardenPy** to check the current state of **GardenPy**.
