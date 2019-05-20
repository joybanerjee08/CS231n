# Understanding Convolutional Neural Networks

This repository is an archive of the course [CS231n: Convolutional
Neural Networks for Visual Recognition](http://cs231n.stanford.edu)
(Winter 2016). If you’re even vaguely interested in this topic, you
should probably take this class. It is outstanding.

To use this repository, [make a fork of
it](https://help.github.com/articles/fork-a-repo/) and then
tick off the items in the following syllabus as you complete
them. (You can tick off items by replacing `[ ]` with `[x]` in
`README.md`.)

Happy learning!

[Harish Narayanan](https://harishnarayanan.org/), 2017

## Course Syllabus

- [x] **Lecture 1:** Intro to computer vision, historical context
   - [x] [Video](https://youtu.be/NfnWJUyUJYU)
   - [x] [Slides](slides/lecture1.pdf)
- [x] **Lecture 2:** Image classification and the data-driven
      approach; k-nearest neighbors; Linear classification I
   - [x] [Video](https://youtu.be/8inugqHkfvE)
   - [x] [Slides](slides/lecture2.pdf)
   - [x] [Python/NumPy tutorial](notes/python-numpy-tutorial.pdf)
   - [x] [Image classification notes](notes/image-classification.pdf)
   - [x] [Linear classification notes](notes/linear-classification.pdf)
- [x] **Lecture 3:** Linear classification II; Higher-level
      representations, image features; Optimization, stochastic
      gradient descent
   - [x] [Video](https://youtu.be/qlLChbHhbg4)
   - [x] [Slides](slides/lecture3.pdf)
   - [x] [Linear classification notes](notes/linear-classification.pdf)
   - [x] [Optimization notes](notes/optimization.pdf)
- [x] **Lecture 4:** Backpropagation; Introduction to neural networks
   - [x] [Video](https://youtu.be/i94OvYb6noo)
   - [x] [Slides](slides/lecture4.pdf)
   - [x] [Backprop notes](notes/backprop.pdf)
   - [x] Related references
      - [x] [Efficient Backprop](papers/efficient-backprop.pdf) -- 3/44
      - [x] [Automatic differentiation survey](papers/automatic-differentiation.pdf)
      - [x] [Calculus on Computational Graphs](papers/backprop-calculus.pdf)
      - [x] [Backpropagation Algorithm](papers/backprop-algorithm.pdf)
      - [x] [Learning: Neural Nets, Back Propagation](https://youtu.be/q0pm3BrIUFo)
- [x] **[Assignment 1](assignments/assignment1/assignment1.pdf)**
   - [x] k-Nearest Neighbor classifier
   - [x] Training a Support Vector Machine
   - [x] Implement a Softmax classifier
   - [x] Two-Layer Neural Network
   - [x] Higher Level Representations: Image Features
   - [x] Cool Bonus: Do something extra!
- [x] **Lecture 5:** Training Neural Networks Part 1; Activation
      functions, weight initialization, gradient flow, batch
      normalization; Babysitting the learning process, hyperparameter
      optimization
   - [x] [Video](https://youtu.be/gYpoJMlgyXA)
   - [x] [Slides](slides/lecture5.pdf)
   - [x] [Neural Nets notes 1](notes/neural-nets-1.pdf)
   - [x] [Neural Nets notes 2](notes/neural-nets-2.pdf)
   - [x] [Neural Nets notes 3](notes/neural-nets-3.pdf)
   - [x] Related references
      - [x] [Tips/Tricks 1](papers/sgd-tricks.pdf)
      - [x] [Tips/Tricks 2](papers/efficient-backprop.pdf)
      - [x] [Tips/Tricks 3](papers/practical-sgd.pdf)
      - [x] [Deep learning review article](papers/deep-review.pdf)
- [x] **Lecture 6:** Training Neural Networks Part 2: parameter
      updates, ensembles, dropout; Convolutional Neural Networks:
      intro
   - [x] [Video](https://youtu.be/hd_KFJ5ktUc)
   - [x] [Slides](slides/lecture6.pdf)
   - [x] [Neural Nets notes 3](notes/neural-nets-3.pdf)
- [x] **Lecture 7:** Convolutional Neural Networks: architectures,
      convolution / pooling layers; Case study of ImageNet challenge
      winning ConvNets
   - [x] [Video](https://youtu.be/LxfUGhug-iQ)
   - [x] [Slides](slides/lecture7.pdf)
   - [x] [ConvNet notes](notes/conv-nets.pdf)
- [x] **Lecture 8:** ConvNets for spatial localization; Object
      detection
   - [x] [Video](https://youtu.be/GxZrEKZfW2o)
   - [x] [Slides](slides/lecture8.pdf)
- [x] **Lecture 9:** Understanding and visualizing Convolutional
      Neural Networks; Backprop into image: Visualizations, deep
      dream, artistic style transfer; Adversarial fooling examples
   - [x] [Video](https://youtu.be/ta5fdaqDT3M)
   - [x] [Slides](slides/lecture9.pdf)
- [x] **[Assignment 2](assignments/assignment2/assignment2.pdf)**
   - [x] Fully-connected Neural Network
   - [x] Batch Normalization
   - [x] Dropout
   - [x] ConvNet on CIFAR-10
   - [x] Do something extra!
- [ ] **Lecture 10:** Recurrent Neural Networks (RNN), Long Short Term
       Memory (LSTM); RNN language models; Image captioning
   - [ ] [Video](https://youtu.be/yCC09vCHzF8)
   - [ ] [Slides](slides/lecture10.pdf)
   - [ ] Related references
      - [ ] [Recurrent neural networks](papers/rnn.html)
      - [ ] [Min Char RNN](https://gist.github.com/karpathy/d4dee566867f8291f086)
      - [ ] [Char RNN](https://github.com/karpathy/char-rnn)
      - [ ] [NeuralTalk2](https://github.com/karpathy/neuraltalk2)
- [ ] **Lecture 11:** Training ConvNets in practice; Data
      augmentation, transfer learning; Distributed training, CPU/GPU
      bottlenecks; Efficient convolutions
   - [ ] [Video](https://youtu.be/pA4BsUK3oP4)
   - [ ] [Slides](slides/lecture11.pdf)
- [ ] **Lecture 12:** Overview of Caffe/Torch/Theano/TensorFlow
   - [ ] [Video](https://youtu.be/Vf_-OkqbwPo)
   - [ ] [Slides](slides/lecture12.pdf)
- [ ] **[Assignment 3](assignments/assignment3/assignment3.pdf)**
   - [ ] Image Captioning with Vanilla RNNs
   - [ ] Image Captioning with LSTMs
   - [ ] Image Gradients: Saliency maps and Fooling Images
   - [ ] Image Generation: Classes, Inversion, DeepDream
   - [ ] Do something extra!
- [ ] **Lecture 13:** Segmentation; Soft attention models; Spatial
      transformer networks
   - [ ] [Video](https://youtu.be/ByjaPdWXKJ4)
   - [ ] [Slides](slides/lecture13.pdf)
- [ ] **Lecture 14:** ConvNets for videos; Unsupervised learning
   - [ ] [Video](https://youtu.be/ekyBklxwQMU)
   - [ ] [Slides](slides/lecture14.pdf)
- [ ] **Invited Lecture:** A sampling of deep learning at Google
   - [ ] [Video](https://youtu.be/T7YkPWpwFD4)
- [ ] **Lecture 15:** Conclusions
   - [ ] [Slides](slides/lecture15.pdf)
