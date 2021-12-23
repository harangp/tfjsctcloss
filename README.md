# Tensorflow.JS CTC loss implementation

Tensorflow's JS implementation as of now (2021.12.23) lacks a native implementation of the CTC loss.
There's even a ticket requesting contribution: [1759](https://github.com/tensorflow/tfjs/issues/1759), but the last time it was touched was in October, 2019.
Even if there are some resources available calculating the loss itself from an imput, they are not plugabble into the layered model infrastructure we all love.
For practical purposes, I've decided to dive into the academic papers, and have a shot at it - in the meantime I'd like to deepen my knowledge of TypeScript and Tensorflow.

The goal of this project is to finalize a CTC loss calculator to enable pure Tensorflow.JS implementation to work on either server side (with tfjs-node) and browser side (with tfjs-webgl).

## What will you find here

The whole repository is just two files:
- **ctc.ts** contains the loss calculator, the gradient custom operation, and a helper function for decoding the results.
- **ctc.spec.ts** contains the test, and also some examples how to include it in the models.
Otherwise, I will just document here my findings related to the implementation.

## Usage

Just include ctc.ts in your code, and start to build your model. You can check the ctc.spec.ts for examples, but the general thing you do is something like this:

```js
import * as tf from '@tensorflow/tfjs-core';
import { ctcLossGradient } from './ctc';

// build your model
const model = tf.sequential();
model.add(tf.layers.inputLayer({ name: 'inputLayer', inputShape: [5, 42] }));
// add your model's wizardy here - for the sake of simplicity, this is just a simple dense layer
model.add(tf.layers.dense({ 
  name: 'denseLayer', 
  units: 42, 
  kernelInitializer: 'ones', 
  useBias: false, 
  activation: 'softmax'
}));
model.compile({
    loss: ctcLossGradient,
    optimizer: tf.train.adam(),
    metrics: ['accuracy']
});
model.summary();
```
Then you can use model.fit() as you usually would, with all the batches and epochs, etc.

## My learning process
What you'll get here is a constant progress and the evolution of the codebase. Here's my approach:

- learn about the CTC algorythm, and it's usage
- examine the existing solutions
- lear about Tensorflow.JS' architecture
- develop a naive implementation - doesn't have to be nice, but it should work
- have tests developed that actually run and test the basics and edge cases
- refactor parts of it to be more tensory, and therefore presumably quicker
- if **it's good enough**, donate the whole stuff to the Tensorflow team to be included in the core product

Let's dive in!

### Learn about the problem itself

There are some articles one should read about the usage of the CTC algorythm. They are a good starting point to understand why we need it:

- https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5
- https://distill.pub/2017/ctc/
- https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c

The papers describing the algorythm are here:

- https://www.cs.toronto.edu/~graves/icml_2006.pdf - Graves et. all: Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
- http://bacchiani.net/resume/papers/ASRU2017.pdf - Improving the efficiency of forward-backward algorithm using batched computation in Tensorflow

Lectures:

- https://axon.cs.byu.edu/~martinez/classes/778/Papers/CTC.pdf - Logan Mitchell: Sequence to sequence learning
- https://www.youtube.com/watch?v=c86gfVGcvh4 - Carnegie Mellon University Deep Learning, S18 Lecture 14: Connectionist Temporal Classification (CTC)
- https://www.youtube.com/watch?v=GxtMbmv169o - Carnegie Mellon University Deep Learning, F18 Recitation 8: Connectionist Temporal Classification (CTC)
 

### Existing solutions

The only thing comes close to the native JS implementation is @martiancba's solution that was commented in this issue: https://github.com/tensorflow/tfjs/issues/1759
However, I couldn't wrap my head around some of the implementation's pecularities (namely, calculation of beta variables and input matching). However, it's usage of the tf operators is very advanced, worth checking out.

The Tensorflow Python implementation is definitely worth checking out: [ctc_ops.py](https://github.com/tensorflow/tensorflow/blob/87462bfac761435a46641ff2f10ad0b6e5414a4b/tensorflow/python/ops/ctc_ops.py)

My personal favourite is the Stanford CTC implementation, since it is understandable for folks who are not well versed in Python: https://github.com/amaas/stanford-ctc/blob/master/ctc/ctc_fast.pyx
The code is very easy to read, and you can relate much of it to the original paper.

### Tensorflow architecture

If you want to develop your custom loss calculator, first check this description: https://www.tensorflow.org/js/guide/custom_ops_kernels_gradients Follow the examples, check them out. For me, it helped a lot to examine the logLoss calculation since it's pretty simple.
Also, there are other reading materials available: https://towardsdatascience.com/creating-custom-loss-functions-using-tensorflow-2-96c123d5ce6c

So the key here is to develop a custom gradient - custom operation, which returns a Tensor with the calculated loss, and a custom gradient function which can be called to calculate the gradients. The infrastructure is set up that during the loss calculation, parameters and intermediate calculations can be saved, which can be reused during the gradient calculation.

### Develop a naive implementation

Throughout the implementation you should have two goals:

- Have everything calculated in tensors
- Keep everything you can in the GPU's memory (in our case WebGL)

From my first experience, this won't happen in your first try. Tensors are immutable, accessing values are a pain, and you sometimes just throw in the towel and do things the old fasioned way: get an array representation, do the trick as you are used to, then convert the result back into a tensor and move along. **This is fine for getting things working**, just don't forget, that there is a reason for pressing work to be done the 'Tensorflow way'. More on that later.

CTC is a dynamic algorythm, which means a lot of slicing / concatenation / conditional stuff is happening. It will take time to get there.

### Test

CTC is a tricky alrgorythm, it has it's perks. The main problem is, that it requires lot's of data for the input, generates lot's of outputs, and it is hard to assemble a list of inputs - expected outputs pairs withouth doing tedious calculations. My approaxch is that there are obvious cases:
- matching inputs and labels should return a zero loss and zero gradiens
- random noise inputs should produce "something" other than error
- should handle signle elements and batched elements
- running fit() with 10 epochs we should see a decreasing loss

There is a big **TODO** here: have somebody who has some experience with the Python implementation, and generate some input-output pairs for our tests. Until then, let's extract some of the tests from here: [ctc_loss_op_test.py](https://github.com/tensorflow/tensorflow/blob/ab7c873202574b3cd549a93ccbfd881d659186ca/tensorflow/python/kernel_tests/nn_ops/ctc_loss_op_test.py)

### Refactor cycle

I've mentioned that it is ok to start with JavaScript arrays and stuff, but let me explain why we need to move to tensors. You see, Tensorflow is organized to have the concept of kernel functions - this means the codebase can utilize different implementations to execute the operations depending on what infrastructure is available withouth having the programmer rework everything for the sake of different platforms. Think about it this way: you have two big, multidimensional tensors, that you need to multiply elementwise. You would use something like this:
```js
const result = tensorA.mul(tensorB);
```
If you would need to do it in a pure JavaScript way, with arrays, you would implement the cycles, do the math, and return a new array with the results.
The execution depending on what's available for Tensorflow can be really different:
- **tfjs only** - just like you would do it in pure javascript, but somebody has programmed it for you. Sweet.
- **tfjs-node** - kernel functions run natively on the processor, so you have the full power of your CPU, including the special instruction-sets you might have
- **tfjs-webgl** - kernel functions take advantage of the parallel processing capabilities of your GPU

Every step brings at least a two-fold drop in execution time, so it is essential to have as many things as possible "tenosry". But it's not trivial - some of the operators are (albeit very useful) pretty hard to grasp. That's where the learning process and the cycle kicks in.

## License

As noted in the LICENSE.md file, this work is licensed under Creative Commons BY-NC 4.0
