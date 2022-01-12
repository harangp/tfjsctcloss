# Tensorflow.JS CTC loss implementation

Tensorflow's JS implementation as of now (2021.12.23) lacks a native implementation of the CTC loss.
There's even a ticket requesting contribution: [1759](https://github.com/tensorflow/tfjs/issues/1759), but the last time it was touched was in October, 2019.
Even if there are some resources available calculating the loss itself from an imput, they are not plugabble into the layered model infrastructure we all love.
For practical purposes, I've decided to dive into the academic papers, and have a shot at it - in the meantime I'd like to deepen my knowledge of TypeScript and Tensorflow.

The goal of this project is to finalize a CTC loss calculator to enable pure Tensorflow.JS implementation to work on either server side (with tfjs-node) and browser side (with tfjs-webgl).

## What will you find here

The whole repository is just three files:
- **ctc.ts** contains the loss calculator, the gradient custom operation, and a helper function for decoding the results.
- **ctc.spec.ts** contains the test, and also some examples how to include it in the models.
- **perf.spec.ts** contains some performance experiments.

Otherwise, I will just document here my findings related to the implementation. I suggest checking out the PDF / Excel file which contains a careful calculation for a simple use-case, so you can follow what happens in each step, what is added up, what is multiplied, etc.

## Status

### Changelog

- **0.0.3** - Forward calculation is tensory. Fallback to Array-based solution is possible
- **0.0.2** - Handles variable length labels in large batches.
- **0.0.1** - Fist version is made public. Basic calculation is validated, works well on single and batch inputs. Can be included in models, `model.fit()` runs nice. 

### Currently woking on

- Making backward calculation tensory

### Known issues

- Thee's no input validation - input and label mismatches will result in runtime error
- No `tidy()` anywhere - larger fit() runs migth get out of memory errors.
- `wasm` backend fails to run gradient calculation. `tensorflow` and `cpu` works just fine

### TODO

- Unit tests - current spec is not comparing expected results
- Make it more tensory and depend less on JS native array operations

## Usage

Just include `ctc.ts` in your code, and start to build your model. You can check the ctc.spec.ts for examples, but the general thing you do is something like this:

```js
import * as tf from '@tensorflow/tfjs-core';
import { ctcLossGradient } from './ctc';

// build your model
const model = tf.sequential();
model.add(tf.layers.inputLayer({ name: 'inputLayer', inputShape: [5, 42] }));
// add your model's wizardry here - for the sake of simplicity, this is just a simple dense layer
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
Then you can use `model.fit()` as you usually would, with all the batches and epochs, etc.

### Inputs, labels and special restrictions

- This implementation operates on a 3D tensor: `[batch][(time)step][one-hot encoded embeddings]`.
- The inputs of the loss calculation are probabilities, prepare a model that each timestep's emberddings values' are between 0 and 1 and sum up to 1.
- For the learning process (fit) the labels should be one-hot encoded. Shape must be the same as for the inputs.
- The **last one-hot embedding** is used as the delimiter - this is different than the current TF Python implementation
- If the label is too short and does not fill the all the sequence, just padd it with the delimiter. All the delimiters are filtered when processing the labels, and are rebuilt from the ground.
- If you want detect empty space between characters, or silence while processing sound, add an embedding for that too, don't use the delimiter for that.

## Fallback to array-based calculation

According to our performance the tests, the batch size and the backend engine has a great impact on performance.
If you work with small batches (< 64), you can try to fallback to the Array-based calculation like this:

```js
import { CTC_LOSS_USE_ARRAY_ENGINE } from './ctc';
tf.env().set(CTC_LOSS_USE_ARRAY_ENGINE, true);
```
You should run some of your own experiments testing which implementation suites you.

## Development process
What you'll get here is a constant progress and the evolution of the codebase. Here's my approach:

- learn about the CTC algorithm, and it's usage
- examine the existing solutions
- lear about Tensorflow.JS' architecture
- develop a naive implementation - doesn't have to be nice, but it should work
- have tests developed that actually run and test the basics and edge cases
- refactor parts of it to be more tensory, and therefore presumably quicker
- if **it's good enough**, donate the whole stuff to the Tensorflow team to be included in the core product

Let's dive in!

### Learn about the problem itself

There are some articles one should read about the usage of the CTC algorithm. They are a good starting point to understand why we need it:

- https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5
- https://distill.pub/2017/ctc/
- https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c

The papers describing the algorithm are here:

- https://www.cs.toronto.edu/~graves/icml_2006.pdf - Graves et al.: Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
- https://www.isca-speech.org/archive_v0/Interspeech_2017/pdfs/1557.PDF - An Efficient Phone N-gram Forward-backward Computation Using Dense Matrix Multiplication
- http://bacchiani.net/resume/papers/ASRU2017.pdf - Improving the efficiency of forward-backward algorithm using batched computation in Tensorflow

Lectures:

- https://axon.cs.byu.edu/~martinez/classes/778/Papers/CTC.pdf - Logan Mitchell: Sequence to sequence learning
- https://www.youtube.com/watch?v=c86gfVGcvh4 - Carnegie Mellon University Deep Learning, S18 Lecture 14: Connectionist Temporal Classification (CTC)
- https://www.youtube.com/watch?v=GxtMbmv169o - Carnegie Mellon University Deep Learning, F18 Recitation 8: Connectionist Temporal Classification (CTC)
 

### Existing solutions

The only thing comes close to the native JS implementation is **@marsiancba**'s solution that was commented in this issue: https://github.com/tensorflow/tfjs/issues/1759
However, I couldn't wrap my head around some of the implementation's pecularities (namely, calculation of beta variables and input matching). However, it's usage of the tf operators is very advanced, worth checking out.

The Tensorflow **Python implementation** is definitely worth checking out: [ctc_ops.py](https://github.com/tensorflow/tensorflow/blob/87462bfac761435a46641ff2f10ad0b6e5414a4b/tensorflow/python/ops/ctc_ops.py)

My personal favourite is the **Stanford CTC implementation**, since it is understandable for folks who are not well versed in Python: https://github.com/amaas/stanford-ctc/blob/master/ctc/ctc_fast.pyx
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

CTC is a dynamic algorithm, which means a lot of slicing / concatenation / conditional stuff is happening. It will take time to get there.

### Test

CTC is a tricky alrgorythm, it has it's perks. The main problem is, that it requires lot's of data for the input, generates lot's of outputs, and it is hard to assemble a list of inputs - expected outputs pairs withouth doing tedious calculations. My approach is that there are obvious cases:
- matching inputs and labels should return a zero loss and zero gradiens
- random noise inputs should produce "something" other than error
- should handle signle elements and batched elements
- should run correctly with different length labels
- running fit() with 10 epochs we should see a decreasing loss

There is a big **TODO** here: have somebody who has some experience with the Python implementation, and generate some input-output pairs for our tests. Until then, let's extract some of the tests from here: [ctc_loss_op_test.py](https://github.com/tensorflow/tensorflow/blob/ab7c873202574b3cd549a93ccbfd881d659186ca/tensorflow/python/kernel_tests/nn_ops/ctc_loss_op_test.py)

Also, it is inevitable to calculate some results by hand. Check the excel for that.

### Refactor cycle

I've mentioned that it is ok to start with JavaScript arrays and stuff, but let me explain why we need to move to tensors. You see, Tensorflow is organized to have the concept of kernel functions - this means the codebase can utilize different implementations to execute the operations depending on what infrastructure is available withouth having the programmer rework everything for the sake of different platforms. Think about it this way: you have two big, multidimensional tensors, that you need to multiply elementwise. You would use something like this:
```js
const result = tensorA.mul(tensorB);
```
If you would need to do it in a pure JavaScript way, with arrays, you would implement the cycles, do the math, and return a new array with the results.
The execution depending on what's available for Tensorflow can be really different:
- **tfjs only** - just like you would do it in pure javascript, but somebody has programmed it for you. Sweet. Since modern JS engines are lightning fast, you'll get surprised how fast this implementation can run
- **tfjs-wasm** - the core is implemented in WebAssembly, so there's a significant improvement on performance. Not all functions are supported though. It is by far the fastest one if you don't use any CUDA-based 
- **tfjs-node** - kernel functions run natively on the processor, so you have the full power of your CPU, including the special instruction-sets you might have. However, usually you have to recompile Tensorflow from scratch to make use of it.
- **tfjs-webgl** - kernel functions take advantage of the parallel processing capabilities of your GPU

You might think that every engine brings at least a two-fold drop in execution time, but that's not the case - check the end of this document for details. But, because the speed is so much reliant on the backend implementation, it is essential to have as many things as possible "tenosry". It's not trivial - some of the operators are (albeit very useful) pretty hard to grasp.

That's where the learning process and the cycle kicks in.

## CTC algorithm implementation specialities

In this chapter, I'll describe what I did for making the algorithm more compatible with the TFJS' tensor concept. If you want to learn more about the algorithm, check the previous chapter for the links.

### Gradient calculation on batches

Nearly all the implementations I've seen handles the `y'` array (matrix, tensor, whichever you want to call it) in a special way: storing the selected embeddings and the separator embedding in different data-structures and choosing where to get from variables on-the-fly. I've choosen a different approach, trading memory for performance: the assembled `y'` array contains the selected predictions with the embeddings as well. This approach enables us to use the tensor functions effectively in the gradient calculation. 

The original paper's equation (16) looks like this:

![\frac{\partial O^{ML}(\left\{(x, z) \right\}, N__{w})}{\partial u_{k}^{t}}= y_{k}^{t}-\frac{1}{y_{k}^{t}Z__{t}}\sum_{s\in lab(z, k)}\hat{\alpha _{t}}(s) \hat{\beta _{t}}(s)](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20%5Cfrac%7B%5Cpartial%20O%5E%7BML%7D(%5Cleft%5C%7B(x,%20z)%20%5Cright%5C%7D,%20N__%7Bw%7D)%7D%7B%5Cpartial%20u_%7Bk%7D%5E%7Bt%7D%7D=%20y_%7Bk%7D%5E%7Bt%7D-%5Cfrac%7B1%7D%7By_%7Bk%7D%5E%7Bt%7DZ__%7Bt%7D%7D%5Csum_%7Bs%5Cin%20lab(z,%20k)%7D%5Chat%7B%5Calpha%20_%7Bt%7D%7D(s)%20%5Chat%7B%5Cbeta%20_%7Bt%7D%7D(s))  
where  
![Z__{t}\overset{\underset{\mathrm{def}}{}}{=}\sum_{s=1}^{\left| l' \right|} \frac{\hat{\alpha _{t}}(s) \hat{\beta _{t}}(s)} {y_{l_{s}^{'}}^{t}}](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20Z__%7Bt%7D%5Coverset%7B%5Cunderset%7B%5Cmathrm%7Bdef%7D%7D%7B%7D%7D%7B=%7D%5Csum_%7Bs=1%7D%5E%7B%5Cleft%7C%20l'%20%5Cright%7C%7D%20%5Cfrac%7B%5Chat%7B%5Calpha%20_%7Bt%7D%7D(s)%20%5Chat%7B%5Cbeta%20_%7Bt%7D%7D(s)%7D%20%7By_%7Bl_%7Bs%7D%5E%7B'%7D%7D%5E%7Bt%7D%7D)  
and  
![](https://latex.codecogs.com/png.image?%5Cdpi%7B110%7D%20%5Cbg_white%20lab(l,%20k)=%5Cleft%5C%7B%20s:%20l_%7Bs%7D%5E%7B'%7D=k%5Cright%5C%7D)

This looks frightening. Let's rephrase to make it a little more programmer friendly.

`alpha` and `beta` are the tensors we've calculated beforehand (these are the normalized forward and backward variables), and `y(l'(s))` is the tensor we've prepared from the original predictions. Let's put aside the `sum s in lab(z,k)` part for now, and concentrate on `Z(t)`.

Note, that `alpha`, `beta` and `y` tensors have the same shape, since they originate from the same data with the shape of `[batch, timestep, |l'|]` So, calculating each element after sum sign is a simple multiplication and division in TensorFlow (these functions work element-wise). Of course, we need to check not to divide by zero, but there's an inbuilt function `divNoNan` too. We are good to go.

Also, summing up by the timestep dimension is pretty easy, calling `sum()` and definig the axis.

```js
const Zt = a.mul(b).divNoNan(y).sum(2, true);
```

The `true` flag at the end of the `sum()` function call notes we don't want to collapse one dimension - this will come in handy in the further calculations. Also, there's a `cumsum()` function which does exactly the same, but during the test, `sum()` was faster.

Now that we have `Z(t)` in hand, we can calculate the first equation.

As you've probably guessed, the `alpha * beta` within the sum is the same thing we did when calculating `Z(t)`. the `sum lab` part is a fancy way of telling "sum up the same elements for the corresponding embedding and timestep". There are two things we can do to remain tensory as long as we can:
- since `Z(t)` is a constant for every timestep, we can move the calculation into the inner part of the sum, which means we divide every element with the corresponding `Z(t)`.
- `y(k, t)` behaves as a constant for the sum, it can be moved into the inner part of the sum too
- we can observe, that the `y(k, t)` (which is the original prediction) in our case equals to `y'(k, t)` (which is the rearranged `y` matrix according to `l'`) and this holds true in every embedding location. The good part is that `y'` is perfectly arranged for the division we are about to do.

The only drawback here is that the separator embedding will be divided `int(|l'|/2)` times more than just once. Also, if you have multiple instances of the same embedding the same thing applies - ex.: in the word 'alamo', `l'` is '\_a_l_a_m_o\_' and the timestep values referring to the embeddings 'a', and '\_' will to be divided more than once. This is the price for not having to rearrange structures in memory.

> I consider executing the same divisions on the same embeddings for keep everything in the Tensorflow pipeline a small price to pay. Using webGL in the browser, or using the pipelining and vectorization features of modern processors should even bring performance enhancements. But this needs to be tested.

With this approach, we gain another optimalization: reusing the result of `a.mul(b).div(y)` we did for calculating `Z(t)` comes handy. So our final heavy lifting (calculating the inner part of the sum of the first equation) takes the following form:

```js
const abDivY = a.mul(b).divNoNan(y);
const Zt = abDivY.sum(2, true); // setting it to true keeps the Zt tensor 3D instead of just 2D
const innerSum = abDivY.divNoNan(Zt); // 3D tensor can be diviced by a 3D tensor
```
Now we can move on to sum up the unique embeddings of `l'` into our gradient tensor - but that's a TODO for now.

### Variable length labels

The first implementation of the algorithm only worked when the the encoded labels in the batch were the same length. The code was usable this way with restrictions:
- batch size should be 1 **OR**,
- arrange the labels to train only on ones that have the same length.

Our friends at Google have thought about this problem, and came up with a good solution, which can be adapted to our case as well:
- before the gradient calculation, pad `alpha`, `beta` and `y` arrays with `0` to have the extended labels' axis the size of `sequenceLenght*2+1`
- the rationale behind using this number is that sequenceLength equals to the maximum number of predictable embedding, so the maximum the size of `l'` is `sequenceLenght*2+1`
- filling up with zero does not affect our calculations: division by zero is handled by `divNoNan()`, and the extra multiplications can be made faster with the enhancements of TFJS engines. On the other hand, the summations (for example: calculating `Z(t)`) won't be affected neither because adding up more zeros is not a problem.

For programming reasons, we keep track of the extended labels' embeddings (`bels` in the code), and this needs to be padded as well. In our representation, bels holds the label's embeddings' ids, so we needed to use one, that's not valid. In our case, it's `-1`, so it was pretty easy during the last summation to filter out the ones with this value.

So to sum it up: careful padding does not screw up our calculations, and makes it possible to handle variable length label-matching.

## Decoding one-hot tensors - some dirty hacks

There's a step, where we need to decode the labels, to gain the id's of the embeddings for later usage. I've developed two version of the code to check performance - each of them take a 3D tensor, and returns the ids of the place, where one-hot was located in a `number[][]` array:
- arraySync the input tensor, and do it in the JavaScript way
- do it with `topk().indices.squeeze()` and then arraySync-ed

Running 100.000 iterations on a 7th gen i5 with tfjs-node and webgl backend, the results are the following:

| Backend | Array method | tensory method |
| ------- | --------- | -------------- |
| tfjs-node | 1010 millisec | 10425 millisec |
| webgl in browser  | 851 millisec | 3765 millisec |

Let's rewrite the decode to return a tensor - since our aim is to keep everything as tensory as possible. The results are the following:

| Backend | Array method | tensory method |
| ------- | --------- | -------------- |
| tfjs-node | 1662 millisec | 9344 millisec |
| webgl in browser  | 1206 millisec | 3657 millisec |

Note the increase on the JS side - converting back from array to tensor takes some time. But even in that case, the JS method wins over the the tensory method. Interesting.

Now, the only thing remains in favour of the tensory method, is it's compactness:

```js
return t.topk(1).indices.squeeze([2]);
// vs
return tf.tensor( (<number[][][]>t.arraySync()).map(b => b.map(e => e. reduce((p, c, i) => c === 1 ? i : p, -1))) );
```

Which is nice.

### Collecting the end-results

There's an ongoind debate whether to use TF in cases where there are more simpler solutions using arrays. Generally, I'd say yes, but sometimes I see there are performance critical high-complexity functions where array-based methods just yields better results, and it's simpler too. Collecting the data and mapping to the return tensors is such a problem. Let's dive in.

We have `y'`, `l'` and `grad` in separate tensors. Our aim is to present two return tensors with the shape identical to the input tensor. All the values are zero, except where `l'` indicates otherwise (which is derived directly from the labeling). `retY` comes from `y'` directly, whereas `retG` comes from the `grad` but these values are not replaced but summed up if they are indicated in `l'` to be in multiple places.

To be frank, I couldn't find a way to produce `retG` with native tensor functions. I was thinking about using `tf.unique()` for this which returns the unique variables in the tensors along the appropriate axis, but unfortunately it does not work: `tf.unique()` on 2d tensors pads values with themselves if tensory properties (different length for different rows) are broken. Try this:

```JS
const a = tf.tensor2d([[1, 2, 3], [1, 1, 1], [2, 0, 0]]);
const {values, indices} = tf.unique(a, 1)
values.print(); 
indices.print();
```
The output will be this:
```
Tensor
    [[1, 2, 3], // this is ok
     [1, 1, 1], // this is not ok
     [2, 0, 0]] // this isn't ok, neither
Tensor
    [0, 1, 2]
```
It's not ok, since we can't use these results for gathering data, so: no, I'll do it the JS way.

Using a tensor buffer wouldn't seem to enhance things, so I stuck with standard arrays: iterate along all the dimensions, and map the results to the return tensor. Pretty easy, and we can handle the transposition as well.

Contrary to `retG`, `retY` could be calculated like this:
- Prepare an index from `l'` to be used with `tf.gather()`. Duplicates are overwritten multiple times, so we don't need to check whether it has been inserted already, or not.
- Use `tf.gather()` on the `y'` tensor to aggregate according to the output shape

Here's some code if you want to try it yourselves:

```JS
// prepare the gather tensor in an array form - which character should be inserted in the output tensor
const mappedBelsArray = belsArray.map( batchItem => {
  const ret = new Array(outputShape[2]).fill(batchItem.length);
  batchItem.filter( x => x != outputShape[2] ).forEach( (character, idx) => ret[character] = idx );
  return ret;
});
// pad the yParam to have a zero row, then do the gather based on mappedBelsArray. 
const retTensorY = tf.gather(yParam.pad([[0, 0], [0, 1], [0, 0]]), mappedBelsArray, 1, 1).transpose([0, 2, 1]);
// transpose is needed, since the end-result of gather has [batch, character, timestamp], and the input was [batch, timestamp, character]
return [retTensorY, tf.tensor3d(retG)];
```

Plug it into the `collectTensors()` function, comment the declaration of `yParam` and `retY` variables, and also remove the `retY[]` overwriting in the innermost loop. The trick is, that we pad the original yParam tensor with one element, and reference that when we need a zero tensor in place.

For me, running it on tfjs-node multiple times, it was 1.7 (JS) msec vs 2.18 msec (TF). WebGL should be faster in theory, but I havent't tried it (yet).

### Array vs. TensorFlow efficiency, speed, etc.

As I mentioned earlier, I've put a great deal of thougth into the performance aspect of the implementation. I couldn't find any real explanation what's happening under the hood, so instead of planning ahead for performance (like I usually do) I experimented a lot with different approaches.

I've already mentioned the kernel functions - it's TF's way of offloading the task to an implementing backend, be it native cpu, WebGL,or just stay in JS space. Now each of the handover and the processing of the return values comes at a price, but that's different at each backend. While using `wasm` or `cpu`, all the data remains in JS' memory space, in the other cases offloading / copy is neccessary. So you need to consider whether you push big enough data to the backend for complex, but optimized calculations which outweighs the price of handoff operations.

Another aspect to consider, is the actual execution environment of the JS engine. The V8 engine that drives Chrome / Node.JS is fantastic in optimizing the code on the run. I strongly suggest to watch this presentation by Franziska Hinkelmann on this topic: [Speed, Speed, Speed: JavaScript vs C++ vs WebAssembly](https://www.youtube.com/watch?v=aC_QLLilwso) So it's not a far fetched idea to run our code with the `cpu` backend utilizing a pure JS-only solution.

The last aspect to think about is that sometimes using TF's methods on tensors is just tedious work, and using arrays seems to be just more efficient. This is especially true if you try to implement complex, dynamic algorithms, which are not simple cases of adding, multiplying, averaging, etc on Tensors.

At the begining I've developed the CTC algorithm using arrays, and later moved some of the parts to tensorflow methods. Currently, the major components' implementation style is as follows:

| component      | Array-based | TF method based | Notes         |
| ------         | :----:      | :-----:         | ------------- |
| prepareTensor  |             | X               | Only filtering and padding is array-ish, everything else is tensory |
| forwardTensor  | X           | X               | You can switch impmelentation with setting the `env` |
| backwardTensor | X           |                 | Tensor implementation is a TODO |
| gradFunc       |             | X               | There will never be an array-based solution |
| collectTensors | X           |                 | See previous chapter for details |

Bacause of this approach, I could make a comparision on the execution times. The test generates random input with the batch size, then calculates the loss 1.000 times. The heavy lifting of the operations work on `[batch][5][11]` sized float tensors, so you can get an idea of size of the data involved. The execution times using my (7th gen. i5 laptop running Win11) for **one batch** is plotted onto this chart:

![Backend type and implementation impact on speed with different batch sizes](./doc/CTC%20batch%20performance%20v01.png)

The interesting thing is, that there's a massive overhead for using TF methods running on tfjs-node. However, this is compensated at how it behaves at larger batches. What's interesting to note, that for the smallest batch numbers (up until 8), running the code on pure JS backend (`cpu`) is the fastest.

Let's check the execution speed **per batch item** to see whether we have something notable (sorry for the log-scale chart):

![Backend type and implementation impact on batch item calculation speed with different batch sizes](./doc/CTC%20batch%20item%20performance%20v01.png)

So it seams that after a certain batch size (128), the calculation times settle - which is expcted, given the items have the same size, but the additional overhead working with batches is overcame by the large batch size.

**What can we learn from this?**

I've tested a lot of different approaches during this development. My bet is on the wasm backend, since smaller-scale tests ran, and I observer 2 to 4 times increase in performance. This means, to prepare for it, one **has to use** TF methods. Currently, something is wrong with using the loss calculation with wasm, so I'll need some time to figure it out, then I'll update the charts.

On the mean time, for small batches, I'll run my experiments with tfjs-node, but with the cpu backend, I'll get all the infrastructure benefits with the speed provided by V8.

On larger batches the native tfjs-node will be good - execution times are not as much different thant the array-based implementation. However, the later one will remain in the code for future reference.

If you want to reproduce the results on your own system, just execute this command:

```
ts-node .\src\perf.spec.ts
```
This will run the different batch sizes on the available backends. Here are my execution times for reference:

| Batch size | tf-node w. TF impl. | JS w. TF impl. |	tf-node w. Array impl. | JS w. Array impl. |  tf-node w. TF impl. per batch item | JS w. TF impl. per batch item |	tf-node w. Array impl. per batch item | JS w. Array impl. per batch item |
| ---- | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: |
| 1 | 7,099 msec | 1,663 msec | 1,651 msec | **0,766** msec | 7,099000 msec | 1,663000 msec | 1,651000 msec | 0,766000 msec |
| 2 | 6,799 msec | 2,059 msec | 1,536 msec | **0,708** msec | 3,399500 msec | 1,029500 msec | 0,768000 msec | 0,354000 msec |
| 4 | 6,751 msec | 2,719 msec | 1,411 msec | **0,937** msec | 1,687750 msec | 0,679750 msec | 0,352750 msec | 0,234250 msec |
| 8 | 6,986 msec | 4,283 msec | 1,641 msec | **1,564** msec | 0,873250 msec | 0,535375 msec | 0,205125 msec | 0,195500 msec |
| 16 | 7,523 msec | 7,326 msec | **2,038** msec | 2,712 msec | 0,470188 msec | 0,457875 msec | 0,127375 msec | 0,169500 msec |
| 32 | 8,944 msec | 13,693 msec | **2,827** msec | 5,383 msec | 0,279500 msec | 0,427906 msec | 0,088344 msec | 0,168219 msec |
| 64 | 10,226 msec | 25,976 msec | **6,172** msec | 9,948 msec | 0,159781 msec | 0,405875 msec | 0,096438 msec | 0,155438 msec |
| 128 | 14,311 msec | 51,954 msec | **8,864** msec | 20,994 msec | 0,111805 msec | 0,405891 msec | 0,069250 msec | 0,164016 msec |
| 256 | 22,77 msec | 103,942 msec | **16,138** msec | 44,863 msec | 0,088945 msec | 0,406023 msec | 0,063039 msec | 0,175246 msec |
| 512 | 41,455 msec | 210,922 msec | **36,626** msec | 89,933 msec | 0,080967 msec | 0,411957 msec | 0,071535 msec | 0,175650 msec |
| 1024 | 90,383 msec | 433,42 msec | **85,509** msec | 197,19 msec | 0,088265 msec | 0,423262 msec | 0,083505 msec | 0,192568 msec |

It is available on the Excel as well.

## Contribution, discussion, etc

Currently, this project is for my own amusement. However, if you find it worthy for your attention, reach out to me here, or at the [Tensorflow Forum](https://discuss.tensorflow.org/t/ctc-loss-implementation-in-tfjs/6645)

## License

As noted in the LICENSE.md file, this work is licensed under Creative Commons BY-NC 4.0
