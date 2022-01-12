/**
 * CTC (Connectionist Temporal Classification) Operations
 * 
 * by Peter Harang
 * 
 * based on the original paper, and the Tensorflow Python implementation.
 * 
 * https://www.cs.toronto.edu/~graves/icml_2006.pdf
 * https://axon.cs.byu.edu/~martinez/classes/778/Papers/CTC.pdf
 * http://bacchiani.net/resume/papers/ASRU2017.pdf
 * 
 * Inspiration from:
 * 
 * https://github.com/tensorflow/tfjs/issues/1759
 * https://github.com/amaas/stanford-ctc/blob/master/ctc/ctc_fast.pyx
 * https://github.com/yhwang/ds2-tfjs
 * 
 * Lectures:
 * 
 * https://www.youtube.com/watch?v=c86gfVGcvh4
 * 
 * License: Creative Commons BY-NC 4.0
 */
import { Tensor, op, customGrad, GradSaveFunc, neg } from '@tensorflow/tfjs';
import * as tf from '@tensorflow/tfjs';

export const DELIMITER = '-';

/**
 * Computes the CTC (Connectionist Temporal Classification) loss. 
 * 
 * Implemetation follows as presented in (Graves et al., 2006).
 * 
 * TODO: input checks: labels tensor's shape should be equal to predictions tensor's shape. tensor's rank should be 3 (onehot)
 * 
 * @param labels Tensor containing the expected labels, according to `oneHotLabels` parameter. Either one-hot encoded, or number format (place of char in embeddings).
 * @param predictions 3-D `float` `Tensor`. This is a `Tensor` shaped: `[batch_size, max_time, num_classes]`. The logits.
 * 
 * @returns A 1-D `float` `Tensor`, size `[batch]`, containing the negative log probabilities
*/
function ctcLoss_<T extends Tensor>(labels: T, predictions: T): Tensor {
    // TODO: input checks: labels tensor's shape should be equal to predictions tensor's shape. tensor's rank should be 3 (onehot)
    const [, sequenceLength, embeddingLength] = labels.shape;

    // TODO: as I've seen, the python implementation moved from the last index to the first (0) index. Let's revise.
    const delimiterIndex = embeddingLength - 1;

    const prepTensor = prepareTensors(labels, predictions, delimiterIndex);
    const fwdTensors = forwardTensor(prepTensor.batchPaddedExtendedLabels, prepTensor.batchPaddedY, prepTensor.batchExtendedLabelLengths, sequenceLength, embeddingLength);

    return fwdTensors.batchLoss;
}

export const ctcLoss = op({ctcLoss_});

/**
 * Computes the CTC (Connectionist Temporal Classification) gradients as TFJS requires it
 * 
 * Implemetation follows as presented in (Graves et al., 2006).
 * 
 * At the current state, I just want this to work, so expect a lot of jumps between arrays and tensors. Later stage I plan to make as many thing as possible tensory.
 * 
 * @param truth label representation 3D tensor of [batch, time, onehot label]
 * @param logits the predictions
 * @returns 
 */
function ctcLossGradient_<T extends Tensor, O extends Tensor>(truth: T, logits: T): O {

    const customOp = customGrad((labels: Tensor, predictions: Tensor, save: GradSaveFunc) => {

        // TODO: input checks: labels tensor's shape should be equal to predictions tensor's shape. tensor's rank should be 3 (onehot)
        const [, sequenceLength, embeddingLength] = labels.shape;

        // TODO: as I've seen, the python implementation moved from the last index to the first (0) index. Let's revise.
        const delimiterIndex = embeddingLength - 1;

        const prepTensor = prepareTensors(labels, predictions, delimiterIndex);
        const fwdTensors = forwardTensor(prepTensor.batchPaddedExtendedLabels, prepTensor.batchPaddedY, prepTensor.batchExtendedLabelLengths, sequenceLength, embeddingLength);
        const bwdTensor = backwardTensor(prepTensor.batchPaddedExtendedLabels, prepTensor.batchPaddedY, sequenceLength, embeddingLength);
        const loss = <O>fwdTensors.batchLoss;

       save([prepTensor.batchPaddedY.transpose([0, 2, 1]), fwdTensors.batchPaddedAlpha, bwdTensor, prepTensor.batchPaddedExtendedLabels]);

        // dy is to multiply the gradient. check `Engine.prototype.gradients` in tf-core.node.js, we don't use it here
        const gradFunc = (dy: O, saved: Tensor[]) => {
            const [yParam, aParam, bParam, bels] = saved;

            const ab = aParam.mul(bParam); // we'll use equation nr. 16 in the original paper to do so, but there's a slight modification. check redme.

            const yParamTransposed =  yParam.transpose([0, 2, 1]);
            const abDivY = ab.divNoNan(yParamTransposed); // for calculating Z(t)

            const Zt = abDivY.sum(2, true); // this is faster than tf.cumsum()

            // this will be eq 16's second part. the divider is moved inside of the sum, because it's faster to calculate - we are tensory
            const grad = neg(abDivY.divNoNan(Zt)).transpose([0, 2, 1]); //shape after transpose: [batch][character][time]

            // summing up the relevant gradients to the return tensors
            const res = collectTensors(labels.shape, yParam, grad, bels, embeddingLength, sequenceLength);
            return res;
        }

        const ret = { value: loss, gradFunc };
        return ret;
    });

    const ret = customOp(truth, logits);

    return ret;
}

export const ctcLossGradient = op({ctcLossGradient_});

/**
 * This will be the same function as prepare(), but works on tensors, and returns tensors. The process goes like this:
 * 
 * - pad the inputs with one extra row on the embedding level. This will be 0 everywhere, and the index will be the length of the original embedding size
 * - do the one-hot detection on the label tensor - using `topk`, location of the highest value
 * - get the array representation, and iterate through the timestep arrays
 * - filter the resulst for the delimiter, and insert a delimiter for every character 
 * - pad the thing to |l|*2+1 with the value |l|
 * - gather the instances from the padded inputs according to the padded labels, and return it 
 * 
 * @param batchLabels labels (ground truth) in the batch
 * @param batchInputs predictions
 * @param delimiterIndex this index will be used as separator
 * @returns {batchPaddedExtendedLabels, paddedBatchY, batchExtendedLabelLengths}
 */
function prepareTensors(batchLabels: Tensor, batchInputs: Tensor, delimiterIndex: number): { batchPaddedExtendedLabels: Tensor, batchPaddedY: Tensor, batchExtendedLabelLengths: Tensor } {

    const paddedInput = batchInputs.pad([[0, 0], [0, 0], [0, 1]]);
    const batchDecodedLabels = batchLabels.topk(1).indices.squeeze([2]); // oneHot decoding, [batch][timestep] = one-hot index

    const belsArray = (<number[][]>batchDecodedLabels.arraySync()).map( x => {
        const ret = <number[]>[];
        x.filter(y => y != delimiterIndex).forEach(z => ret.push(delimiterIndex, z));
        ret.push(delimiterIndex);
        return ret;
    });

    const batchPaddedExtendedLabels = padLabels(belsArray, batchInputs.shape[1] * 2 + 1, batchInputs.shape[2]);
    const paddedBatchY = tf.gather(paddedInput, batchPaddedExtendedLabels, 2, 1);
    const batchExtendedLabelLengths = tf.tensor(belsArray.map( x => x.length ));

    return { batchPaddedExtendedLabels, batchPaddedY: paddedBatchY, batchExtendedLabelLengths }
}

/**
 * Pads the 2D array that holds the (usually extended) labels
 * 
 * @param bExtendedLabels the labels we need to pad
 * @param padTo sequence length to pad to
 * @param padValue default is -1 indicating that the value should not taken into account
 * @returns 2D tensor with the shape of [batch][seqLength]
 */
function padLabels(bExtendedLabels: number[][], padTo: number, padValue = -1): Tensor {
    const ret = bExtendedLabels.map( x => {
        const padder = padTo - x.length;
        return padder > 0 ? x.concat(new Array(padder).fill(padValue)) : x;
    });
    return tf.tensor2d(ret, [bExtendedLabels.length, padTo], "int32");
}

/**
 * Forward algoritm
 * 
 * The algorithm would generally underflow, so it's better to do it in the log space: log(a(t, l)) = log(e^(log(e^a(t-1, l) + e^(log(a(t-1, l-1)))))) + log(y(t))
 * The other way to do this is described in the original paper: normalizing the numbers on the go. We'll sitck with that.
 * 
 * batch alpha has the shape of [batch][symbol, or z', or l'][timesteps]. Please note, that l' is padded with zeros because of tensorflow, so at the original length
 * of l' we need to clip the results.
 * 
 * @param batchExtendedLabels 
 * @param yBatchMatrix 
 * @param sequenceLength 
 * @returns batchAlpha 3D array of the alpha variable, batchLoss - 1D array containing the loss for eatch batch item
 * 
 * @deprecated use forwardTensor - though this implementation is generally the faster one. Please note, once wasm will be ready, this implementation doesn't have a chance
 */
function forward(batchExtendedLabels: number[][], yBatchMatrix: number[][][], sequenceLength:number, labelPadder = -1 ): { batchAlpha: number[][][], batchLoss: number[] } {
    const batchAlpha = <number[][][]>[];
    const batchLoss  = <number[]>[];

    // with every batch item we calculate alpha, and do the normalization in place as noted in the original paper
    batchExtendedLabels.forEach( (extendedLabel, i) => {

        const ret = [];

        let labelLength = extendedLabel.findIndex( x => x === labelPadder );
        if (labelLength < 0) labelLength = extendedLabel.length;

        const initStep = new Array(extendedLabel.length).fill(0);
        const c0 = yBatchMatrix[i][0][0] + yBatchMatrix[i][1][0]; // rescale

        let loss = -Math.log(c0);

        initStep[0] = yBatchMatrix[i][0][0] / c0;
        initStep[1] = yBatchMatrix[i][1][0] / c0;
        ret.push(initStep);

        let prevStep = initStep;
        for(let t = 1; t < sequenceLength; t++) {
            const fwdStep = [];
            const allowedIndex = labelLength - (sequenceLength - t) * 2;
            for(let l = 0; l < extendedLabel.length; l++) {
                // this will cut out the ones that couldn't possibly reach the end (last elements)
                if (l >= allowedIndex && l < labelLength) {
                    // note: original paper states f(u) = u - 1 if l'(u) = blank or l'(u) = l'(u-2) 
                    // The first blank is handled (index can't be negative) all the other cases are covered with the later condition.
                    // The l-2 part is only added if character is not equal to the l-2. so current and previous is allways added
                    const sum = prevStep[l] + (l-1 >= 0 ? prevStep[l-1] : 0) + (extendedLabel[l] === extendedLabel[l-2] ? 0 : l-2 >= 0 ? prevStep[l-2] : 0);
                    fwdStep.push( yBatchMatrix[i][l][t] * sum );
                } else {
                    // if it's not within the allowed index range, just set it to zero
                    fwdStep.push(0);
                }
            }

            const c = fwdStep.reduce( (prev, curr) => prev + curr, 0); // C(t) calculation, sum of all items
            prevStep = fwdStep.map( x => x / c); // normalization to prevent under / overflow

            loss -= Math.log(c);

            ret.push( prevStep );
        }

        batchAlpha.push(ret);
        batchLoss.push(loss);
    });

    return { batchAlpha, batchLoss };
}

/**
 * Prepares the tensors for the original forward calculation.
 * 
 * batch alpha has the shape of [batch][symbol, or z', or l'][timesteps]. Please note, that l' is padded with zeros because of tensorflow, so at the original length
 * of l' we need to clip the results.
 * 
 * @param batchPaddedExtendedLabels - padded l'
 * @param batchPaddedY - padded y'
 * @param batchExtendedLabelLengths - the length of the unpadded extended labels
 * @param sequenceLength
 * @param labelPadder - the id which the labels were padded if they were shorter (usually the length of the embeddings)
 * @returns {batchPaddedAlpha, batchLoss}
 */
function forwardTensor(batchPaddedExtendedLabels: Tensor, batchPaddedY: Tensor, batchExtendedLabelLengths: Tensor, sequenceLength: number, labelPadder: number): { batchPaddedAlpha: Tensor, batchLoss: Tensor } {

    // Let's check wether the second addition is needed or not. l'[x] === l'[x-2]. Shift = pad-then-slice. Pad should be done with a value that's out of bounds in the embeddings indexes
    const shiftedBpel = batchPaddedExtendedLabels.pad([[0, 0], [2, 0]], -1).slice(0, batchPaddedExtendedLabels.shape);
    // integers can be safely checked against each other
    const summaryChooser = batchPaddedExtendedLabels.equal(shiftedBpel).expandDims(1);

    // let's prepare the masks. this will be unique for each batch item. remarkably simpple. you can mul() boolean and float numbers, it works.
    const padddingMask = batchPaddedExtendedLabels.notEqual(tf.scalar(labelPadder)).expandDims(1);

    // the first iteration: y[0] and y[1] is inserted, the other ones are filled with zeros
    const init = batchPaddedY.slice([0, 0], [batchPaddedY.shape[0], 1, 2]).pad([[0, 0], [0, 0], [0, batchPaddedY.shape[2]-2]]);
    const c0 = init.sum(2, true);
    
    let prevStep = init.divNoNan(c0);
    let loss = c0.log().neg();

    const stackable = [prevStep]; // results are collected here

    for (let i = 1; i < batchPaddedY.shape[1]; i++) {
        const y = batchPaddedY.slice([0, i, 0], [batchPaddedY.shape[0], 1, batchPaddedY.shape[2]]);

        const fwdMask = prepareFwdMask(batchExtendedLabelLengths, batchPaddedY.shape, i);

        // this is much speedier than doing a convolution (0.8019 msec vs 0.2439 msec in favour of the pad-and-add method)
        const rollingSum1 = tf.add(prevStep, prevStep.pad([[0, 0], [0, 0], [1, 0]], 0).slice([0, 0, 0], prevStep.shape));
        const rollingSum2 = tf.add(rollingSum1, prevStep.pad([[0, 0], [0, 0], [2, 0]], 0).slice([0, 0, 0], prevStep.shape));

        // we choose either the y'[s] + y'[s-1] or y'[s] + y'[s-1] + y'[s-2] based on l'[s] = l'[s-2]
        const fwdStep = tf.where(summaryChooser, rollingSum1, rollingSum2).mul(y).mul(fwdMask).mul(padddingMask);
        const c = fwdStep.sum(2, true); // C(t) calculation, sum of all items

        loss = loss.sub(c.log());

        prevStep = fwdStep.divNoNan(c);

        stackable.push(prevStep);
    }

    const bpa = tf.stack(stackable, 2).squeeze([1]);
    const bl = loss.squeeze([1, 2]);

    return { batchPaddedAlpha: bpa, batchLoss: bl }

    /*
    // proxy to the original calculation
    const fwd = forward(<number[][]>batchPaddedExtendedLabels.arraySync(), <number[][][]>batchPaddedY.transpose([0, 2, 1]).arraySync(), sequenceLength, labelPadder);

    const batchPaddedAlpha = tf.tensor(fwd.batchAlpha);
    const batchLoss = tf.tensor(fwd.batchLoss);

    // console.log(new Date(), "alpha with new method and old method");
    // bpa.print();
    // batchPaddedAlpha.print();

    // console.log(new Date(), "loss with new method and old method");
    // bl.print();
    // batchLoss.print();

    return { batchPaddedAlpha, batchLoss }
    */
}

/**
 * Prepares the mask to filter out unwanted calculations.
 * 
 * TODO: so tmpMask is constant. The creation can be factored out. However, the creation process is pretty fast, so no need to hurry
 * 
 * @param batchExtendedLabelLengths 
 * @param bpyShape 
 * @param timestep 
 * @returns 
 */
function prepareFwdMask(batchExtendedLabelLengths, bpyShape: number[], timestep: number): Tensor {

    const tmpMask = tf.range(0, bpyShape[2], 1);
    const stack = <Tensor[]>[];
    for (let i = 0; i < bpyShape[0]; i++) stack.push( tmpMask.clone() );
    const stackedMask = tf.stack(stack);

    const allowedIndexes = batchExtendedLabelLengths.sub((bpyShape[1] - timestep) * 2).expandDims(1);
    const ret = stackedMask.greaterEqual(allowedIndexes).expandDims(1);

    return ret;
}



/**
 * Calculating the backward variables (betas)
 * 
 * beta has the shape of [batch][symbol, or z', or l'][timesteps]
 * 
 * @param batchExtendedLabels l' parameter in batches
 * @param yBatchMatrix calculated parameters from the neural network
 * @param sequenceLength expected sequence length - the 2nd dimension of the batchextendedlabels, and the 3rd dim of the ybatchamatrix should equal to this
 * @param labelPadder the id which the labels were padded if they were shorter (usually the length of the embeddings)
 * @returns pure beta paramteres as specified in the original paper: 9-10-11 equations
 */
function backward(batchExtendedLabels: number[][], yBatchMatrix: number[][][], sequenceLength: number, labelPadder = -1): number[][][] {

    const batchBeta = <number[][][]>[];

    batchExtendedLabels.forEach( (extendedLabel, i) => {

        let labelLength = extendedLabel.findIndex( x => x === labelPadder );
        if (labelLength < 0) labelLength = extendedLabel.length;
        
        const ret = [];

        const initStep = new Array(extendedLabel.length).fill(0);
        const lastSeqStepId = sequenceLength-1;
        const c0 = yBatchMatrix[i][labelLength-1][lastSeqStepId] + yBatchMatrix[i][labelLength-2][lastSeqStepId];

        initStep[labelLength - 1] = yBatchMatrix[i][labelLength-1][lastSeqStepId] / c0; // delimiter
        initStep[labelLength - 2] = yBatchMatrix[i][labelLength-2][lastSeqStepId] / c0; // last character

        ret.push(initStep);

        let prevStep = initStep;
        for(let t = sequenceLength - 2; t >= 0; t--) {
            const bkwdStep = [];

            const allowedIndex = labelLength - (sequenceLength - t - 3) * 2 - 1;
            for(let l = 0; l < extendedLabel.length; l++) {
                if (l < allowedIndex && l < labelLength) {

                    let sum = prevStep[l];
                    sum += (l+1) < prevStep.length ? prevStep[l+1] : 0;
                    sum += !(l+2 < prevStep.length) || (extendedLabel[l] === extendedLabel[l+2]) ? 0 : prevStep[l+2];

                    bkwdStep.push( sum * yBatchMatrix[i][l][t] );
                } else {
                    bkwdStep.push(0);
                }
            }

            const c = bkwdStep.reduce( (prev, curr) => prev + curr, 0); // C(t) calculation, sum of all items
            prevStep = bkwdStep.map( x => x / c); // normalization to prevent under / overflow

            ret.push( prevStep );
        }

        // push() and reverse() is way faster than unshift() - 0.00006 vs. 0.01443 sec
        batchBeta.push(ret.reverse());
    });

    return batchBeta;
}


/**
 * Prepares the tensors for the original backward calculation. The inputs are the already padded l' and the y' tensors
 * Wraps the results to tensors
 * 
 * TODO: this will be eventually the tensory calculation method. I include it here as a starting point to adjust gradient calculation functions
 * 
 * @param batchPaddedExtendedLabels - padded l'
 * @param batchPaddedY - padded y'
 * @param sequenceLength
 * @param labelPadder - the id which the labels were padded if they were shorter (usually the length of the embeddings)
 * @returns pure beta paramteres as specified in the original paper: 9-10-11 equations
 */
function backwardTensor(batchPaddedExtendedLabels: Tensor, batchPaddedY: Tensor, sequenceLength: number, labelPadder: number): Tensor {

    const beta = backward(<number[][]>batchPaddedExtendedLabels.arraySync(), <number[][][]>batchPaddedY.transpose([0, 2, 1]).arraySync(), sequenceLength, labelPadder);

    return tf.tensor(beta);
}


/**
 * Collects the relevant parts of the calculated gradients into the output tensors. Two tensors are returned:
 * - the first is the y parameters of the embeddings indicated by the label (and the delimiter)
 * - the second is the gradient, where all the indices of l' should be added up and inserted into the tensor
 * 
 * There's a debate wether it is useful to do the calculations in a tensory way. I've tried to do for the first tensor, but
 * it was 25% slower on tfjs-node (1.7 msec vs 2.18 msec on average execution). Check readme for further elaboration.
 * 
 * TODO: outputShape contains embeddingLength and seqLength, they are unneccessary to pass to the function. Let's check.
 * 
 * @param outputShape should be equal to the input/label tensors' shape
 * @param yParam - y' tensor (padded)
 * @param grad - gradient tensor (padded)
 * @param bels - batch extended labels (padded)
 * @param embeddingLength - should be equal to the outputShape's 3rd value.
 * @param sequenceLength - number of steps to anticipate
 * @returns array of Tensors containing the gradients
 */
function collectTensors(outputShape: number[], yParam: Tensor, grad: Tensor, bels: Tensor, embeddingLength: number, sequenceLength: number  ): Tensor[] {

    const retY = <number[][][]>tf.fill(outputShape, 0).arraySync();
    const retG = <number[][][]>tf.fill(outputShape, 0).arraySync();
    // note: this is faster than: tf.buffer(labels.shape).toTensor().arraySync(), and more faster than building a zero-filled array by hand

    // let's just pick-and-sum the relevant character's values from 'grad' according to the 'sum' part of eq 16
    const yArray = <number[][]>yParam.arraySync();
    const gradArray = <number[][]>grad.arraySync();
    const belsArray = <number[][]>bels.arraySync();

    belsArray.forEach( (batchItem, b) => {
        batchItem.forEach( (character, c) => {
            // the length if the original embeddings (which is out of bounds) denotes it is a padder character - not to be handled
            if (character != embeddingLength) { 
                for(let t = 0; t < sequenceLength; t++) {
                    // updating the gradient vector. every instance should be added up
                    retG[b][t][character] = retG[b][t][character] + gradArray[b][c][t];
                    // every character only needed to be included once from the yParam matrix.
                    // TODO: is there a simpler way to achieve this? what this does is essentially masks the prediction tensor for the letters that are contained in the label, erverything else is ignored / zeroed
                    retY[b][t][character] = yArray[b][c][t];
                }
            }
        });
    });

    return [tf.tensor3d(retY), tf.tensor3d(retG)];
}


/**
 * Decodes a 3D oneHot tensor to a 3D array representing the instance the onehot encoded character's place in the list
 * 
 * Plese note: this will require a one-hot encoded input. If this is not met, it will return with the last place where it found a '1'
 * character, OR -1 if it couldn't find anything.
 * 
 * Investigating wheter tf.topk() could be used for this: yes, it could, but it's significantly slower (at least 3x, on node it's 8x).
 * Yes, even if you want to refactor it to return a Tensor, it's still faster to do it in the JS way, and creating a new Tensor.
 * But it's form is very nice, so I just leave it here: <number[][]>t.topk(1).indices.squeeze([2]).arraySync();
 * 
 * @param t Tensor to return the numeric representation of the number the oneHot character embedding
 * @returns number array of character placement
 */
export function decodeOneHot(t: Tensor): number[][] {
    const ret = (<number[][][]>t.arraySync()).map(b => b.map(e => e. reduce((p, c, i) => c === 1 ? i : p, -1)));
    return ret;
}

/**
 * Decodes CTC encoded labels from one-hot representations according to the given labels, removing separators:
 * 
 * a- -> a
 * -a- -> a
 * 
 * when mergeRepeated is set to true (default is false), consecutive collapsed characters are reduced to one instance
 * unless thely are separated by separator
 * 
 * aa -> a
 * ab -> ab
 * a-a -> aa
 * abb -> ab
 * 
 * @param input tensor of [barch, sequence, onehot_logits]
 * @param labels 
 * @param mergeRepeated
 */
 export function decode(input: Tensor, labels: string[], mergeRepeated = false): string[] {

    const reg = new RegExp(DELIMITER, 'g');

    const labelsPlusDelimiter = Array.from(labels);
    labelsPlusDelimiter.push(DELIMITER);

    if (input.shape.length != 3 && input.shape[2] != labelsPlusDelimiter.length) {
        throw Error('CTC decoder error: shape and label length does not match.' );
    }

    const oneHot = <number[][]>input.argMax(2).arraySync();

    const tmp = <string[][]>oneHot.map(x => x.map(y => labelsPlusDelimiter[y]));

    let ret : string[];

    if (mergeRepeated) {
        // TODO: can make it faster. instead of scanning the lastindex, let's check the last character of prev directly.
        ret = tmp.map( x => x.reduce( (prev, curr) => (prev.length != 0) && (prev.lastIndexOf(curr) + 1 == prev.length) ? prev : prev + curr, '' ));
    } else {
        ret = tmp.map( x => x.join(''));
    }

    return ret.map(x => x.replace(reg, ''));
}
