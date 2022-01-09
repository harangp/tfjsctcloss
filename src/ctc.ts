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
    const fwdTensors = forwardTensors(prepTensor.batchPaddedExtendedLabels, prepTensor.batchPaddedY, sequenceLength, embeddingLength);

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

        // We are moving to tensory implementation
        const prepTensor = prepareTensors(labels, predictions, delimiterIndex);
        const fwdTensors = forwardTensors(prepTensor.batchPaddedExtendedLabels, prepTensor.batchPaddedY, sequenceLength, embeddingLength);
        const loss = <O>fwdTensors.batchLoss;

        // TODO: this will be removed
        // onehots are decoded, then filtered not to include the delimiter, since it can confuse the algorithm
        const batchLabels = decodeOneHot(labels).map(e => e.filter((v) => v != delimiterIndex));
        // [batch][timesteps][embeddings] -> [batch][embeddings][timesteps]
        const batchInputs = <number[][][]>predictions.transpose([0, 2, 1]).arraySync();
        const prep = prepare(batchLabels, batchInputs, delimiterIndex);
        const batchExtendedLabels = prep.batchExtendedLabels; // l' in the papers
        const y = prep.yBatchMatrix;
        // const fwd = forward(batchExtendedLabels, y, sequenceLength);
        // const a = fwd.batchAlpha;

        const b = backward(batchExtendedLabels, y, sequenceLength);
        
        // we shouldn't really do this, since Javascript retains the context. But, it forces us to make everything tenosry, so it's not a problem
        // we need padding to handle variable length senteces. maximum recognisable length is equal to sequenceLength, so the padding length should be 2*seqLen+1
        const padTo = sequenceLength*2+1;

        // these are not required after we moved to tensory.
        // const paddedY = padY(y, padTo, 0);
        // const paddedA = padFwdBwd(a, padTo, 0); 
        const paddedB = padFwdBwd(b, padTo, 0); 
        // const paddedL = padLabels(batchExtendedLabels, padTo, -1);
        
        save([prepTensor.batchPaddedY.transpose([0, 2, 1]), fwdTensors.batchPaddedAlpha, paddedB, prepTensor.batchPaddedExtendedLabels]);

        // dy is to multiply the gradient. check `Engine.prototype.gradients` in tf-core.node.js, we don't use it here
        const gradFunc = (dy: O, saved: Tensor[]) => {
            const [yParam, aParam, bParam, bels] = saved;

            // we need to calculate the gradient - we'll use equation nr. 16 in the original paper to do so, but there's a slight modification. check redme.
            const ab = aParam.mul(bParam);

            const yParamTransposed =  yParam.transpose([0, 2, 1]);
            const abDivY = ab.divNoNan(yParamTransposed); // for calculating Z(t)

            const Zt = abDivY.sum(2, true); // this is faster than tf.cumsum()

            // this will be eq 16's second part. the divident is moved inside of the sum, because it's faster to calculate - we are tensory
            const grad = neg(abDivY.divNoNan(Zt)).transpose([0, 2, 1]); //shape after transpose: [batch][character][time]

            const retY = <number[][][]>tf.fill(labels.shape, 0).arraySync();
            const retG = <number[][][]>tf.fill(labels.shape, 0).arraySync();
            // note: this is faster than: tf.buffer(labels.shape).toTensor().arraySync(), and more faster than building a zero-filled array by hand

            // let's just pick-and-sum the relevant character's values from 'grad' according to the 'sum' part of eq 16
            const yArray = <number[][]>yParam.arraySync();
            const gradArray = <number[][]>grad.arraySync();
            const belsArray = <number[][]>bels.arraySync();

            belsArray.forEach( (batchItem, b) => {
                const foundChar: number[] = [];
                batchItem.forEach( (character, c) => {
                    // the length if the original embeddings (which is out of bounds) denotes it is a padder character not to be handled
                    if (character != embeddingLength) { 
                        // updating the gradient vector. every instance should be added up
                        for(let t = 0; t < sequenceLength; t++) {
                            retG[b][t][character] = retG[b][t][character] + gradArray[b][c][t];
                        }
                        // every character only needed to be included once from the yParam matrix.
                        // TODO: is there a simpler way to achieve this? what this does is essentially masks the prediction tensor for the letters that are contained in the label, erverything else is ignored / zeroed
                        if (!foundChar.includes(character)) {
                            for(let t = 0; t < sequenceLength; t++) {
                                retY[b][t][character] = retY[b][t][character] + yArray[b][c][t];
                            }
                            foundChar.push(character);
                        }
                    }
                });
            });

            // TODO: retY shoud be calculated lik this:
            // - get the unique values from bels tensor. should be done with tf.unique()
            // - use tf.scatterND() to aggregate?

            return [tf.tensor3d(retY), tf.tensor3d(retG)];
        }

        const ret = { value: loss, gradFunc };
        return ret;
    });

    const ret = customOp(truth, logits);

    return ret;
}

export const ctcLossGradient = op({ctcLossGradient_});

/**
 * Preparing everyhting for a successful  CTC loss calculation. This will return both the one-hot character ids and the y matrix.
 * 
 * TODO: filtered batchlabels contain all the characters we need in the bels list and the y matrix. so instead of scanning through, we just need to prepare the bels variable from batch labels
 * TODO: tf.gather() can be used for this, and we can stay tensory as follows:
 * 
 * labels should be gathered in [separator label1 separator label2 ...] fashion to form batchExtendedLabels
 * then yBatchMatrix should be gathered according to batchExtendedLabels
 * 
 * @param batchLabels labels (ground truth) in the batch
 * @param batchTransposedInputs inputs (predictions) in the batch
 * @param delimiterIndex
 * @returns batchExtendedLabels - the id list of the encoded logit, yBatchMatrix - the y matrix of the CTC calculation
 */
function prepare(batchLabels: number[][], batchTransposedInputs: number[][][], delimiterIndex: number): { batchExtendedLabels: number[][], yBatchMatrix: number[][][] } {

    const batchExtendedLabels = <number[][]>[];

    const yBatchMatrix = batchLabels.map( (x, i) => {
        const ret: number[][] = [];
        const extendedLabels = [];
        
        // the first one will allways be the separator
        ret.push(batchTransposedInputs[i][delimiterIndex]);
        extendedLabels.push(delimiterIndex);

        // we get the predictions for the letter we were supposed to get
        x.forEach( e => {
            ret.push(batchTransposedInputs[i][e]); // these are the predictions corersponding to the letter we assume based on the labels' characters
            ret.push(batchTransposedInputs[i][delimiterIndex]); // after every letter, we include a delimiter character

            extendedLabels.push(e); // this is the index of the found characters. it will be needed for the forward/backward calculation
            extendedLabels.push(delimiterIndex);
        });

        batchExtendedLabels.push(extendedLabels);
        return ret;
    });

    return { batchExtendedLabels, yBatchMatrix }
}

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
 * @returns {batchPaddedExtendedLabels, paddedBatchY}
 */
export function prepareTensors(batchLabels: Tensor, batchInputs: Tensor, delimiterIndex: number): { batchPaddedExtendedLabels: Tensor, batchPaddedY: Tensor } {

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

    return { batchPaddedExtendedLabels, batchPaddedY: paddedBatchY }
}

/**
 * Pads a 3D array's second dimension  (length - |embedding|) times. It is needed to be able to convert variable length arrays to tensors
 * Since the given length is usually the number of timesteps, the result will be a square for all of the batch elements
 * 
 * @param batchInput [batch][timestep][l' embeddings]
 * @param padTo length to pad to (usually the seqLen*2+1)
 * @param padValue 
 * @returns 3D tensor which has the shape of [batch][length][timestep]
 */
function padFwdBwd(batchInput: number[][][], padTo: number, padValue = 0): Tensor {
    const ret = batchInput.map( x => {
        return x.map( y => {
            const padder = padTo - y.length;
            return padder > 0 ? y.concat(new Array(padder).fill(padValue)) : y;
        });
    });
    return tf.tensor3d(ret);
}

/**
 * Pads the Y matrix, which has a different shape than the alpha/beta matrix
 * 
 * @param batchInput 
 * @param embeddingLen 
 * @param padTo 
 * @param padValue 
 * @returns 
 */
function padY(batchInput: number[][][], padTo: number, padValue = 0): Tensor {
    batchInput.forEach( x => {
        const padder = padTo - x.length;
        const seqLen = x[0].length;
        for(let i = 0; i < padder; i++) {
            x.push( new Array(seqLen).fill(padValue));
        }
    });
    return tf.tensor3d(batchInput);
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
 * batch alpha has the shape of [batch][symbol, or z', or l'][timesteps]
 * 
 * @param batchExtendedLabels 
 * @param yBatchMatrix 
 * @param sequenceLength 
 * @returns batchAlpha 3D array of the alpha variable, batchLoss - 1D array containing the loss for eatch batch item
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
                    // TODO: make the next line branchless to speed it up
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
 * Prepares the tensors for the original forward calculation. The input is the already padded l' and the y' tensor
 * Wraps the results to tensors
 * 
 * TODO: this will be eventually the tensory calculation method. I include it here as a starting point.
 * 
 * @param batchPaddedExtendedLabels - padded l'
 * @param batchPaddedY - padded y'
 * @param sequenceLength
 * @param labelPadder - the which the labels were padded if they were shorter (usually the length of the embeddings)
 * @returns {batchPaddedAlpha, batchLoss}
 */
function forwardTensors(batchPaddedExtendedLabels: Tensor, batchPaddedY: Tensor, sequenceLength: number, labelPadder: number): { batchPaddedAlpha: Tensor, batchLoss: Tensor } {

    const fwd = forward(<number[][]>batchPaddedExtendedLabels.arraySync(), <number[][][]>batchPaddedY.transpose([0, 2, 1]).arraySync(), sequenceLength, labelPadder);

    const batchPaddedAlpha = tf.tensor(fwd.batchAlpha);
    const batchLoss = tf.tensor(fwd.batchLoss);

    return { batchPaddedAlpha, batchLoss }
}

/**
 * Calculating the backward variables (betas)
 * 
 * beta has the shape of [batch][symbol, or z', or l'][timesteps]
 * 
 * @param batchExtendedLabels l' parameter in batches
 * @param yBatchMatrix calculated parameters from the neural network
 * @param sequenceLength expected sequence length - the 2nd dimension of the batchextendedlabels, and the 3rd dim of the ybatchamatrix should equal to this
 * @returns pure beta paramteres as specified in the original paper: 9-10-11 equations
 */
function backward(batchExtendedLabels: number[][], yBatchMatrix: number[][][], sequenceLength: number): number[][][] {

    const batchBeta = <number[][][]>[];

    batchExtendedLabels.forEach( (extendedLabel, i) => {

        const ret = [];

        const initStep = new Array(extendedLabel.length).fill(0);
        const lastElementNum = sequenceLength-1;
        const c0 = yBatchMatrix[i][extendedLabel.length-1][lastElementNum] + yBatchMatrix[i][extendedLabel.length-2][lastElementNum];

        initStep[initStep.length - 1] = yBatchMatrix[i][extendedLabel.length-1][lastElementNum] / c0; // delimiter
        initStep[initStep.length - 2] = yBatchMatrix[i][extendedLabel.length-2][lastElementNum] / c0; // last character

        ret.push(initStep);

        let prevStep = initStep;
        for(let t = sequenceLength - 2; t >= 0; t--) {
            const bkwdStep = [];

            const allowedIndex = extendedLabel.length - (sequenceLength - t - 3) * 2 - 1;
            for(let l = 0; l < extendedLabel.length; l++) {
                if (l < allowedIndex) {

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


