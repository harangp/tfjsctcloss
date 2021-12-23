import * as tf from '@tensorflow/tfjs-node';
import { ctcLossGradient } from './ctc';

const EMBEDDINGS = "aábcčdeéfgďhiíjklmnñoóöőpqrsštťuúüűvwxyzž";

// const labelDut = ['cat', 'cat', 'cat', 'cat'];
// const labels = labelDut.map( x => Array.from(x).map( c => EMBEDDINGS.indexOf(c)));
// this is the coded truth. the example uses these long embeddings, because the utterances include accented characters. This encodes thw word  "cat".
// const oneHotLabel = [
//     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
//     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
//     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
// ];

// this is the first try to match - this one comes from the label. it needs to be padded as mentioned in the readme, because we have 5 timeslots here
const groundTruth = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
];

// characters fully match on the first try, characters have separators, prediction is not fully fit, random noise.
const batchPrediction = tf.tensor([
    groundTruth,
    [
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ],
    [
        [0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]
    ],
    [
        <number[]>tf.randomUniform([EMBEDDINGS.length+1]).softmax().arraySync(),
        <number[]>tf.randomUniform([EMBEDDINGS.length+1]).softmax().arraySync(),
        <number[]>tf.randomUniform([EMBEDDINGS.length+1]).softmax().arraySync(),
        <number[]>tf.randomUniform([EMBEDDINGS.length+1]).softmax().arraySync(),
        <number[]>tf.randomUniform([EMBEDDINGS.length+1]).softmax().arraySync()
    ]
]);

const batchLabels = tf.tensor([
    groundTruth,
    groundTruth,
    groundTruth,
    groundTruth
]);

const oneSample = tf.tensor([
    groundTruth
]);

const oneLabel = tf.tensor([
    groundTruth
]);

/** ------------------- unit test ----------------------
 * oneLabel and oneSample is the same. the expected result is a zero gradient in two parts
 */
console.log(new Date(), 'trying to calculate simple gradients outside of fit()');
const gFunc = tf.grads( (x, y) => ctcLossGradient(x, y));
const ret1 = gFunc([oneLabel, oneSample]);
console.log(new Date(), 'finished calculation');
ret1.forEach(x => x.print());

/** ------------------- unit test ----------------------
 * batchPrediction / batchLabel invoked stuff must run without errors.
 */
console.log(new Date(), 'trying to calculate gradients outside of fit() on a batch with element number:', batchPrediction.shape[0]);
const ret2 = gFunc([batchLabels, batchPrediction]);
console.log(new Date(), 'finished calculation');
ret2.forEach(x => x.print());


/** ------------------- unit test ----------------------
 * CTC loss / gradient working inside of fit() - this is a standard model we'll use
 */
const model = tf.sequential();
model.add(tf.layers.inputLayer({ name: 'inputLayer', inputShape: [5, 42] }));
model.add(tf.layers.dense({ name: 'denseLayer', units: 42, kernelInitializer: 'ones', useBias: false, activation: 'softmax'}))
model.compile({
    loss: ctcLossGradient,
    optimizer: tf.train.adam(),
    metrics: ['accuracy']
});
model.summary();


async function modelTest() {

    /** ------------------- unit test ----------------------
     * we use one sample with a batch of one to see it working
     */
    const res1 = await model.fit(oneSample, oneLabel, {
        epochs: 10,
        batchSize: 1
    });
    
    console.log(new Date(), res1.history.loss[0]);

    /** ------------------- unit test ----------------------
     * let's test with a correct batch, like above
     */
    const res2 = await model.fit(batchPrediction, batchLabels, {
        epochs: 10,
        batchSize: 1
    });

    console.log(new Date(), res2.history.loss[0]);
}

modelTest().then( () => { console.log(new Date(), "done"); });