import * as tf from '@tensorflow/tfjs';
import { ctcLossGradient, ctcLoss, decodeOneHot, CTC_LOSS_USE_ARRAY_ENGINE } from './ctc';

const EMBEDDINGS = "aábcčdeéfgďhiíjklmnñoóöőpqrsštťuúüűvwxyzž";

// tf.setBackend("tensorflow").then( () => main() );
// tf.setBackend("cpu").then( () => main() );
// tf.setBackend("wasm").then( () => main() );

// speed comparison
function logSpeed( fn: () => number ): void {
    const start = new Date().getTime();
    const elementNumber = fn();
    const stop = new Date().getTime();

    console.log(new Date(), 'ellapsed time:', stop-start, 'millisec, per batch element:', elementNumber == 0 ? 'N/A' : (stop-start)/elementNumber, 'millisec' );
}

// we have a gradient fucntion.
const gFunc = tf.grads( (x, y) => ctcLossGradient(x, y));

/**
 * --------------------------------------------------------------------------------------------------------------------------------------------------
 * Test data for the various tests
 * --------------------------------------------------------------------------------------------------------------------------------------------------
 */

// this is the first try to match - this one comes from the label. it needs to be padded as mentioned in the readme, because we have 5 timeslots here
const groundTruth = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
];

const groundNoise = [
    <number[]>tf.randomUniform([EMBEDDINGS.length+1]).softmax().arraySync(),
    <number[]>tf.randomUniform([EMBEDDINGS.length+1]).softmax().arraySync(),
    <number[]>tf.randomUniform([EMBEDDINGS.length+1]).softmax().arraySync(),
    <number[]>tf.randomUniform([EMBEDDINGS.length+1]).softmax().arraySync(),
    <number[]>tf.randomUniform([EMBEDDINGS.length+1]).softmax().arraySync()
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
    groundNoise
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

const expectedOneHotDecoded = tf.tensor([ [ 3, 0, 29, 41, 41 ] ]);

// example from Mr. Logan's lectures
const loganInput = tf.tensor([0.2, 0.1, 0.1, 0.6, 0, 0.7, 0.2, 0.1, 0.2, 0, 0, 0.8, 0.6, 0.1, 0.1, 0.2], [1, 4, 4]);

// loganLabels, modelling "CA__" - check the excel tables for refernce'
const loganLabels = tf.tensor([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], [1, 4, 4]);

// variable length labels, modelling "CA__, CAT_" for variable length'
const variInputs = tf.tensor([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [2, 4, 4]);
const variLabels = tf.tensor([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [2, 4, 4]);

// variables for matching the result from the python CTC implementation. check the Juniper notebook in the doc folder for details
const pyLabels = tf.tensor([[
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
]]);
const pyLogits = tf.tensor([[
    [1.0, 0.0, 0.0, 0.0], 
    [0.0, 0.0, 0.0, 1.0], 
    [1.0, 0.0, 0.0, 0.0], 
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0]
]]);
const expectedLoss = tf.tensor([4.448741]);
const expectedGradient = tf.tensor([[
    [-0.51064587,  0.17487772,  0.17487772,  0.16089036],
    [ 0.12286875, -0.6697599 ,  0.17487772,  0.37201348],
    [-0.29322746, -0.01850624,  0.17487772,  0.13685611],
    [-0.15988137, -0.20941935,  0.17487772,  0.1944232 ],
    [ 0.17487772, -0.5441786 ,  0.17487772,  0.19442326]
]]);


/**
 * --------------------------------------------------------------------------------------------------------------------------------------------------
 * Tests on CPU
 * --------------------------------------------------------------------------------------------------------------------------------------------------
 */
describe("Testing CTC algorythm on CPU with default CTC engine (tensory)", function() {

    beforeAll(async function() {
        await tf.setBackend("cpu");
        
        console.log(new Date(), tf.env().features);
        console.log(new Date(), "Backend is:", tf.getBackend());
    });

    it("should have CTC_LOSS_USE_ARRAY_ENGINE set to 'false'", function() {
        const dut = tf.env().getBool(CTC_LOSS_USE_ARRAY_ENGINE);
        console.log(new Date(), "ENV.CTC_LOSS_USE_ARRAY_ENGINE is set to", dut);
        expect(dut).toBeFalse();
    });

    it("should be able to use decodeOneHot function with oneLabel", function() {
        // decodeOneHot test
        console.log(new Date(), 'trying to decodeOneHot function with oneLabel');
        const tst1 = decodeOneHot(oneLabel);
        const dut = tf.tensor(tst1).flatten(); 

        // boolean tensor, multiply between elements - if there is one false, everything will be false.
        const cmp = dut.equal(expectedOneHotDecoded.flatten()).prod().arraySync();

        console.log(new Date(), "finished decoding, should be: [ [ 3, 0, 29, 41, 41 ] ]. Result:", tst1, cmp);
        expect(cmp).toBeTruthy();
    })

    it("should be performant", function() {
        const repetition = 1000;

        console.log(new Date(), "decodeOneHot speed test with", tf.getBackend());
        logSpeed( () => {
            for (let i=0; i< repetition; i++) decodeOneHot(batchLabels);
            return repetition;
        });    
    });

    /** ------------------- unit test ----------------------
     * oneLabel and oneSample is the same. the expected result is a zero gradient in two parts
     */
    it("should be able to calculate gradien on a 1 element batch", function() {
        console.log(new Date(), 'trying to calculate simple gradients outside of fit(): oneLabel/oneSample');
        const ret1 = gFunc([oneLabel, oneSample]);
        console.log(new Date(), 'finished calculation');
        ret1.forEach(x => x.print());
        
        expect(ret1.length).toBe(2);
    });
    
    /** ------------------- unit test ----------------------
     * batchPrediction / batchLabel invoked stuff must run without errors.
    */
    it("should work on batches", function() {

        console.log(new Date(), 'trying to calculate gradients outside of fit() on batchPrediction/batchLabel with element number:', batchPrediction.shape[0]);
        const ret2 = gFunc([batchLabels, batchPrediction]);
        console.log(new Date(), 'finished calculation');
        ret2.forEach(x => x.print());

        expect(ret2.length).toBe(2);
    });
    
    /** ------------------- unit test ----------------------
     * Let's test loss with Logan's inputs, since it is precalculated in the excel
    */
    it("should calculate an expected value for LOSS", function() {

        console.log(new Date(), 'trying to calculate LOSS on Mr. Logan\'s example with element number:', loganInput.shape[0]);
        const ret3 = ctcLoss(loganLabels, loganInput);
        console.log(new Date(), 'finished calculation');
        console.log('Exptected value based on Excel calculation: 1.13943')
        ret3.print(); 
        // according to the PDF, this should be 0.4665 according to p66, but it's not, since the PDF numbers are off (forward and backward matrices are not correctly constructed)
        // the other thing is, that the PDF is calculated as the variables are as-is, but the algorithm uses a softmax layer, so calculation is a bit messier.
        // TODO: calculate the expected value by hand, and have it match up
    });
    
    /** ------------------- unit test ----------------------
     * Let's test the gradient with Logan's inputs, since it is precalculated in the excel
    */
    it("should calculate GRADIENTS on Mr. Logan's example", function() {
        console.log(new Date(), 'trying to calculate GRADIENTS outside of fit() on Mr. Logan\'s example with element number:', loganInput.shape[0]);
        const ret4 = gFunc([loganLabels, loganInput]);
        console.log(new Date(), 'finished calculation');
        ret4.forEach(x => x.print());
        // TODO: testable criteria, not just a simple run without major errors
    });
   
    /** ------------------- unit test ----------------------
     * Let's test the gradient with variable length sentences. Actual results are not interesting (should be zero btw), we are checking wether length variation
     * is handled correctly in the algorithm.
     */
    it("should calculate GRADIENTS on variable length examples", function() {
        console.log(new Date(), 'trying to calculate GRADIENTS outside of fit() on variable length example with element number:', variLabels.shape[0]);
        const ret5 = gFunc([variLabels, variInputs]);
        console.log(new Date(), 'finished calculation');
        ret5.forEach(x => x.print());
        // TODO: test criteria, not just a simple run withouth major errors
    });

    /** ------------------- unit test ----------------------
     * Let's compare the results to Tensorflow's Python implementation. It's safe to assume, that for the same inputs, the same outputs are generated.
     * However, there's a catch: pyTF.ctc_loss() takes different structures of input: the labels are NOT one-hot encoded, and the logits are served
     * as-is, and the algorythm performs a log_softmax() on it. To compare the two, a softmax-ed input should be fed to TFJS, and inspect the results.
     * The expected results should be:
     *
     *   CTC loss: [4.448741]
     *   CTC gradients:  [[
     *   [-0.51064587  0.17487772  0.17487772  0.16089036]
     *   [ 0.12286875  0.17487772 -0.6697599   0.37201348]
     *   [-0.29322746  0.17487772 -0.01850624  0.13685611]
     *   [-0.15988137  0.17487772 -0.20941935  0.1944232 ]
     *   [ 0.17487772  0.17487772 -0.5441786   0.19442326]
     *   ]]
     */
    it("should match up with the Python implementation", function() {        
        console.log(new Date(), 'trying to calculate GRADIENTS outside of fit() on Python Tensorflow input:', pyLogits.shape);
        const ret6 = ctcLoss(pyLabels, pyLogits);
        const ret7 = gFunc([pyLabels, pyLogits]);
        console.log(new Date(), 'finished calculating grad calculation on Python Tensorflow input.');
        
        console.log('CTC loss:');
        ret6.print();
        const check6 = expectedLoss.sub(ret6).flatten().sum().arraySync();
        console.log(new Date(), 'Deviation from exptected loss:', check6);
        expect(check6).toBeCloseTo(0);

        console.log(new Date(), 'CTC gradients:');
        ret7.forEach(x => x.print());
        
        const check7 = expectedGradient.sub(ret7[1]).flatten().sum().arraySync();
        console.log(new Date(), 'Deviation from expected gradient:', check7);
        expect(check7).toBeCloseTo(0);
    });
        
    function buildModel(): tf.Sequential {
        console.log(new Date(), 'Building model for model tests');
        const model = tf.sequential();
        model.add(tf.layers.inputLayer({ name: 'inputLayer', inputShape: [5, 42] }));
        model.add(tf.layers.dense({ name: 'denseLayer', units: 42, kernelInitializer: 'ones', useBias: false, activation: 'relu'}))
        model.compile({
            loss: ctcLossGradient,
            optimizer: tf.train.adam(),
            metrics: ['accuracy']
        });
    
        return model;
    }

    /** ------------------- unit test ----------------------
     * CTC loss / gradient working inside of fit() - this is a standard model we'll use
     */
    it("should be able to build a model with ctcLossGradient as a loss generator", function() {        
        const model = buildModel();
        expect(model).toBeDefined();
        model.summary();
    });
    
    /** ------------------- unit test ----------------------
     * we use one sample with a batch of one to see it working
     */
    it("should be able to work inside model.fit() with a one element batch", async function() {
        const model = buildModel();

        const res1 = await model.fit(oneSample, oneLabel, {
            epochs: 10,
            batchSize: 1
        });

        expect(res1).toBeDefined();
        expect(res1.history.loss.length).toBeGreaterThan(0);

        console.log(new Date(), res1.history.loss[0]);
    });
    
    /** ------------------- unit test ----------------------
     * let's test with a correct batch, like above
    */
    it("should be able to work inside model.fit() with a normal batch", async function() {

        const model = buildModel();
        const res2 = await model.fit(batchPrediction, batchLabels, {
            epochs: 10,
            batchSize: 4
        });

        expect(res2).toBeDefined();
        expect(res2.history.loss.length).toBeGreaterThan(0);
        
        console.log(new Date(), res2.history.loss[0]);
    });  

});