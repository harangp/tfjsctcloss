import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
// import '@tensorflow/tfjs-backend-wasm';
import { ctcLossGradient } from '../ctc';

console.log(new Date(), "entering production mode");
tf.enableProdMode();

tf.setBackend("tensorflow").then( () => perfTest() )
.then( () => tf.setBackend("cpu").then( () => perfTest() ) );
// tf.setBackend("wasm").then( () => main() );

const EMBEDDINGS = "aábcčdeéfgďhiíjklmnñoóöőpqrsštťuúüűvwxyzž";

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

function perfTest() {

    const repetition = 1000;
    const convRep = 1000;
    const batchExponent = 8;

    const gFunc = tf.grads( (x, y) => ctcLossGradient(x, y));

    /** ------------------- performance test ----------------------
     * we are testing with different size batches
     */
    for (let i = 0; i < batchExponent; i++) {

        const batchSize = Math.pow(2, i);
        const [largeBatchPredictions, largeBatchLabels] = generateBatchData(batchSize);

        console.log(new Date(), 'trying to calculate GRADIENTS outside of fit() on variable length example with shape:', largeBatchLabels.shape, "iteration count:", repetition, "on", tf.getBackend());
        logSpeed( () => {
            for (let i = 0; i < repetition; i++) gFunc([largeBatchLabels, largeBatchPredictions]);
            return repetition;
        });
    }

    const y = tf.tensor(groundTruth).expandDims(1);
    y.print();

    console.log(new Date(), "speed testing forward calculating method: convolution, with", tf.getBackend());
    logSpeed( () => {
        const filter1 = <tf.Tensor3D>tf.ones([2, 1, 1]);
        const filter2 = <tf.Tensor3D>tf.ones([3, 1, 1]);
        for(let i = 0; i < convRep; i++) {
            tf.tidy( () => {
        
                y.pad([[0, 0], [0, 0], [1, 0]]).reverse(2).transpose([0, 2, 1]).conv1d(filter1, 1, "valid").transpose([0, 2, 1]).reverse(2);
                y.pad([[0, 0], [0, 0], [2, 0]]).reverse(2).transpose([0, 2, 1]).conv1d(filter2, 1, "valid").transpose([0, 2, 1]).reverse(2);
            });
        }    
        return convRep;
    });

    console.log(new Date(), "speed testing forward calculation method: padding-and-add, with", tf.getBackend());
    logSpeed( () => {
        for(let i = 0; i < convRep; i++) {
            tf.tidy( () => {
                const pad1 = tf.add(y, y.pad([[0, 0], [0, 0], [1, 0]], 0).slice([0, 0, 0], y.shape));
                tf.add(pad1, y.pad([[0, 0], [0, 0], [2, 0]], 0).slice([0, 0, 0], y.shape));
            });
        }    
        return convRep;
    });

}

function generateBatchData(size = 4): tf.Tensor[] {
    const pred = [];
    const labe = [];
    for (let i = 0; i < size; i++) {
        pred.push(groundNoise);
        labe.push(groundTruth);
    }

    return [tf.tensor(pred), tf.tensor(labe)];
}

// logging the speed of function
function logSpeed( fn: () => number ): void {
    const start = new Date().getTime();
    const elementNumber = fn();
    const stop = new Date().getTime();

    console.log(new Date(), 'ellapsed time:', stop-start, 'millisec, per batch element:', elementNumber == 0 ? 'N/A' : (stop-start)/elementNumber, 'millisec' );
}

