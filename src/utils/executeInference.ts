import * as tf from '@tensorflow/tfjs';

const SPLIT_LAYER_NAME = 'PartitionedCall/model_21/tf.math.multiply_61/Mul'; // model.jsonから適切な分割点を取得

export const executeInference = async ({
  model,
  inputTensor,
}: {
  model: tf.GraphModel;
  inputTensor: tf.Tensor<tf.Rank>;
}) => {
  // console.log(`inputTensor: ${inputTensor.shape}`); [1,640,640,3]
  // console.log(
  //   `inputTensor bytes: ${inputTensor.size * tf.util.bytesPerElement(inputTensor.dtype)} bytes`
  // ); // inputTensor bytes: 4915200 bytes

  // 1. モデルを分割して前半の推論を実行
  const middleTensor = model.execute(inputTensor, SPLIT_LAYER_NAME) as tf.Tensor;
  // console.log(`middleTensor: ${middleTensor.shape}`); // [1,40,40,128]
  // console.log(
  //   `middleTensor bytes: ${middleTensor.size * tf.util.bytesPerElement(middleTensor.dtype)} bytes`
  // ); // middleTensor bytes: 819200 bytes

  // 2.1. 中間テンソルを量子化・逆量子化
  const maxAbs = middleTensor.abs().max();
  const scale = maxAbs.div(127.0);
  const quantizedTensor = middleTensor.div(scale).round();

  // 2.2. GPUからデータを取得し、JavaScriptのInt8Arrayに変換【データ圧縮】
  const quantizedFloatData = await middleTensor.data();
  const quantizedInt8Array = new Int8Array(quantizedFloatData);
  // console.log(
  //   `quantizedInt8Array bytes: ${quantizedInt8Array.byteLength} bytes `
  // ); // quantizedInt8Array bytes: 204800 bytes

  // 2.3. 後続の計算のため、Int8Arrayをfloat32テンソルに再変換【データ展開】
  const dequantizedData = new Float32Array(quantizedInt8Array);
  const dequantizedTensor = tf.tensor(dequantizedData, middleTensor.shape).mul(scale);

  // 3. 後半の推論を実行
  const dummyInput = tf.zeros(model.inputs![0].shape!) as tf.Tensor; // エラー回避のため、images入力にダミーのテンソルを与える
  const predictions = model.execute(
    {
      images: dummyInput, // ダミー入力を渡す
      [SPLIT_LAYER_NAME]: middleTensor, // 処理した中間テンソルを渡す
    },
    model.outputs![0].name
  ) as tf.Tensor;

  tf.dispose([
    inputTensor,
    middleTensor,
    maxAbs,
    scale,
    quantizedTensor,
    dequantizedTensor,
    dummyInput,
  ]);

  return predictions;
};
