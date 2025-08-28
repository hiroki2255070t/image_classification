import * as tf from '@tensorflow/tfjs';


const SPLIT_LAYER_NAME = 'PartitionedCall/model_21/tf.math.multiply_61/Mul'; // model.jsonから適切な分割点を取得

export const executeInference = async ({
  model,
  inputTensor,
}: {
  model: tf.GraphModel;
  inputTensor: tf.Tensor<tf.Rank>;
}) => {
  // 1. モデルを分割して前半の推論を実行
  const middleTensor = model.execute(inputTensor, SPLIT_LAYER_NAME) as tf.Tensor;

  // 2. 後半の推論を実行
  const dummyInput = tf.zeros(model.inputs![0].shape!) as tf.Tensor; // エラー回避のため、images入力にダミーのテンソルを与える
  const predictions = model.execute(
    {
      images: dummyInput, // ダミー入力を渡す
      [SPLIT_LAYER_NAME]: middleTensor, // 処理した中間テンソルを渡す
    },
    model.outputs![0].name
  ) as tf.Tensor;

  tf.dispose([inputTensor, middleTensor, dummyInput]);

  return predictions;
};
