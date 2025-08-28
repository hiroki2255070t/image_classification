import * as tf from '@tensorflow/tfjs';

export const quantize = ({ tensor }: { tensor: tf.Tensor<tf.Rank> }) => {
  const maxAbs = tensor.abs().max();
  const scale = maxAbs.div(127.0);
  const quantizedTensor = tensor.div(scale).round();
  return quantizedTensor;
};

export const dequantize = async ({
  tensor,
  scale,
}: {
  tensor: tf.Tensor<tf.Rank>;
  scale: tf.Tensor<tf.Rank>;
}) => {
  const quantizedFloatData = await tensor.data();
  const quantizedInt8Array = new Int8Array(quantizedFloatData);

  const dequantizedData = new Float32Array(quantizedInt8Array);
  const dequantizedTensor = tf.tensor(dequantizedData, tensor.shape).mul(scale);
  return dequantizedTensor;
};
