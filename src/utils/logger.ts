import * as tf from '@tensorflow/tfjs';

export const showTensorSize = ({
  tensor,
  tensorName,
}: {
  tensor: tf.Tensor<tf.Rank>;
  tensorName: string;
}) => {
  const shape = tensor.shape;
  const byteSize = tensor.size * tf.util.bytesPerElement(tensor.dtype);
  console.log(`${tensorName}: ${shape} | ${byteSize} bytes`);
};
