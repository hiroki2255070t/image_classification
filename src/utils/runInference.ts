import * as tf from '@tensorflow/tfjs';
import { processOutput } from './processOutput';
import { drawBoundingBoxes } from './draw';
import { COCO_CLASSES } from '../data/coco_classes';

export const runDetectionLoop = async (
  model: tf.GraphModel,
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement
) => {
  if (video.readyState < 3) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const [inputHeight, inputWidth] = model.inputs![0].shape!.slice(1, 3) as number[];

  const predictions = tf.tidy(() => {
    const frame = tf.browser.fromPixels(video);
    const resized = tf.image.resizeBilinear(frame, [inputHeight, inputWidth]);
    const inputTensor = resized.div(255).expandDims(0);

    const output = model.execute(inputTensor) as tf.Tensor;

    // inputTensorはdispose済み、outputは返す
    return output;
  });

  // processOutputは非同期処理なので tidy 外で実行
  const [boxes, scores, classes] = await processOutput(
    predictions,
    video.videoWidth / inputWidth,
    video.videoHeight / inputHeight
  );

  drawBoundingBoxes(ctx, boxes, scores, classes, COCO_CLASSES);

  // predictionsは使用後にdispose
  tf.dispose(predictions);
};
