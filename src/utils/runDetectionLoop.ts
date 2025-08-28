import * as tf from '@tensorflow/tfjs';
import { processOutput } from './processOutput';
import { executeInference } from './executeInference';
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

  // 1. フレームから入力テンソルを作成
  const frame = tf.browser.fromPixels(video);
  const resized = tf.image.resizeBilinear(frame, [inputHeight, inputWidth]);
  const inputTensor = resized.div(255).expandDims(0);

  const predictions = await executeInference({ model, inputTensor });

  // 5. 後処理と描画
  const [boxes, scores, classes] = await processOutput(
    predictions,
    video.videoWidth / inputWidth,
    video.videoHeight / inputHeight
  );
  drawBoundingBoxes(ctx, boxes, scores, classes, COCO_CLASSES);

  // 6. 不要になったテンソルを全て破棄
  tf.dispose([frame, resized, predictions]);
};
