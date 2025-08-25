import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';

/**
 * COCO-SSDモデルをロードする
 * @returns 読み込んだモデルオブジェクト
 */
const loadCocoSsdModel = async (): Promise<cocoSsd.ObjectDetection> => {
  console.log('モデルを読み込んでいます...');
  // TensorFlow.jsのバックエンドを初期化
  await tf.ready();
  // COCO-SSDモデルをロード
  const model = await cocoSsd.load();
  console.log('モデルの読み込みが完了しました。');
  return model;
};

/**
 * dependency injection
 */
export const useLoadModel = () => {
  return loadCocoSsdModel;
};
