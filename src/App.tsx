import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { GraphModel } from '@tensorflow/tfjs';
import { IMAGENET_CLASSES } from './data/imageset';

// 予測結果の型定義
interface Prediction {
  className: string;
  probability: number;
}

// モデルと中間層の情報
const MODEL_URL =
  'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1';
const INTERMEDIATE_LAYER_NAME = 'module_apply_default/MobilenetV2/Conv1/Conv2D';

const App: React.FC = () => {
  // useRefにDOM要素の型を指定
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // useStateに状態の型を指定
  const [model, setModel] = useState<GraphModel | null>(null);
  const [intermediateModel, setIntermediateModel] = useState<GraphModel | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);

  // 1. モデルの読み込み
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('モデルを読み込んでいます...');
        const loadedModel = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
        setModel(loadedModel);
        console.log('モデルの読み込みが完了しました。');

        // 中間層の出力を取得するためのモデルを構築
        // GraphModelのlayersプロパティはオプショナルなため、存在をチェック
        if ('layers' in loadedModel && loadedModel.layers) {
          if (loadedModel.layers) {
            const layer = loadedModel.layers.find((l) => l.name === INTERMEDIATE_LAYER_NAME);
            if (layer) {
              const interModel = tf.model({
                inputs: loadedModel.inputs,
                outputs: layer.output,
              }) as GraphModel; // tf.modelの戻り値はModelなのでGraphModelにキャスト
              setIntermediateModel(interModel);
              console.log(`中間層 (${INTERMEDIATE_LAYER_NAME}) モデルの構築が完了しました。`);
            } else {
              console.warn(`警告: 中間層 '${INTERMEDIATE_LAYER_NAME}' が見つかりませんでした。`);
              console.log(
                '利用可能なレイヤー名:',
                loadedModel.layers.map((l) => l.name)
              );
            }
          } else {
            console.warn(`警告: このモデルではレイヤー情報の取得ができません。`);
          }
        }
      } catch (error) {
        console.error('モデルの読み込みに失敗しました:', error);
      }
    };
    loadModel();
  }, []);

  // 2. Webカメラのセットアップ
  useEffect(() => {
    const setupCamera = async () => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 224, height: 224, facingMode: 'environment' },
          });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
          }
        } catch (error) {
          console.error('カメラへのアクセスに失敗しました:', error);
        }
      }
    };
    setupCamera();
  }, []);

  // 3. 予測の実行
  useEffect(() => {
    const intervalId = setInterval(() => {
      // videoRef.currentがnullでないことと、再生状態を確認
      if (model && videoRef.current && videoRef.current.readyState === 4) {
        predict();
      }
    }, 500);

    return () => clearInterval(intervalId);
  }, [model, intermediateModel]);

  // 4. 予測と中間特徴量の取得
  const predict = async (): Promise<void> => {
    // modelとvideoRef.currentの存在を再度チェック (型ガード)
    if (!model || !videoRef.current) return;

    // tf.tidyを使用してテンソルのメモリを自動的にクリーンアップ
    tf.tidy(() => {
      // ビデオからテンソルを作成
      const tensor = tf.browser
        .fromPixels(videoRef.current)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims();

      // 画像の正規化
      const normalizedTensor = tensor.div(127.5).sub(1);

      // 非同期処理をtidy外で扱うため、Promiseを直接操作
      const processPredictions = async () => {
        // 通常の予測
        const result = model.predict(normalizedTensor) as tf.Tensor;
        const probabilities = await result.data();

        const topK: Prediction[] = Array.from(probabilities)
          .map((p, i) => ({
            probability: p,
            // 変更前: className: `ID: ${i}`
            className: IMAGENET_CLASSES[i] || `不明なID: ${i}`, // インデックスを使ってクラス名を取得
          }))
          .sort((a, b) => b.probability - a.probability)
          .slice(0, 3);
        setPredictions(topK);

        // 中間特徴量の取得とコンソールへの表示
        if (intermediateModel) {
          const intermediateResult = intermediateModel.predict(normalizedTensor) as tf.Tensor;
          console.log('--- 中間特徴量 ---');
          console.log(`Shape: [${intermediateResult.shape.join(', ')}]`);
          intermediateResult.print();
          console.log('--------------------');
          intermediateResult.dispose(); // 手動でdispose
        }
      };

      processPredictions();
    });
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold mb-4">画像分類 (TypeScript)</h1>
      <div className="relative border-4 border-teal-500 rounded-lg shadow-lg">
        <video
          ref={videoRef}
          width="224"
          height="224"
          className="rounded"
          muted // ブラウザの自動再生ポリシーのため
          playsInline // iOSでのインライン再生のため
        />
        <canvas ref={canvasRef} className="absolute top-0 left-0" />
      </div>
      <div className="mt-6 w-full max-w-sm">
        <h2 className="text-2xl font-semibold mb-2 text-center">予測結果</h2>
        {model ? (
          <ul className="bg-gray-800 p-4 rounded-lg">
            {predictions.length > 0 ? (
              predictions.map((pred, index) => (
                <li key={index} className="flex justify-between items-center mb-2">
                  <span className="text-gray-300">{pred.className}</span>
                  <div className="w-1/2 bg-gray-700 rounded">
                    <div
                      className="bg-teal-500 text-xs font-medium text-blue-100 text-center p-0.5 leading-none rounded"
                      style={{ width: `${(pred.probability * 100).toFixed(2)}%` }}
                    >
                      {`${pred.probability.toFixed(2)}%`}
                    </div>
                  </div>
                </li>
              ))
            ) : (
              <p className="text-center text-gray-400">カメラ映像を解析中...</p>
            )}
          </ul>
        ) : (
          <p className="text-center text-yellow-400">モデルを読み込み中です...</p>
        )}
      </div>
      <p className="mt-4 text-sm text-gray-500">
        開発者ツールのコンソールで中間特徴量を確認できます。
      </p>
    </div>
  );
};

export default App;
