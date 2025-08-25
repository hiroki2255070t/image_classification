import React, { useEffect, useRef, useState } from 'react';
import { useLoadModel } from './utils/loadModel';
import { drawDetection } from './utils/draw';
import { ObjectDetection } from '@tensorflow-models/coco-ssd';

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<ObjectDetection | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // 1. モデルの読み込み
  useEffect(() => {
    const loadModel = useLoadModel();
    loadModel()
      .then((loadedModel) => {
        setModel(loadedModel);
        setLoading(false);
        console.dir({loadedModel})
      })
      .catch((error) => {
        console.error('モデルの読み込みに失敗しました:', error);
        setLoading(false);
      });
  }, []);

  // 2. Webカメラのセットアップ
  useEffect(() => {
    const setupCamera = async () => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'environment' },
          });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.onloadedmetadata = () => {
              if (videoRef.current && canvasRef.current) {
                canvasRef.current.width = videoRef.current.videoWidth;
                canvasRef.current.height = videoRef.current.videoHeight;
              }
            };
            videoRef.current.play();
          }
        } catch (error) {
          console.error('カメラへのアクセスに失敗しました:', error);
        }
      }
    };
    setupCamera();
  }, []);

  // 3. 物体検出の実行
  useEffect(() => {
    const detectObjects = async () => {
      if (model && videoRef.current && videoRef.current.readyState === 4) {
        const predictions = await model.detect(videoRef.current);
        if (canvasRef.current) {
          const ctx = canvasRef.current.getContext('2d');
          if (ctx) {
            // 以前の描画をクリア
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            // 検出結果を描画
            drawDetection(predictions, ctx);
          }
        }
      }
    };

    const intervalId = setInterval(() => {
      detectObjects();
    }, 100); // 100ミリ秒ごとに検出

    return () => clearInterval(intervalId);
  }, [model]);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <h1 className="text-4xl font-bold mb-4">リアルタイム物体検出</h1>
      <div className="relative border-4 border-teal-500 rounded-lg shadow-lg">
        <video ref={videoRef} width="640" height="480" className="rounded" muted playsInline />
        <canvas ref={canvasRef} className="absolute top-0 left-0" />
      </div>
      <div className="mt-6 w-full max-w-sm">
        <h2 className="text-2xl font-semibold mb-2 text-center">ステータス</h2>
        {loading ? (
          <p className="text-center text-yellow-400">モデルを読み込み中です...</p>
        ) : (
          <p className="text-center text-green-400">モデル読み込み完了。検出を開始します。</p>
        )}
      </div>
    </div>
  );
};

export default App;
