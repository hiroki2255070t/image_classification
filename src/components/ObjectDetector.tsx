import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { useModel } from '../hooks/useModel';
import { COCO_CLASSES } from '../data/coco_classes';

const RealTimeDetector: React.FC = () => {
  const {
    model,
    loading: modelLoading,
    status: modelStatus,
  } = useModel('/model/yolov8_tfjs/model.json');

  const [webcamStatus, setWebcamStatus] = useState<string>('Awaiting model...');
  const [cameraReady, setCameraReady] = useState(false);
  const [loopStarted, setLoopStarted] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const runDetectionLoop = useCallback(async () => {
    if (!model || !videoRef.current || !canvasRef.current || videoRef.current.readyState < 3) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const modelInputShape = model.inputs![0].shape;
    const inputHeight = modelInputShape![1];
    const inputWidth = modelInputShape![2];

    const tensor = tf.tidy(() => {
      const frame = tf.browser.fromPixels(video);
      const resized = tf.image.resizeBilinear(frame, [inputHeight, inputWidth]);
      return resized.div(255.0).expandDims(0);
    });

    const predictions = model.execute(tensor) as tf.Tensor;
    tensor.dispose();

    const [boxes, scores, classes] = await processOutput(
      predictions,
      video.videoWidth / inputWidth,
      video.videoHeight / inputHeight
    );

    drawBoundingBoxes(ctx, boxes, scores, classes);
    tf.dispose(predictions);
  }, [model]);

  // 🎯 ループ開始関数
  const startLoop = useCallback(() => {
    if (loopStarted) return; // 二重起動防止
    setLoopStarted(true);

    let animationFrameId: number;
    const loop = async () => {
      await runDetectionLoop();
      animationFrameId = requestAnimationFrame(loop);
    };
    loop();

    return () => cancelAnimationFrame(animationFrameId);
  }, [runDetectionLoop, loopStarted]);

  // Webカメラ起動
  useEffect(() => {
    const startWebcam = async () => {
      setWebcamStatus('Starting camera...');
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        if (videoRef.current && canvasRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            setWebcamStatus('Webcam ready.');
            videoRef.current?.play();

            // 🎯 canvasサイズを動画に合わせる
            if (canvasRef.current && videoRef.current) {
              canvasRef.current.width = videoRef.current.videoWidth;
              canvasRef.current.height = videoRef.current.videoHeight;
            }

            setCameraReady(true);
          };
        }
      } catch (error) {
        console.error('Failed to start webcam:', error);
        setWebcamStatus('Failed to start webcam.');
      }
    };

    if (!modelLoading) startWebcam();
  }, [modelLoading]);

  // 🎯 「カメラ準備完了 + モデルロード完了」でループ開始
  useEffect(() => {
    if (model && !modelLoading && cameraReady) {
      startLoop();
    }
  }, [model, modelLoading, cameraReady, startLoop]);

  // === 以下 processOutput, drawBoundingBoxes は同じ ===
  const processOutput = async (
    predictions: tf.Tensor,
    widthRatio: number,
    heightRatio: number
  ): Promise<[number[][], number[], number[]]> => {
    const data = predictions.dataSync();
    const transposed = tf.tensor(data, [84, 8400]).transpose();
    const proposals = transposed.arraySync() as number[][];
    transposed.dispose();

    const boxes: number[][] = [];
    const scores: number[] = [];
    const classes: number[] = [];
    const confidenceThreshold = 0.5;

    proposals.forEach((proposal) => {
      const [x_center, y_center, width, height] = proposal.slice(0, 4);
      const classScores = proposal.slice(4);

      let maxScore = -1;
      let classId = -1;
      classScores.forEach((score, i) => {
        if (score > maxScore) {
          maxScore = score;
          classId = i;
        }
      });

      if (maxScore > confidenceThreshold) {
        scores.push(maxScore);
        classes.push(classId);
        boxes.push([
          (x_center - width / 2) * widthRatio,
          (y_center - height / 2) * heightRatio,
          width * widthRatio,
          height * heightRatio,
        ]);
      }
    });

    if (boxes.length === 0) return [[], [], []];

    const boxTensors = tf.tensor2d(boxes);
    const scoreTensors = tf.tensor1d(scores);
    const indices = await tf.image.nonMaxSuppressionAsync(boxTensors, scoreTensors, 20, 0.45);

    const finalBoxes = (await tf.gather(boxTensors, indices).array()) as number[][];
    const finalScores = (await tf.gather(scoreTensors, indices).array()) as number[];
    const finalClasses = (await tf
      .gather(tf.tensor1d(classes, 'int32'), indices)
      .array()) as number[];

    tf.dispose([boxTensors, scoreTensors, indices]);
    return [finalBoxes, finalScores, finalClasses];
  };

  const drawBoundingBoxes = (
    ctx: CanvasRenderingContext2D,
    boxes: number[][],
    scores: number[],
    classes: number[]
  ) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    ctx.strokeStyle = '#16a34a';
    ctx.lineWidth = 3;
    ctx.font = '16px sans-serif';
    ctx.fillStyle = '#16a34a';

    for (let i = 0; i < scores.length; i++) {
      const [x, y, width, height] = boxes[i];
      const label = `${COCO_CLASSES[classes[i]]}: ${scores[i].toFixed(2)}`;
      const mirroredX = ctx.canvas.width - x - width;

      ctx.strokeRect(mirroredX, y, width, height);

      const textWidth = ctx.measureText(label).width;
      const textHeight = parseInt(ctx.font, 10);
      const labelY = y > textHeight ? y : textHeight + 4;

      ctx.fillRect(mirroredX, labelY - textHeight, textWidth + 4, textHeight + 4);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, mirroredX + 2, labelY);
      ctx.fillStyle = '#16a34a';
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <h1 className="text-4xl font-bold mb-4">Real-Time Object Detection 🚀</h1>
      <div className="relative w-full max-w-4xl aspect-video rounded-lg shadow-xl overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover transform scale-x-[-1]"
        />
        <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full z-10" />
        {(modelLoading || webcamStatus !== 'Webcam ready.') && (
          <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center bg-black bg-opacity-50">
            <p className="text-xl">{modelLoading ? modelStatus : webcamStatus}</p>
          </div>
        )}
      </div>
      <p className="mt-4 text-gray-400">{modelStatus}</p>
    </div>
  );
};

export default RealTimeDetector;
