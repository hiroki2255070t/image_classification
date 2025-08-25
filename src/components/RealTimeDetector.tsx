import React, { useRef, useEffect, useCallback } from 'react';
import { useModel } from '../hooks/useModel';
import { useWebcam } from '../hooks/useWebcam';
import { runDetectionLoop } from '../utils/runDetectionLoop';

const RealTimeDetector: React.FC = () => {
  const {
    model,
    loading: modelLoading,
    status: modelStatus,
  } = useModel('/model/yolov8_tfjs/model.json');
  const { videoRef, canvasRef, webcamStatus, cameraReady } = useWebcam(modelLoading);

  const rafIdRef = useRef<number | null>(null);

  const startLoop = useCallback(() => {
    if (!model || !videoRef.current || !canvasRef.current) return;
    if (rafIdRef.current != null) cancelAnimationFrame(rafIdRef.current);

    const tick = async () => {
      await runDetectionLoop(model, videoRef.current!, canvasRef.current!);
      rafIdRef.current = requestAnimationFrame(tick);
    };

    rafIdRef.current = requestAnimationFrame(tick);
  }, [model, videoRef, canvasRef]);

  useEffect(() => {
    if (model && !modelLoading && cameraReady) startLoop();

    return () => {
      if (rafIdRef.current != null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, [model, modelLoading, cameraReady, startLoop]);

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
