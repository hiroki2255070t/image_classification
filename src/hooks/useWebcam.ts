import { useState, useRef, useEffect } from 'react';

export const useWebcam = (modelLoading: boolean) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [webcamStatus, setWebcamStatus] = useState('Awaiting model...');
  const [cameraReady, setCameraReady] = useState(false);

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

            // canvasサイズを動画に合わせる
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

  return { videoRef, canvasRef, webcamStatus, cameraReady };
};
