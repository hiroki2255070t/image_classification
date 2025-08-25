import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { GraphModel } from '@tensorflow/tfjs';

/**
 * 
 * @param modelPath : '/model/<MODEL_NAME>/model.json'
 * @returns 
 */
export const useModel = (modelPath: string) => {
  // モデル、ローディング状態、ステータスメッセージを管理
  const [model, setModel] = useState<GraphModel | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [status, setStatus] = useState<string>('Loading model...');

  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadGraphModel(modelPath);
        setModel(loadedModel);
        setLoading(false);
        setStatus('Model loaded successfully.');
      } catch (error) {
        console.error('Failed to load model:', error);
        setLoading(false);
        setStatus('Failed to load model. Check console for details.');
      }
    };

    loadModel();
  }, [modelPath]); // modelPathが変更された場合（通常はない）に再実行

  return { model, loading, status };
};
