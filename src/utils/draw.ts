import type { DetectedObject } from '@tensorflow-models/coco-ssd';

/**
 * 検出結果を描画する
 * @param detections 検出されたオブジェクトの配列
 * @param ctx 描画対象のCanvasRenderingContext2D
 */
export const drawDetection = (
  detections: DetectedObject[],
  ctx: CanvasRenderingContext2D
): void => {
  detections.forEach((prediction) => {
    const [x, y, width, height] = prediction.bbox;
    const text = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;

    // スタイルの設定
    ctx.strokeStyle = '#00FFFF'; // バウンディングボックスの色 (シアン)
    ctx.lineWidth = 2;
    ctx.fillStyle = '#00FFFF';
    ctx.font = '16px Arial';

    // バウンディングボックスの描画
    ctx.beginPath();
    ctx.rect(x, y, width, height);
    ctx.stroke();

    // ラベルの描画
    ctx.fillText(text, x, y > 10 ? y - 5 : 10);
  });
};
