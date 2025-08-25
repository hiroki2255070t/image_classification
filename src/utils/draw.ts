export const drawBoundingBoxes = (
  ctx: CanvasRenderingContext2D,
  boxes: number[][],
  scores: number[],
  classes: number[],
  labels: string[]
) => {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.strokeStyle = '#16a34a';
  ctx.lineWidth = 3;
  ctx.font = '16px sans-serif';
  ctx.fillStyle = '#16a34a';

  for (let i = 0; i < scores.length; i++) {
    const [x, y, width, height] = boxes[i];
    const label = `${labels[classes[i]]}: ${scores[i].toFixed(2)}`;
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
