import * as tf from '@tensorflow/tfjs';

export const processOutput = async (
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
