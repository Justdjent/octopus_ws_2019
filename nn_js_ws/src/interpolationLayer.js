import * as tf from '@tensorflow/tfjs';

class Lambda extends tf.layers.Layer {
  constructor() {
    super({});
    this.supportsMasking = true;
    this.constOutputShape = [256, 256]
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], this.constOutputShape[0], this.constOutputShape[1], inputShape[3]]
  }

  call(inputs, kwargs) {
    let input = inputs;
    if (Array.isArray(input)) {
      input = input[0];
    }
    return tf.image.resizeBilinear(input, this.constOutputShape);
  }

  static get className() {
    return 'Lambda';
  }
}
tf.serialization.registerClass(Lambda);

// export function Lambda() {
//   return new Lambda();
// }