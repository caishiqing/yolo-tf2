from typing import Tuple, List
from tensorflow.keras import layers
import tensorflow as tf
from layers import *
from utils import compute_iou
from loss import IoULoss


class Detector(tf.keras.Model):

    def compile(self, optimizer, iou_mode='ciou'):
        super(Detector, self).compile(optimizer=optimizer)
        self.box_loss = IoULoss(mode=iou_mode)
        self.prb_loss = tf.keras.losses.BinaryCrossentropy()
        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def train_step(self, data):
        image, (label_box, label_cls) = data
        mask = tf.reduce_any(tf.not_equal(label_box, 0), axis=-1)
        with tf.GradientTape() as tape:
            box, prb, cls = self(image, training=True)
            iou_matrix = compute_iou(label_box, box, mode='iou', return_matrix=True)
            iou_matrix *= tf.cast(mask[:, :, tf.newaxis], dtype=iou_matrix.dtype)
            detect_idx = tf.argmax(iou_matrix, axis=1)
            detect_iou = tf.reduce_max(iou_matrix, axis=1)
            detect_box = tf.gather(label_box, detect_idx, batch_dims=0)
            detect_cls = tf.gather(label_cls, detect_idx, batch_dims=0)

            positive_mask = tf.greater(detect_iou, 0.7)
            negtive_mask = tf.greater(detect_iou, 0.0) & tf.less(detect_iou, 0.3)
            pred_box = tf.boolean_mask(box, positive_mask)
            label_box = tf.boolean_mask(detect_box, positive_mask)
            pred_cls = tf.boolean_mask(cls, positive_mask)
            label_cls = tf.boolean_mask(detect_cls, positive_mask)
            pred_positive_prb = tf.boolean_mask(prb, positive_mask)
            pred_negtive_prb = tf.boolean_mask(prb, negtive_mask)

            box_loss = self.box_loss(label_box, pred_box)
            prb_loss = self.prb_loss(tf.ones_like(pred_positive_prb), pred_positive_prb) +\
                self.prb_loss(tf.zeros_like(pred_negtive_prb), pred_negtive_prb)
            cls_loss = self.cls_loss(label_cls, pred_cls)
            loss = box_loss + prb_loss + cls_loss

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"box_loss": box_loss, "prb_loss": prb_loss, "cls_loss": cls_loss}

    def detect(self, image, max_num_objects=1, iou_threshold=0.7, score_threshold=0.5):
        image = tf.convert_to_tensor(image)
        box, prb, cls = self(image)
        cls = tf.argmax(cls, axis=-1)
        idx, num_valid = tf.image.non_max_suppression_padded(box, prb, max_num_objects,
                                                             iou_threshold=iou_threshold,
                                                             score_threshold=score_threshold,
                                                             pad_to_max_output_size=True)

        results = []
        for boxes, classes, indices, num in zip(box.numpy().tolist(),
                                                cls.numpy().tolist(),
                                                idx.numpy().tolist(),
                                                num_valid.numpy().tolist()):
            objects = []
            for i in indices[:num]:
                objects.append({'bbox': boxes[i], 'class': classes[i]})

            results.append(objects)

        return results


def Yolov3(image_size: Tuple[int] = (608, 608),
           num_classes: int = 80,
           anchor_priors: List[List[tuple]] = [
               [(0.27884614, 0.21634616), (0.375, 0.47596154), (0.89663464, 0.78365386)],
               [(0.07211538, 0.14663461), (0.14903846, 0.10817308), (0.14182693, 0.28605768)],
               [(0.02403846, 0.03125), (0.03846154, 0.07211538), (0.07932692, 0.05528846)]]
           ) -> Detector:

    assert len(image_size) == 2
    assert len(anchor_priors) == 3
    img = layers.Input(shape=tuple(image_size) + (3,))

    # Darknet53 Backbone
    x = conv(img, 32, 3, 1, activation="leaky")
    x = resX(x, 64, 1)
    x = resX(x, 128, 2)
    x1 = resX(x, 256, 8)
    x2 = resX(x1, 512, 8)
    x3 = resX(x2, 1024, 4)

    # Prediction 19 * 19
    x = convX(x3, [512, 1024, 512, 1024, 512], [1, 3, 1, 3, 1])
    h = conv(x, 1024, 3)
    logits1 = layers.Dense(len(anchor_priors[0]) * (5 + num_classes))(h)

    # Prediction 38 * 38
    x = FPN(x, x2, 256)
    x = convX(x, [256, 512, 256, 512, 256], [1, 3, 1, 3, 1])
    h = conv(x, 512, 3)
    logits2 = layers.Dense(len(anchor_priors[1]) * (5 + num_classes))(h)

    # Prediction 76 * 76
    x = FPN(x, x1, 128)
    x = convX(x, [128, 256, 128, 256, 128], [1, 3, 1, 3, 1])
    h = conv(x, 256, 3)
    logits3 = layers.Dense(len(anchor_priors[2]) * (5 + num_classes))(h)

    outputs = Heads([logits1, logits2, logits3], image_size, anchor_priors)
    model = Detector(inputs=img, outputs=outputs)
    return model


def Yolov4(image_size: Tuple[int] = (608, 608),
           num_classes: int = 80,
           anchor_priors: List[List[tuple]] = [
               [(0.27884614, 0.21634616), (0.375, 0.47596154), (0.89663464, 0.78365386)],
               [(0.07211538, 0.14663461), (0.14903846, 0.10817308), (0.14182693, 0.28605768)],
               [(0.02403846, 0.03125), (0.03846154, 0.07211538), (0.07932692, 0.05528846)]]
           ) -> Detector:

    assert len(image_size) == 2
    assert len(anchor_priors) == 3

    img = layers.Input(shape=tuple(image_size) + (3,))
    x = conv(img, 32, 3, 1, activation="leaky")

    # CSPDarknet53 Backbone
    x = conv(x, 32, 3, activation="mish")
    x = CSPX(x, 64, 1, activation="mish")
    x = CSPX(x, 128, 2, activation="mish")
    x1 = CSPX(x, 286, 8, activation="mish")
    x2 = CSPX(x1, 512, 8, activation="mish")
    x3 = CSPX(x2, 1024, 4, activation="mish")

    # Neck
    x = convX(x3, [512, 1024, 512], [1, 3, 1])
    x = SPP(x, [1, 5, 9, 13])
    x3 = convX(x, [512, 1024, 512], [1, 3, 1])
    x2 = conv(x2, 256, 1)
    x = FPN(x3, x2, 256)
    x4 = convX(x, [256, 512, 256, 512, 256], [1, 3, 1, 3, 1])
    x1 = conv(x1, 128, 1)
    x = FPN(x4, x1, 128)
    x5 = convX(x, [128, 256, 128, 256, 128], [1, 3, 1, 3, 1])
    x = PAN(x5, x4, 256)
    x6 = convX(x, [256, 512, 256, 512, 256], [1, 3, 1, 3, 1])
    x = PAN(x6, x3, 512)
    x7 = convX(x, [512, 1024, 512, 1024, 512], [1, 3, 1, 3, 1])

    # Prediction 76 * 76
    h = conv(x5, 256, 3)
    logits3 = layers.Dense(len(anchor_priors[2]) * (5 + num_classes))(h)

    # Prediction 38 * 38
    h = conv(x6, 512, 3)
    logits2 = layers.Dense(len(anchor_priors[1]) * (5 + num_classes))(h)

    # Prediction 19 * 19
    h = conv(x7, 1024, 3)
    logits1 = layers.Dense(len(anchor_priors[0]) * (5 + num_classes))(h)

    outputs = Heads([logits1, logits2, logits3], image_size, anchor_priors)
    model = Detector(inputs=img, outputs=outputs)
    return model


def Yolov5(image_size: Tuple[int] = (608, 608),
           num_classes: int = 80,
           anchor_priors: List[List[tuple]] = [
               [(0.27884614, 0.21634616), (0.375, 0.47596154), (0.89663464, 0.78365386)],
               [(0.07211538, 0.14663461), (0.14903846, 0.10817308), (0.14182693, 0.28605768)],
               [(0.02403846, 0.03125), (0.03846154, 0.07211538), (0.07932692, 0.05528846)]]
           ) -> Detector:

    assert len(image_size) == 2
    assert len(anchor_priors) == 3
    img = layers.Input(shape=tuple(image_size) + (3,))

    # 304 * 304 * 32
    x = Focus(img, 32, activation="leaky")
    # 152 * 152 * 64
    x = conv(x, 64, 3, 2, activation="leaky")
    x = CSP1_X(x, 64, 1, activation="leaky")
    # 76 * 76 * 128
    x = conv(x, 128, 3, 2, activation="leaky")
    x1 = CSP1_X(x, 128, 3, activation="leaky")
    # 38 * 38 * 256
    x = conv(x1, 256, 3, 2, activation="leaky")
    x2 = CSP1_X(x, 256, 3, activation="leaky")
    # 19 * 19 * 512
    x = conv(x2, 512, 3, 2, activation="leaky")
    x = SPP(x, [1, 5, 9, 13])
    x = conv(x, 512, 3)
    x = CSP2_X(x, 512, 1, activation="leaky")
    x3 = conv(x, 512, 3)

    # 38 * 38 * 256
    x = FPN(x3, x2, 256)
    x = CSP2_X(x, 256, 1, activation="leaky")
    x4 = conv(x, 256, 3)
    # 76 * 76 * 128
    x = FPN(x4, x1, 128)
    x5 = CSP2_X(x, 128, 1, activation="leaky")
    # 38 * 38 * 256
    x = PAN(x5, x4, 256)
    x6 = CSP2_X(x, 256, 1, activation="leaky")
    # 19 * 19 * 512
    x = PAN(x6, x3, 512)
    x7 = CSP2_X(x, 512, 1, activation="leaky")

    # Prediction 76 * 76
    logits3 = layers.Dense(len(anchor_priors[2]) * (5 + num_classes))(x5)

    # Prediction 38 * 38
    logits2 = layers.Dense(len(anchor_priors[1]) * (5 + num_classes))(x6)

    # Prediction 19 * 19
    logits1 = layers.Dense(len(anchor_priors[0]) * (5 + num_classes))(x7)

    outputs = Heads([logits1, logits2, logits3], anchor_priors)
    model = Detector(inputs=img, outputs=outputs)
    return model


if __name__ == "__main__":
    model = Yolov5()
    model.summary()
    x = tf.random.uniform((1, 608, 608, 3))
    y = model(x)
    print(y.shape)
