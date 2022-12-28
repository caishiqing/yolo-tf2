import tensorflow as tf
from utils import compute_iou


class IoULoss(tf.keras.losses.Loss):
    def __init__(self,
                 mode: str = "iou",
                 reduction: str = tf.keras.losses.Reduction.AUTO,
                 name: str = "iou_loss"):

        super(IoULoss, self).__init__(name=name, reduction=reduction)
        self.mode = mode

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        if not y_pred.dtype.is_floating:
            y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, y_pred.dtype)
        giou = tf.squeeze(compute_iou(y_pred, y_true, mode=self.mode, return_matrix=True))

        return 1 - giou
