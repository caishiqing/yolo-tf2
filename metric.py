import tensorflow as tf
from utils import batch_box_filter, compute_iou


class MAP(tf.keras.metrics.Mean):
    def __init__(self,
                 score_threshold=0.7,
                 iou_threshold=0.5,
                 tp_threshold=0.5,
                 top_k=10,
                 **kwargs):

        kwargs['name'] = f'MAP@{top_k}'
        super(MAP, self).__init__(**kwargs)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.tp_threshold = tp_threshold
        self.top_k = top_k

        self.zero = tf.constant([0.0], dtype=tf.float32)
        self.one = tf.constant([1.0], dtype=tf.float32)
        self.tp, self.fp = None, None

    def update_state(self, ground_truth, predictions, sample_weight=None):
        # shape: (batch, max_num_boxes, 4 + 1 + num_classes)
        gt_mask = self.compute_mask(ground_truth)
        gt_bbox = ground_truth[:, :, :4]
        gt_class = tf.argmax(ground_truth[:, :, 5:], axis=-1)

        pred = batch_box_filter(predictions,
                                score_threshold=self.score_threshold,
                                iou_threshold=self.iou_threshold,
                                max_num_boxes=self.top_k)
        pred_mask = self.compute_mask(pred)
        pred_bbox = pred[:, :, :4]
        pred_score = pred[:, :, 4]
        pred_class = tf.argmax(pred[:, :, 5:], axis=-1)

        mask = tf.logical_and(gt_mask[:, :, tf.newaxis], pred_mask[:, tf.newaxis, :])
        iou_matrix = compute_iou(gt_bbox, pred_bbox, mode="iou", return_matrix=True)
        class_comp = tf.equal(gt_class[:, :, tf.newaxis], pred_class[:, tf.newaxis, :])
        tp_score = tf.where(tf.greater(iou_matrix, self.tp_threshold) & class_comp & mask, iou_matrix, 0)
        tp_score = tf.where(tf.equal(tp_score, tf.reduce_max(tp_score, -1, keepdims=True)), tp_score, 0)
        tp = tf.reduce_any(tf.greater(tp_score, 0), axis=1)
        fp = tf.logical_not(tp)

        confidence = tf.boolean_mask(pred_score, pred_mask)
        cls = tf.boolean_mask(pred_class, pred_mask)
        tp = tf.boolean_mask(tp, pred_mask)
        fp = tf.boolean_mask(fp, pred_mask)

        indices = tf.argsort(confidence, direction="DESCENDING")
        confidence = tf.gather(confidence, indices)
        cls = tf.gather(cls, indices)
        self.tp = tf.gather(tp, indices)
        self.fp = tf.gather(fp, indices)
        cls_mask = tf.equal(tf.unique(cls)[:, tf.newaxis], cls[tf.newaxis, :])

        ap = self.compute_ap(cls_mask)
        return tf.reduce_mean(ap)

    def compute_mask(self, x):
        return tf.reduce_any(tf.not_equal(x, 0), axis=-1)

    def _compute_ap(self, cls_mask):
        precision, recall = self._compute_pr(self.tp, self.fp, cls_mask)
        mrec = tf.concat([self.zero, recall, self.one], axis=-1)
        mpre = tf.concat([self.one, precision, self.zero], axis=-1)
        # Compute the precision envelope
        mpre = tf.reverse(cummax(tf.reverse(mpre, axis=[-1])), axis=[-1])
        # points where x axis (recall) changes
        indices = tf.squeeze(tf.where(mrec[1:] != mrec[:-1]))
        ap = tf.reduce_sum((tf.gather(mrec, indices+1) - tf.gather(mrec, indices))
                           * tf.gather(mpre, indices+1))
        return ap

    def _compute_pr(self, tp, fp, cls_mask=None):
        if cls_mask is not None:
            tp = tf.boolean_mask(tp, cls_mask)
            fp = tf.boolean_mask(fp, cls_mask)

        acc_tp = tf.cumsum(tf.cast(tp, tf.float32))
        acc_fp = tf.cumsum(tf.cast(fp, tf.float32))
        num_gt = tf.reduce_sum(tf.cast(cls_mask, tf.float32))
        precision = tf.math.divide_no_nan(acc_tp, acc_tp + acc_fp)
        recall = tf.math.divide_no_nan(acc_tp, num_gt)
        return precision, recall

    @tf.function
    def compute_ap(self, cls_mask):
        ap = tf.map_fn(self._compute_ap, cls_mask)
        return ap


def tf_while_condition(x, loop_counter):
    return tf.not_equal(loop_counter, 0)


def tf_while_body(x, loop_counter):
    loop_counter -= 1
    y = tf.concat(([x[0]], x[:-1]), axis=0)
    new_x = tf.maximum(x, y)
    return new_x, loop_counter


def cummax(x):
    cumulative_max, _ = tf.while_loop(cond=tf_while_condition,
                                      body=tf_while_body,
                                      loop_vars=(x, tf.shape(x)[0]))
    return cumulative_max


if __name__ == "__main__":
    b1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    b2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
