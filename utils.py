from tensorflow.keras import backend
import tensorflow as tf


def mish(x):
    return x * tf.math.tanh(tf.math.log(1 + tf.exp(x)))


def iou(boxes: tf.Tensor, candidates: tf.Tensor):
    """Intersection over Union

    Args:
        boxes (tf.Tensor): N * (left x, top y, w, h) for references.
        candidates (tf.Tensor): M * (left x, top y, w, h) for candidates.

    Returns:
        tf.Tensor: N * M iou score matrix.
    """
    bbox_tl = boxes[:, :2]
    bbox_br = boxes[:, :2] + boxes[:, 2:]
    cand_tl = candidates[:, :2]
    cand_br = candidates[:, :2] + candidates[:, 2:]

    tl = tf.maximum(bbox_tl[:, tf.newaxis, :], cand_tl[tf.newaxis, :, :])
    br = tf.minimum(bbox_br[:, tf.newaxis, :], cand_br[tf.newaxis, :, :])
    wh = tf.maximum(0.0, br-tl)
    area_intersection = tf.reduce_prod(wh, axis=-1)

    area_bbox = tf.reduce_prod(boxes[:, 2:], axis=-1)
    area_cand = tf.reduce_prod(candidates[:, 2:], axis=-1)
    area_union = area_bbox[:, tf.newaxis] + area_cand[tf.newaxis, :] - area_intersection
    return tf.math.divide_no_nan(area_intersection, area_union)


class BoxFilter(object):
    def __init__(self,
                 score_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 max_num_boxes: int = 100,
                 padding: str = None):

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_num_boxes = max_num_boxes
        self.padding = padding

    def __call__(self, predictions):
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        indices = tf.image.non_max_suppression(boxes,
                                               scores,
                                               self.max_num_boxes,
                                               self.iou_threshold,
                                               self.score_threshold,
                                               name="nms")

        if self.padding is not None:
            _pad = -tf.ones(self.max_num_boxes - tf.shape(indices),
                            dtype=indices.dtype)

            if self.padding == "pre":
                indices = tf.concat([_pad, indices], axis=-1)
            elif self.padding == "post":
                indices = tf.concat([indices, _pad], axis=-1)

        return tf.gather(predictions, indices)


@tf.function
def batch_box_filter(predictions,
                     score_threshold=0.5,
                     iou_threshold=0.5,
                     max_num_boxes=100):

    return tf.map_fn(BoxFilter(score_threshold,
                               iou_threshold,
                               max_num_boxes,
                               padding="post"),
                     predictions)


def compute_iou(b1: tf.Tensor, b2: tf.Tensor, mode="iou"):
    ndim1 = backend.ndim(b1)
    ndim2 = backend.ndim(b2)
    assert ndim1 == ndim2
    if ndim1 > 1 and tf.shape(b1)[-2] != tf.shape(b2)[-2]:
        b1 = b1[..., tf.newaxis, :]
        b2 = b2[..., tf.newaxis, :, :]

    zero = tf.convert_to_tensor(0.0, b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
    return giou


if __name__ == "__main__":
    x1y1 = tf.random.uniform(shape=(10, 100, 2))
    x2y2 = x1y1 + tf.random.uniform(shape=(10, 100, 2))
    b1 = tf.concat([x1y1, x2y2], axis=-1)

    x1y1 = tf.random.uniform(shape=(10, 50, 2))
    x2y2 = x1y1 + tf.random.uniform(shape=(10, 50, 2))
    b2 = tf.concat([x1y1, x2y2], axis=-1)

    iou = compute_iou(b1, b2)
    print(iou.shape)
