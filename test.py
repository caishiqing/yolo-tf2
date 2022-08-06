import tensorflow as tf


def tf_while_condition(x, loop_counter):
    return tf.not_equal(loop_counter, 0)


def tf_while_body(x, loop_counter):
    loop_counter -= 1
    y = tf.concat(([x[0]], x[:-1]), axis=0)
    new_x = tf.maximum(x, y)
    return new_x, loop_counter


x = tf.constant([0, 2, 5, 3, 8, 1, 7])

cumulative_max, _ = tf.while_loop(cond=tf_while_condition,
                                  body=tf_while_body,
                                  loop_vars=(x, tf.shape(x)[0]))

print(cumulative_max)
