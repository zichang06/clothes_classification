# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

resnet = nets.resnet_v2


def grasp_net(images, is_training=True, lmbda=0.0, base_model=''):
    num_classes = 18
    if base_model.startswith('resnet'):
        with slim.arg_scope(resnet.resnet_arg_scope()):
            if base_model == 'resnet200':
                net, _ = resnet.resnet_v2_200(images, num_classes=1024, is_training=is_training, reuse=tf.AUTO_REUSE)
            elif base_model == 'resnet50':
                net, _ = resnet.resnet_v2_50(images, num_classes=1024, is_training=is_training, reuse=tf.AUTO_REUSE)
            else:
                raise ValueError('Unexpected base model')
        with tf.variable_scope('extra', reuse=tf.AUTO_REUSE):
            net = slim.fully_connected(net, num_classes, tf.sigmoid)
            net = tf.squeeze(net)
        return net
    else:
        raise ValueError('Unexpected base model')


def custom_loss_function(logits, theta_labels, class_labels):
    """
    Custom loss function according to the paper.

    :param logits: should be shape of [batch_size, 18]
    :param theta_labels: each denoted by an one-hot vector
    :param class_labels: each denoted by 0 or 1, should be converted to float
    :return: reduce mean of loss for a batch
    """
    class_labels = tf.cast(class_labels, tf.float32)
    filtered_scores = tf.reduce_sum(logits*theta_labels, 1)
    # Reshape the scores, such that it shares the shape with labels
    filtered_scores = tf.reshape(filtered_scores, [-1, 1])
    clipped_scores = tf.clip_by_value(filtered_scores, 1e-5, 1.0-1e-5)
    entropys = - class_labels * tf.log(clipped_scores)\
               - (1-class_labels) * tf.log(1-clipped_scores)
    loss = tf.reduce_mean(entropys)

    return loss


def get_metrics(logits, theta_labels, class_labels):
    """
    Function to calculate accuracy.

    :param logits: should be shape of [batch_size, 18]
    :param theta_labels: each denoted by an one-hot vector
    :param class_labels: each denoted by 0 or 1
    :return: accuray, precision and recall
    """
    # Extract outputs of coresponding angles
    angle_outputs = tf.reduce_sum(theta_labels * logits, 1)
    # angle_outputs = tf.Print(angle_outputs, [angle_outputs], 'prediction: ', summarize=10)
    # Convert class labels to 1-D array
    class_labels = tf.cast(tf.squeeze(class_labels), tf.int32)
    # class_labels = tf.Print(class_labels, [class_labels], 'label: ', summarize=10)
    # Get positive and negative indexes of labels
    # Remember to cast bool to int for later computing
    p_label_indexes = tf.cast(tf.equal(class_labels, tf.ones(class_labels.shape, dtype=tf.int32)), tf.int32)
    n_label_indexes = tf.cast(tf.equal(class_labels, tf.zeros(class_labels.shape, dtype=tf.int32)), tf.int32)
    threshold = tf.constant(0.5, shape=angle_outputs.shape)
    # Among the outputs of angles, those >= threshold will be considered positive and vice versa
    p_logits_indexes = tf.cast(tf.greater_equal(angle_outputs, threshold), tf.int32)
    n_logits_indexes = tf.cast(tf.less(angle_outputs, threshold), tf.int32)
    # Finally, we can calculate numbers of true positive and true negative
    num_true_p = tf.reduce_sum(p_label_indexes * p_logits_indexes)  # 该batch中正样本被判断正确的个数
    num_true_n = tf.reduce_sum(n_label_indexes * n_logits_indexes)  # 该batch中负样本被判断正确的个数
    # Compute number of false positive and false negative
    num_positive_labels = tf.reduce_sum(p_label_indexes)  # 正样本的个数
    num_negative_labels = tf.reduce_sum(n_label_indexes)  # 负样本的个数
    num_false_p = num_positive_labels - num_true_n  # 正样本中被判断错误的个数（误认为不可抓）
    num_false_n = num_negative_labels - num_true_p  # 负样本中被判断错误的个数（误认为可抓）
    # num_true_p = tf.Print(num_true_p, [num_true_p], 'num true p')
    # num_true_n = tf.Print(num_true_n, [num_true_n], 'num true n')
    # num_false_p = tf.Print(num_false_p, [num_false_p], 'num false p')
    # num_false_n = tf.Print(num_false_n, [num_false_n], 'num false n')
    # Compute accuracy, precision and recall
    accuracy = (num_true_p + num_true_n) / (num_true_n + num_true_p + num_false_n + num_false_p)
    precison = num_true_p / (num_true_p + num_false_p)
    recall = num_true_p / (num_true_p + num_false_n)

    return accuracy, precison, recall

'''
如果模型足够好，则能够判断每一个角度是否可抓取
准确率accuracy: 正负样本对应角度被判断正确的个数/总样本数
因为验证集中负样本个数特别多，导致有时一个batch基本为负样本，如果所有角度值都预测特别低，
则该准确率将因此较高。易出现准确率时而很高，时而下降的震荡。
精度：正样本被判断正确的个数/所有预测为正的样本个数，越接近1越好，说明预测角度可抓确实是可以抓取的
召回率：正样本被正确判断的个数/实际所有正样本个数，越接近1越好，说明可以抓取的角度确实是可以预测出来的

'''