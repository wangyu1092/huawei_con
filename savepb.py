
import tensorflow as tf

def main(x_in, y_in, load):


    X = tf.placeholder(tf.float32, name='X', shape=[None, b])
    Y = tf.placeholder(tf.float32, name='Y', shape=[None, 1])

    pre = Y
    # pred = train_line_model(X, 1)

    # bias = tf.Variable(tf.fill(pred.get_shape().as_list(), -1), name="bias")
    # preds = tf.matmul(pred, tf.cast(bias, tf.float32))
    # pcrr = CaculatePcrr(Y, pred)
    # print(final_output.shape)

    # loss = tf.reduce_mean(tf.square(Y - pred, name="loss"))

    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    init_op = tf.global_variables_initializer()

    # train_total = []
    # val_total = []

    # with tf.Session(graph=tf.Graph()) as sess:
    #     meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], checkpoint)
    #     signature = meta_graph_def.signature_def
    #
    #     in_tensor_name = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['myInput'].name
    #     out_tensor_name = signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['myOutput'].name
    #     x = sess.graph.get_tensor_by_name(in_tensor_name)
    #     y = sess.graph.get_tensor_by_name(out_tensor_name)
    #
    #     for i in range(vn_batchs):
    #
    #         batch_xs, batch_ys = fetch_batch_data(x_val, y_val,batch_size, i)
    #         pre = sess.run(y,feed_dict={x: batch_xs})
    #         print("predict: {0}, y_true: {1}".format(pre[200], batch_ys[200]))


    with tf.Session() as sess:

        sess.run(init_op)
        writer = tf.summary.FileWriter('graphs', sess.graph)
        # pre = list()
        pr = sess.run([pre], feed_dict={X: x_in, Y: y_in})
        # bias = tf.Variable(tf.fill(pred.get_shape().as_list(), -1), name="bias")
        # preds = tf.matmul(pred, tf.cast(bias, tf.float32))
        writer.close()
        tf.saved_model.simple_save(sess, load, inputs={"myInput": X}, outputs={"myOutput": pr})


if __name__ == '__main__':
    main()
