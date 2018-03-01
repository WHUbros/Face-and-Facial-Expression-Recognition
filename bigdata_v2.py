import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
import time

if __name__ == "__main__":
    Reader=csv.reader(open('./fer2014.csv','r'))
    data = []
    label = []
    y_test = []
    X_test = []
#  load data
    for i, row in enumerate(Reader):
        if i >= 1:
            if str(row[2]) == "Training":
                temp_list = []
                for pixel in row[1].split( ):
                    temp_list.append(int(pixel))

                temp_list = np.array(temp_list)
                temp_list = np.reshape(temp_list, [48,48])
                data.append(temp_list)
                label.append(int(row[0]))
            elif str(row[2]) == "PublicTest" :
                temp_list = []
                for pixel in row[1].split( ):
                    temp_list.append(int(pixel))

                temp_list = np.array(temp_list)
                temp_list = np.reshape(temp_list, [48,48])
                X_test.append(temp_list)
                y_test.append(int(row[0]))
    data = np.array(data)
    label = np.array(label)

####### preprocess
    num_data = len(data)
    num_validation = 2000
    num_training = num_data - num_validation
    index = np.linspace(0,num_data - 1, data)

    train = np.random.choice(num_data - 1, num_training, replace=False)
    val = np.setdiff1d(index,train)
    val = np.array(val,dtype=int)
    X_val = data[val]
    y_val = label[val]

    X_train = data[train]
    y_train = label[train]

    mean_image = np.mean(X_train, axis=0)
    X_train = X_train.astype(np.float32) - mean_image.astype(np.float32)
    X_val = X_val.astype(np.float32) - mean_image

    X_train = X_train.reshape([-1,48,48])/255
    X_val = X_val.reshape([-1,48,48])/255

    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)

######
    class conv_layer(object):
        def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, index=0):
            print("len(input_x.shape),input_x.shape[1],input_x.shape[2],input_x.shape[3],in_channel",len(input_x.shape),input_x.shape[1],input_x.shape[2],input_x.shape[3],in_channel)
            assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

            with tf.variable_scope('conv_layer_%d' % index):
                with tf.name_scope('conv_kernel'):
                    w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                    weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                             initializer=tf.contrib.layers.xavier_initializer())
                    self.weight = weight

                with tf.variable_scope('conv_bias'):
                    b_shape = [out_channel]
                    bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                           initializer=tf.contrib.layers.xavier_initializer())
                    self.bias = bias

                # strides [1, x_movement, y_movement, 1]
                conv_out = tf.nn.conv2d(input_x, weight, strides=[1, 1, 1, 1], padding="SAME")
                cell_out = tf.nn.relu(conv_out + bias)

                self.cell_out = cell_out

                tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
                tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)

        def output(self):
            return self.cell_out

    class max_pooling_layer(object):
        def __init__(self, input_x, k_size, padding="SAME"):

            with tf.variable_scope('max_pooling'):
                # strides [1, k_size, k_size, 1]
                pooling_shape = [1, k_size, k_size, 1]
                cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                          ksize=pooling_shape, padding=padding)
                self.cell_out = cell_out

        def output(self):
            return self.cell_out



    class norm_layer(object):
        def __init__(self, input_x):
            with tf.variable_scope('batch_norm'):
                mean, variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
                cell_out = tf.nn.batch_normalization(input_x,
                                                     mean,
                                                     variance,
                                                     offset=None,
                                                     scale=None,
                                                     variance_epsilon=1e-6,
                                                     name=None)
                self.cell_out = cell_out

        def output(self):
            return self.cell_out


    class fc_layer(object):
        def __init__(self, input_x, in_size, out_size, rand_seed, activation_function=tf.nn.relu, index=0):
            with tf.variable_scope('fc_layer_%d' % index):
                with tf.name_scope('fc_kernel'):
                    w_shape = [in_size, out_size]
                    weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                             initializer=tf.contrib.layers.xavier_initializer())
                    self.weight = weight

                with tf.variable_scope('fc_kernel'):
                    b_shape = [out_size]
                    bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                           initializer=tf.contrib.layers.xavier_initializer())
                    self.bias = bias
                ####dropout
                input_drop = tf.nn.dropout(input_x, keep_prob=0.5)
                cell_out = tf.add(tf.matmul(input_drop, weight), bias)
                if activation_function is not None:
                    cell_out = activation_function(cell_out)

                self.cell_out = cell_out

                tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
                tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)

        def output(self):
            return self.cell_out


    def my_LeNet(input_x, input_y,
              img_len=48, channel_num=3, output_size=7,
              conv_featmap=[6, 16], fc_units=[84],
              conv_kernel_size=[5, 5], pooling_size=[2, 2],
              l2_norm=0.01, seed=235):

        assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

        # conv layer*3
        conv_layer_0_1 = conv_layer(input_x=input_x,
                                  in_channel=channel_num,
                                  out_channel=conv_featmap[0],
                                  kernel_shape=conv_kernel_size[0],
                                  rand_seed=seed,index=0)


        pooling_layer_0 = max_pooling_layer(input_x=conv_layer_0_1.output(),
                                            k_size=pooling_size[0],
                                            padding="VALID")

        pool_shape = pooling_layer_0.output().get_shape()




        ######### conv layer *3
        conv_layer_1_1 = conv_layer(input_x=pooling_layer_0.output(),
                                  in_channel=pool_shape[3],
                                  out_channel=conv_featmap[1],
                                  kernel_shape=conv_kernel_size[0],
                                  rand_seed=seed,index=1)


        pooling_layer_1 = max_pooling_layer(input_x=conv_layer_1_1.output(),
                                            k_size=pooling_size[1],
                                            padding="VALID")

        pool1_shape = pooling_layer_1.output().get_shape()




        #######################################

        conv_layer_2_1 = conv_layer(input_x=pooling_layer_1.output(),
                                  in_channel=pool1_shape[3],
                                  out_channel=conv_featmap[2],
                                  kernel_shape=conv_kernel_size[0],
                                  rand_seed=seed,index=2)


        pooling_layer_2 = max_pooling_layer(input_x=conv_layer_2_1.output(),
                                            k_size=pooling_size[1],
                                            padding="VALID")

    #     pool2_shape = pooling_layer_2.output().get_shape()

        ########################################3


        # flatten
        pool_shape1 = pooling_layer_2.output().get_shape()
        img_vector_length = pool_shape1[1].value * pool_shape1[2].value * pool_shape1[3].value
        flatten = tf.reshape(pooling_layer_2.output(), shape=[-1, img_vector_length])

        # fc layer
        fc_layer_0 = fc_layer(input_x=flatten,
                              in_size=img_vector_length,
                              out_size=fc_units[0],
                              rand_seed=seed,
                              activation_function=None,
                              index=0)

        fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                              in_size=fc_units[1],
                              out_size=output_size,
                              rand_seed=seed,
                              activation_function=None,
                              index=1)

        # saving the parameters for l2_norm loss
        conv_w = [conv_layer_0_1.weight,\
                 conv_layer_1_1.weight,\
                 conv_layer_2_1.weight]
        fc_w = [fc_layer_0.weight, fc_layer_1.weight]

        # loss
        with tf.name_scope("loss"):
            l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
            l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])

            label = tf.one_hot(input_y, 7)
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_1.output()),
                name='cross_entropy')
            loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

            tf.summary.scalar('LeNet_loss', loss)

        layerout = [conv_layer_0_1.output(), \
                    conv_layer_1_1.output(),\
                    conv_layer_2_1.output()]
        return layerout, fc_layer_1.output(), loss



    def cross_entropy(output, input_y):
        with tf.name_scope('cross_entropy'):
            label = tf.one_hot(input_y, 7)
            ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

        return ce



    def train_step(loss, learning_rate=1e-3):
        with tf.name_scope('train_step'):
            step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        return step



    def evaluate(output, input_y):
        with tf.name_scope('evaluate'):
            pred = tf.argmax(output, axis=1)
            error_num = tf.count_nonzero(pred - input_y, name='error_num')
            tf.summary.scalar('LeNet_error_num', error_num)
        return error_num

    def predict(output):
        with tf.name_scope('predict'):
            pred = tf.argmax(output, axis=1)
        return pred    


    def my_training(X_train, y_train, X_val, y_val,x_test, 
                 conv_featmap=[6],
                 fc_units=[84],
                 conv_kernel_size=[5],
                 pooling_size=[2],
                 l2_norm=0.01,
                 seed=235,
                 learning_rate=1e-2,
                 epoch=20,
                 batch_size=250,
                 verbose=False,
                 pre_trained_model=None):
        print("Building my LeNet. Parameters: ")
        print("conv_featmap={}".format(conv_featmap))
        print("fc_units={}".format(fc_units))
        print("conv_kernel_size={}".format(conv_kernel_size))
        print("pooling_size={}".format(pooling_size))
        print("l2_norm={}".format(l2_norm))
        print("seed={}".format(seed))
        print("learning_rate={}".format(learning_rate))

        # define the variables and parameter needed during training
        with tf.name_scope('inputs'):
            xs = tf.placeholder(shape=[None, 48, 48, 1], dtype=tf.float32)
            ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

        layerout, output, loss = my_LeNet(xs, ys,
                             img_len=48,
                             channel_num=1,
                             output_size=7,
                             conv_featmap=conv_featmap,
                             fc_units=fc_units,
                             conv_kernel_size=conv_kernel_size,
                             pooling_size=pooling_size,
                             l2_norm=l2_norm,
                             seed=seed)

        iters = int(X_train.shape[0] / batch_size)# 72
        print('number of batches for training: {}'.format(iters))

        step = train_step(loss)
        eve = evaluate(output, ys)
        pre = predict(output)

        iter_total = 0
        best_acc = 0
        cur_model_name = 'lenet_{}'.format(int(time.time()))

        with tf.Session() as sess:
            merge = tf.summary.merge_all()

            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            # try to restore the pre_trained
            if pre_trained_model is not None:
                try:
                    print("Load the model from: {}".format(pre_trained_model))
                    saver.restore(sess, 'model/{}'.format(pre_trained_model))
                except Exception:
                    print("Load model Failed!")
                    pass

            for epc in range(epoch):
                print("epoch {} ".format(epc + 1))

                for itr in range(iters):
                    iter_total += 1

                    training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                    training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                    _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})
                    loss_list.append(cur_loss)

                    if iter_total % 100 == 0:
                        # do validation
                        valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                        valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                        acc_list.append(valid_acc)
                        if verbose:
                            print('{}/{} loss: {} validation accuracy : {}%'.format(
                                batch_size * (itr + 1),
                                X_train.shape[0],
                                cur_loss,
                                valid_acc))

                        # save the merge result summary
                        writer.add_summary(merge_result, iter_total)

                        # when achieve the best validation accuracy, we store the model paramters
                        if valid_acc > best_acc:
                            featureout = sess.run(layerout, feed_dict ={xs: training_batch_x, ys: training_batch_y})
                            for j, f in enumerate(featureout):
                                Features[j].append(f)

                            print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                            prediction = sess.run([pre],feed_dict={xs: x_test})
                            best_acc = valid_acc
                            saver.save(sess, 'model/{}'.format(cur_model_name))

        print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
        return prediction


    def my_training(X_train, y_train, X_val, y_val,x_test, 
                 conv_featmap=[6],
                 fc_units=[84],
                 conv_kernel_size=[5],
                 pooling_size=[2],
                 l2_norm=0.01,
                 seed=235,
                 learning_rate=1e-2,
                 epoch=20,
                 batch_size=250,
                 verbose=False,
                 pre_trained_model=None):
        print("Building my LeNet. Parameters: ")
        print("conv_featmap={}".format(conv_featmap))
        print("fc_units={}".format(fc_units))
        print("conv_kernel_size={}".format(conv_kernel_size))
        print("pooling_size={}".format(pooling_size))
        print("l2_norm={}".format(l2_norm))
        print("seed={}".format(seed))
        print("learning_rate={}".format(learning_rate))

        # define the variables and parameter needed during training
        with tf.name_scope('inputs'):
            xs = tf.placeholder(shape=[None, 48, 48, 1], dtype=tf.float32)
            ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

        layerout, output, loss = my_LeNet(xs, ys,
                             img_len=48,
                             channel_num=1,
                             output_size=7,
                             conv_featmap=conv_featmap,
                             fc_units=fc_units,
                             conv_kernel_size=conv_kernel_size,
                             pooling_size=pooling_size,
                             l2_norm=l2_norm,
                             seed=seed)

        iters = int(X_train.shape[0] / batch_size)# 72
        print('number of batches for training: {}'.format(iters))

        step = train_step(loss)
        eve = evaluate(output, ys)
        pre = predict(output)

        iter_total = 0
        best_acc = 0
        cur_model_name = 'lenet_{}'.format(int(time.time()))

        with tf.Session() as sess:
            merge = tf.summary.merge_all()

            writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            if pre_trained_model is not None:
                try:
                    print("Load the model from: {}".format(pre_trained_model))
                    saver.restore(sess, 'model/{}'.format(pre_trained_model))
                except Exception:
                    print("Load model Failed!")
                    pass

            for epc in range(epoch):
                print("epoch {} ".format(epc + 1))

                for itr in range(iters):
                    iter_total += 1

                    training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                    training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                    _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})
                    loss_list.append(cur_loss)

                    if iter_total % 100 == 0:
                        # do validation
                        valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                        valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                        acc_list.append(valid_acc)
                        if verbose:
                            print('{}/{} loss: {} validation accuracy : {}%'.format(
                                batch_size * (itr + 1),
                                X_train.shape[0],
                                cur_loss,
                                valid_acc))

                        # save the merge result summary
                        writer.add_summary(merge_result, iter_total)

                        # when achieve the best validation accuracy, store the model paramters
                        # and get the feature of every conv layer
                        if valid_acc > best_acc:
                            featureout = sess.run(layerout, feed_dict ={xs: training_batch_x, ys: training_batch_y})
                            for j, f in enumerate(featureout):
                                Features[j].append(f)

                            print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                            prediction = sess.run([pre],feed_dict={xs: x_test})
                            best_acc = valid_acc
                            saver.save(sess, 'model/{}'.format(cur_model_name))

        print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
        return prediction



    X_train = np.reshape(X_train, [-1,48,48,1])
    X_val = np.reshape(X_val,[-1,48,48,1])
    X_test = np.reshape(X_test, [-1,48,48,1])

#######
    loss_list = []
    acc_list = []
    Features = [[] for _ in range(9)]
    tf.reset_default_graph()
    pre = my_training(X_train, y_train, X_val, y_val, X_test,
             conv_featmap=[32,64,128],
             fc_units=[1536,384],
             conv_kernel_size=[5,5,5],
             pooling_size=[2,2,2],
             l2_norm=0.01,
             seed=235,
             learning_rate=5e-2,
             epoch=100,
             batch_size=200,
             verbose=True,
             pre_trained_model=None)

    np.save("./Features",np.array(Features))
    np.save("./loss_list", np.array(loss_list))
    np.save("./acc_list", np.array(acc_list))


