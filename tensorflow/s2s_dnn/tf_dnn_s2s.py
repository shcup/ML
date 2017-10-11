#!/usr/bin/env python
# -*- coding: utf-8 -*-

#source ~/anaconda/bin/activate tensorflow

import numpy as np
import tensorflow as tf
import math
import linecache
import time
import sys

class TFDnnS2S(object):

  def __init__(self):
    self.learning_rate = 0.01
    self.max_steps = 100
    self.batch_size = 10
    self.display_step = 1
    self.embedding_size = 98
    self.n_hidden_1 = 256
    self.n_hidden_2 = 128
    self.n_classes = 0
    self.max_window_size = 99
    self.seed_seller_size = 0
    self.seed_seller_id_dict = {}
    self.label_seller_size = 0
    self.label_seller_id_dict = {}
    self.training_epochs = 10
    self.embedding = {}
    self.weights = {}
    self.biases = {}

    self.emb_mask = None
    self.word_num = None
    self.x_batch = None
    self.y_batch = None

    self.vector = None


  def init_data(self, training_data, word2vec):
    
    idx = 0
    idx1 = 0
    for line in open(training_data):
      sp = line.strip().split('\t')
      if len(sp) != 2:
        continue
      history,target = sp
      histories = history.split(',')
      for h in histories:
        if h not in self.seed_seller_id_dict:
          self.seed_seller_id_dict[h]=idx
          idx = idx + 1
      if target not in self.label_seller_id_dict:
        self.label_seller_id_dict[target] = idx1
        idx1 = idx1 + 1

    print "Finish loading the click data," + str(idx) + " sellers."
    self.seed_seller_size = len(self.seed_seller_id_dict)
    self.n_classes = len(self.label_seller_id_dict)

    return

    for line in open(word2vec):
      seller_id, vec = line.split('\t', 1)
      if seller_id in self.seller_id_dict:
        idx = self.seller_id_dict[seller_id]
        vec = np.fromstring(vec, dtype=np.float32, sep=' ')
        self.vector[idx]=vec
    print "Finish loading the word2vec data"
          


  def read_data(self, pos, batch_size, data_lst):
    batch = data_lst[pos:pos + batch_size]
    x = np.zeros((batch_size, self.max_window_size))
    mask = np.zeros((batch_size, self.max_window_size))
    y = []
    word_num = np.zeros((batch_size))
    line_no = 0
    for line in batch:
      seed,label = line.strip().split('\t')
      seeds = seed.split(',')
      y.append(self.label_seller_id_dict[label])
      col_no = 0
      for i in seeds:
        if i in self.seed_seller_id_dict:
          x[line_no][col_no] = self.seed_seller_id_dict[i]
          mask[line_no][col_no] = 1
          col_no += 1
          if col_no >= self.max_window_size:
            break
          word_num[line_no] = col_no
      line_no += 1
    return x, np.array(y).reshape(batch_size, 1), mask.reshape(batch_size, self.max_window_size, 1), word_num.reshape(batch_size, 1)

  def create_data_for_infer(self):
    print self.seed_seller_size
    print self.max_window_size
    batch = [x for x in range(self.seed_seller_size)]
    x = np.zeros((self.seed_seller_size, self.max_window_size))
    mask = np.zeros((self.seed_seller_size, self.max_window_size))
    
    word_num = np.zeros((self.seed_seller_size))
    line_no = 0
    for line in batch:
      x[line_no][0] = line
      mask[line_no][0] = 1
      word_num[line_no] = 1
      line_no += 1

    return x, mask.reshape(self.seed_seller_size, self.max_window_size, 1), word_num.reshape(self.seed_seller_size, 1)


  def build_graph(self):

   # embedding layyer
    self.embedding = {
        #'input':tf.Variable(self.vector)
        'input':tf.Variable(tf.random_uniform([self.seed_seller_size+1, self.embedding_size], -1.0, 1.0))
        # 'output':tf.Variable(tf.random_uniform([len(label_dict)+1, emb_size], -1.0, 1.0))
    }
    
   # hidden layers
    self.weights = {
        'h1': tf.Variable(tf.random_normal([self.embedding_size, self.n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
        'out': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_classes]))
    }
    self.biases = {
        'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
        'out': tf.Variable(tf.random_normal([self.n_classes]))
    }

    self.emb_mask = tf.placeholder(tf.float32, shape=[None, self.max_window_size, 1])
    self.word_num = tf.placeholder(tf.float32, shape=[None, 1])
    # the input user sp history
    self.x_batch = tf.placeholder(tf.int32, shape=[None, self.max_window_size])
    # current user click seller
    self.y_batch = tf.placeholder(tf.int64, [None, 1])

    # get all the embeding for sp user history
    input_embedding = tf.nn.embedding_lookup(self.embedding['input'], self.x_batch)
    # mean all the embedding for sp user history
    project_embedding = tf.div(tf.reduce_sum(tf.multiply(input_embedding,self.emb_mask), 1),self.word_num)

    # layer 1
    layer_1 = tf.nn.relu(
              tf.add(tf.matmul(project_embedding, self.weights['h1']), self.biases['b1']))
    dlayer_1 = tf.nn.dropout(layer_1, 0.5)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))

    return layer_2, self.y_batch


  def build_loss(self, pred, y_batch):
      nce_weights = tf.Variable(
        tf.truncated_normal([self.n_classes, self.n_hidden_2],
        stddev=1.0 / math.sqrt(self.n_hidden_2)))
      nce_biases = tf.Variable(tf.zeros([self.n_classes]))

      loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=y_batch,
                                         inputs=pred,
                                         num_sampled=10,
                                         num_classes=self.n_classes))

      cost = tf.reduce_sum(loss) / self.batch_size
      out_layer = tf.matmul(pred, tf.transpose(nce_weights)) + nce_biases
      return cost, out_layer

  def build_optimizer(self,cost):
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
    return optimizer

  def run(self):  
  
    train_file='click_data.txt'
    self.init_data(train_file, 'word2vec.txt')

    pred,y_batch=self.build_graph()
    cost, out_layer=self.build_loss(pred, y_batch)
    optimizer = self.build_optimizer(cost)
    train_lst = linecache.getlines(train_file)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.log_device_placement = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(config=config) as sess:
      sess.run(init)

      start_time = time.time()
      total_batch = int(len(train_lst) / self.batch_size)
      print("total_batch of training data: ", total_batch)
      for epoch in range(self.training_epochs):
        avg_cost = 0.
        for i in range(total_batch):
          x, y, batch_mask, word_number = self.read_data(i * self.batch_size, self.batch_size, train_lst)
          _,c = sess.run([optimizer, cost], feed_dict={self.x_batch: x, self.emb_mask: batch_mask, self.word_num: word_number, self.y_batch: y})
          avg_cost += c / total_batch

            #print ("Batch:", "%04d" % (i), "cost=", \
            #      "{:.9f}".format(c))

        if epoch % self.display_step == 0:
          print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost))
      y = tf.nn.softmax(out_layer)
      tf.add_to_collection('predictor', y)
      tf.add_to_collection('x_batch', self.x_batch)
      tf.add_to_collection('emb_mask', self.emb_mask)
      tf.add_to_collection('word_num', self.word_num)
      saver.save(sess, './model/model.ckpt')

  def restore(self):
    train_file='click_data_sample.txt'
    self.init_data(train_file, 'word2vec.txt')

    with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
      new_saver.restore(sess, './model/model.ckpt')
      # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
      y = tf.get_collection('predictor')[0]
      x_batch = tf.get_collection('x_batch')[0]
      emb_mask = tf.get_collection('emb_mask')[0]
      word_num = tf.get_collection('word_num')[0]

      graph = tf.get_default_graph()

      # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
      #x_batch = graph.get_operation_by_name('self.x_batch').outputs[0]
      #emb_mask = graph.get_operation_by_name('self.emb_mask').outputs[0]
      #word_num = graph.get_operation_by_name('self.word_num').outputs[0]

      # 使用y进行预测  
      x, batch_mask, word_number = self.create_data_for_infer()
      print len(x)
      print len(batch_mask)
      print len(word_number)
      predict_op = tf.nn.top_k(y, 200, False)
      predict_result = sess.run(predict_op, feed_dict={x_batch:x, emb_mask: batch_mask, word_num: word_number})
      print predict_result

exp = TFDnnS2S()
#exp.restore()
exp.run()
