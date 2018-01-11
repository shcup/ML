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
    self.batch_size = 500
    self.display_step = 1
    self.embedding_size = 100
    self.n_hidden_1 = 128
    self.n_hidden_2 = 64
    self.n_classes = 1000000
    self.max_window_size = 100
    self.seed_seller_size = 1000000
    self.seed_seller_id_dict = {}
    self.label_seller_size = 0
    self.label_seller_id_dict = {}
    self.training_epochs = 3
    self.embedding = {}
    self.weights = {}
    self.biases = {}

    self.emb_mask = None
    self.word_num = None
    self.x_batch = None
    self.y_batch = None

    self.vector = None         

  def preprocess_data(self, raw_data, processed_data, seed_seller_mapping, target_seller_mapping):

    seed_seller_id_dict = {}
    label_seller_id_dict = {} 

    for line in open(raw_data):
      seeds, label = line.strip().split('\t')
      sp = seeds.split(',')
      for s in sp:
        if s not in seed_seller_id_dict:
          seed_seller_id_dict[s] = 1
        else:
          seed_seller_id_dict[s] = seed_seller_id_dict[s] + 1

      if label not in label_seller_id_dict:
        label_seller_id_dict[label] = 1
      else:
        label_seller_id_dict[label] = label_seller_id_dict[label] + 1

    sorted_seed_seller = sorted(seed_seller_id_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    select_seed_seller = sorted_seed_seller[0:1000000]

    sorted_label_seller = sorted(label_seller_id_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    select_label_seller = sorted_label_seller[0:1000000]

    seed_seller_encode = {}
    idx = 0
    for k,v in select_seed_seller:
      seed_seller_encode[k] = idx
      idx += 1
    label_seller_encode = {}
    idx = 0
    for k,v in select_label_seller:
      label_seller_encode[k] = idx
      idx += 1


    print "Dump seed id mapping"
    f_seed = open(seed_seller_mapping,"w")
    for k,v in select_seed_seller:
      f_seed.write(k+"\t"+str(v)+"\t"+str(seed_seller_encode[k])+"\n")
    f_seed.close()
    f_target = open(target_seller_mapping,"w")
    for k,v in select_label_seller:
      f_target.write(k+"\t"+str(v)+"\t"+str(label_seller_encode[k])+"\n")
    f_target.close()

    print "Start process data"
    f_out = open(processed_data,"w")
    for line in open(raw_data):
      seeds, label = line.strip().split('\t')
      sp = seeds.split(',')
      seed_array = []
      for s in sp:
        if s in seed_seller_encode:
          seed_array.append(str(seed_seller_encode[s]))

      if len(seed_array) > 0 and label in label_seller_encode:
        label_id = str(label_seller_encode[label])
        f_out.write(','.join(seed_array)+"\t"+label_id+"\n")
    f_out.close() 



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
      y.append(int(label))
      col_no = 0
      for i in seeds:
        x[line_no][col_no] = int(i) 
        mask[line_no][col_no] = 1
        col_no += 1
        if col_no >= self.max_window_size:
          break
      word_num[line_no] = col_no
      line_no += 1
    return x, np.array(y).reshape(batch_size, 1), mask.reshape(batch_size, self.max_window_size, 1), word_num.reshape(batch_size, 1)

  def create_data_for_infer(self, pos, batch_size, data_lst):
    batch = data_lst[pos:pos+batch_size]
    x = np.zeros((batch_size, self.max_window_size))
    mask = np.zeros((batch_size, self.max_window_size))
    
    word_num = np.zeros((batch_size))
    line_no = 0
    for line in batch:
      x[line_no][0] = line
      mask[line_no][0] = 1
      word_num[line_no] = 1
      line_no += 1

    return x, mask.reshape(batch_size, self.max_window_size, 1), word_num.reshape(batch_size, 1)


  def build_graph(self):

   # embedding layyer
    self.embedding = {
        #'input':tf.Variable(self.vector)
        'input':tf.Variable(tf.random_uniform([self.seed_seller_size+1, self.embedding_size], -1.0, 1.0), name='embedding')
        # 'output':tf.Variable(tf.random_uniform([len(label_dict)+1, emb_size], -1.0, 1.0))
    }
    
   # hidden layers
    self.weights = {
        'h1': tf.Variable(tf.random_normal([self.embedding_size, self.n_hidden_1]), name='h1'),
        'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name='h2'),
        #'out': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_classes]))
    }
    self.biases = {
        'b1': tf.Variable(tf.random_normal([self.n_hidden_1]), name='b1'),
        'b2': tf.Variable(tf.random_normal([self.n_hidden_2]), name='b2'),
        #'out': tf.Variable(tf.random_normal([self.n_classes]))
    }

    self.emb_mask = tf.placeholder(tf.float32, shape=[None, self.max_window_size, 1], name='emb_mask')
    self.word_num = tf.placeholder(tf.float32, shape=[None, 1], name='word_num')
    # the input user sp history
    self.x_batch = tf.placeholder(tf.int32, shape=[None, self.max_window_size], name='x_batch')
    # current user click seller
    self.y_batch = tf.placeholder(tf.int64, [None, 1], name='y_batch')

    # get all the embeding for sp user history
    input_embedding = tf.nn.embedding_lookup(self.embedding['input'], self.x_batch, name='input_embedding')
    # mean all the embedding for sp user history
    project_embedding = tf.div(tf.reduce_sum(tf.multiply(input_embedding,self.emb_mask), 1),self.word_num)

    # layer 1
    layer_1 = tf.nn.relu(
              tf.add(tf.matmul(project_embedding, self.weights['h1']), self.biases['b1']))
    #dlayer_1 = tf.nn.dropout(layer_1, 0.5)
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
  
    train_file='processed_click_data.txt'
    #self.init_data(train_file, 'word2vec.txt')

    pred,y_batch=self.build_graph()
    cost, out_layer=self.build_loss(pred, y_batch)
    optimizer = self.build_optimizer(cost)
    train_lst = linecache.getlines(train_file)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #config.gpu_options.allocator_type="BFC"

    print "TAGGGGG start to session"
    with tf.Session(config=config) as sess:
      sess.run(init)

      print "TAGGGGG finish to init"

      start_time = time.time()
      total_batch = int(len(train_lst) / self.batch_size)
      print("total_batch of training data: ", total_batch, ", start at: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
      for epoch in range(self.training_epochs):
        avg_cost = 0.
        for i in range(total_batch):
          x, y, batch_mask, word_number = self.read_data(i * self.batch_size, self.batch_size, train_lst)
          #print "x"
          #print x
          #print "y"
          #print y
          #print "batch mask"
          #print batch_mask
          #print "word_number"
          #print word_number
          _,c = sess.run([optimizer, cost], feed_dict={self.x_batch: x, self.emb_mask: batch_mask, self.word_num: word_number, self.y_batch: y})
          avg_cost += c / total_batch

            #print ("Batch:", "%04d" % (i), "cost=", \
            #      "{:.9f}".format(c))

        if epoch % self.display_step == 0:
          print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), " Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost))
          sys.stdout.flush()
      y = tf.nn.softmax(out_layer)
      tf.add_to_collection('predictor', y)
      tf.add_to_collection('x_batch', self.x_batch)
      tf.add_to_collection('emb_mask', self.emb_mask)
      tf.add_to_collection('word_num', self.word_num)
      #tf.add_to_collection("embedding", self.embedding['input'])
      saver.save(sess, './model1/model.ckpt')

  def restore(self):
    train_file='processed_click_data.txt'
    #self.init_data(train_file, 'word2vec.txt')
    #pred_ids = [x for x in range(self.seed_seller_size)]
    pred_ids = linecache.getlines(train_file)

    with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
      new_saver.restore(sess, './model/model.ckpt')
      # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
      y = tf.get_collection('predictor')[0]
      x_batch = tf.get_collection('x_batch')[0]
      emb_mask = tf.get_collection('emb_mask')[0]
      word_num = tf.get_collection('word_num')[0]

      graph = tf.get_default_graph()
      #embedding = graph.get_operation_by_name('self.embedding').outputs[0]

      # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
      #x_batch = graph.get_operation_by_name('self.x_batch').outputs[0]
      #emb_mask = graph.get_operation_by_name('self.emb_mask').outputs[0]
      #word_num = graph.get_operation_by_name('self.word_num').outputs[0]




      # 使用y进行预测  
      self.batch_size = 100
      total_batch = int(len(pred_ids))/self.batch_size
      print ("Total inference batch: ", str(total_batch))
      for i in range(total_batch):
        #x, batch_mask, word_number = self.create_data_for_infer(i*self.batch_size, self.batch_size, pred_ids)
        x, y_batch, batch_mask, word_number = self.read_data(i*self.batch_size, self.batch_size, pred_ids)
        print x
        predict_op = tf.nn.top_k(y, 200, False)
        predict_result = sess.run(predict_op, feed_dict={x_batch:x, emb_mask: batch_mask, word_num: word_number})
        print predict_result
        #
        sys.stdout.flush()

exp = TFDnnS2S()
#exp.restore()
exp.run()
#exp.preprocess_data('click_data.txt', 'processed_click_data.txt', 'seed_id_mapping.txt', 'label_id_mapping.txt')
