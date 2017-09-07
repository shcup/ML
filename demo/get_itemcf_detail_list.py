#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import logging
import os.path
import unittest
import tempfile
import itertools
import sys
import traceback
import sqlite3 as lite


class LiteSQL():
  def __init__(self):
    self.con = lite.connect('itemcf.db')
    self.con.text_factory = str
    self.cur = self.con.cursor()
    
  def create_table(self, table_name):
    try:
      sql = 'CREATE TABLE if not exists %s \
            ( id1 long,  \
              id2 long, \
              score double,\
              PRIMARY KEY (id1, id2) \
            );' % (table_name)
      self.cur.execute(sql)
      self.cur.execute("create index if not exists id1_index on itemcf_table (id1);")
    except lite.IntegrityError:
      sys.stderr.write(tableName)
      sys.exit(1)

    try:
      sql = 'CREATE TABLE if not exists detail_table \
            ( id long PRIMARY KEY, \
              nick, \
              title, \
              pict_url, \
              shop_domain \
            );'
      self.cur.execute(sql)
      self.cur.execute("create index if not exists id_index on detail_table(id);")
    except lite.IntegrityError:
      sys.stderr.write(tableName)
      sys.exit(1)


  def get(self, id, itemcf_table):
  
    sql = '\
        select t1.id2, t1.score, t2.nick, t2.title \
        from ( \
        select  id2, score \
        from %s \
        where id1 = %d)t1  \
        left outer join detail_table t2 \
        on t1.id2 == t2.id \
        order by t1.score ' % (itemcf_table, id)
    print sql

    try:
      self.cur.execute(sql)
      r = self.cur.fetchall()
      #for e in range(len(r)):
      #  print >> f, '\t'.join([r[e][0], r[e][1], base64.b64encode(r[e][2]), str(r[e][3]), str(r[e][4])])
      return r
    except:
      traceback.print_exc()

    return

  def get2(self, id, itemcf_table):
    sql1 = "select list from %s where id = %d" % (itemcf_table, id)
    sql2 = "select id, nick, shop_name, shop_url from detail_table where "
    relate_list = {}
    id_detail = {}
    res = []

    try:
      self.cur.execute(sql1)
      r = self.cur.fetchall()
      if len(r) > 0:
        items = r[0][0].strip().split(';')
        first = 0
        print len(items)
        for item in items:
          id,score,source = item.split(',', 2)
          if float(score) == 0:
            continue
          relate_list[int(id)]=float(score)
          if (first != 0):
            sql2 = sql2 + " or id=" + id
            first = 1
          else:
            sql2 = sql2 + " id=" + id
            first = 1
        print sql2
        self.cur.execute(sql2)
        r2 = self.cur.fetchall()
        print r2
        for row in r2:
          id_detail[row[0]]=row
    except:
      traceback.print_exc()

    sorted_related_list = sorted(relate_list.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    for k,v in sorted_related_list:
      if k in id_detail:
        res.append([k, v, id_detail[k][1], id_detail[k][2], id_detail[k][3]])
      else:
        res.append([k, v, '', '', ''])
    return res



  def import_file(self, file_name, table):
    cnt = 0
    for line in open(file_name):
      sp = line.strip().split('\t')
      if len(sp) != 4:
        print line
        continue
      try:
        cnt = cnt + 1
        row=[0]*4
        row[0]= sp[0]
        row[1] = sp[1]
        row[2] = sp[2]
        row[3] = sp[3]
        sql = "INSERT INTO " + table + " VALUES (?,?,?,?);"
        self.cur.executemany(sql, (row,))
        if cnt % 100000 == 0:
          print 'commit for ' + str(cnt) 
          self.con.commit()
      except:
        traceback.print_exc()      
      self.con.commit()







#db=LiteSQL()
#db.create_table()
#print db.get2(10007761, 'sw_buy')
#db.import_file('detail_data.txt', 'detail_table')
