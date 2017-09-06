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
    self.con = lite.connect('sw_click.db')
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



#db=LiteSQL()
#db.create_table()
#print db.get(1)
