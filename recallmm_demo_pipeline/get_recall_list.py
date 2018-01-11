#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import traceback
import sqlite3 as lite



class LiteSQL():
  def __init__(self):
    self.con = lite.connect('itemcf.db')
    self.con.text_factory = str
    self.cur = self.con.cursor()

  def get(self, query):
    sql1 = "select list from relevence where query = %s" % (query)
    sql2 = "select id, title, pic, cat, subtitle from detail_table where "
    sql3 = "select query, quyercate from query_table where query = %s" % (query)
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
          id, score = item.split(',', 1)
          if float(score) == 0:
            continue
          relate_list[int(id)] = float(score)
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
          id_detail[row[0]] = row
    except:
      traceback.print_exc()

    sorted_related_list = sorted(relate_list.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    for k, v in sorted_related_list:
      if k in id_detail:
        res.append([k, v, id_detail[k][1], id_detail[k][2], id_detail[k][3],  id_detail[k][3] ])
      else:
        res.append([k, v, '', '', '', ''])
    return res

