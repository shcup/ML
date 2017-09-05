#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import logging
import os.path
import unittest
import tempfile
import itertools
import sys

import numpy as np
from scipy.linalg.misc import norm

from BaseHTTPServer import BaseHTTPRequestHandler
import cgi
import json
import io,shutil,urllib 
import urlparse

from get_itemcf_detail_list import *


db=LiteSQL()
class TodoHandler(BaseHTTPRequestHandler):
    """A simple TODO server

    which can display and manage todos for you.
    """

    # Global instance to store todos. You should use a database in reality.
    TODOS = []

    def do_GET(self):
        # return all todos

        print 'get request:'+self.path
        rs = urlparse.urlparse(self.path)
        params=urlparse.parse_qs(rs.query)

        if not self.path.startswith('/itemcf'):
            self.send_error(404, "File not found.")
            return
        elif 'id' not in params:
            self.send_error(404, 'id not found')
            return
        elif 'table' not in params:
            self.send_error(404, 'table not found')
            return

        # Just dump data to json, and return it
        res = db.get(int(params['id'][0]), params['table'][0])
        message = json.dumps(res)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(message)

    def do_POST(self):
        """Add a new todo

        Only json data is supported, otherwise send a 415 response back.
        Append new todo to class variable, and it will be displayed
        in following get request
        """
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        if ctype == 'application/json':
            length = int(self.headers['content-length'])
            post_values = json.loads(self.rfile.read(length))
            self.TODOS.append(post_values)
        else:
            self.send_error(415, "Only json data is supported.")
            return

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        self.wfile.write(post_values)

if __name__ == '__main__':
    # Start a simple server, and loop forever
    from BaseHTTPServer import HTTPServer
    server = HTTPServer(('localhost', 8888), TodoHandler)
    print("Starting server, use <Ctrl-C> to stop")
    server.serve_forever()
