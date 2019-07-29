#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import zmq, msgpack
from multiprocessing import Process
import msgpack_numpy as m

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class ExperienceSender():
    """ Opens zmq DEALER socket & sends 'experiences' over from the environment
    """
    def __init__(self, connect_to='tcp://195.0.0.7:5555'):

        #important zmq initialization stuff to connect to server
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.DEALER)
        self.zmq_socket.connect(connect_to)
        myid = b'0'
        self.zmq_socket.identity = myid.encode('ascii')
        self.num_batch = 0
        m.patch()
        
    def obs_to_dump(self, obs_array, serial_func):
        dump_array = []
        for obs_dict in obs_array:
            dump_array += serial_func(obs_dict)
        return dump_array
    
    def send_recv(self, dump_array, recv_callback):
        pass

    def send_obs(self, obs_array, serial_func, recv_callback, header_dict = {}):
        dump_array = self.obs_to_dump(obs_array, serial_func)
        header_dict['batchnum'] = self.num_batch
        header_dump = [msgpack.dumps(header_dict)]
        dump_array = header_dump + dump_array
        p = Process(target=self.send_recv, args=(dump_array, recv_callback))
        p.start()
        p.join()
        self.num_batch += 1