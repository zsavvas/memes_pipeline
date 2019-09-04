#!/usr/bin/env python
'''
This code is part of the publication "On the Origins of Memes by Means of Fringe Web Communities" at IMC 2018.
If you use this code please cite the publication.
'''
import os, sys, shutil, traceback
import json
import time
import threading

from optparse import OptionParser

import multiprocessing
from time import sleep
import settings

import tensorflow as tf
import numpy as np
import pickle
import numpy as np

tf.app.flags.DEFINE_integer("batch_size", 4000000, "Search batch size")
FLAGS = tf.app.flags.FLAGS

DISTANCE_THRESHOLD = 10
DEBUG = False

config=tf.ConfigProto() #allow_soft_placement=True, log_device_placement=True
#config.gpu_options.allow_growth=True
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44

def load_json(outdir_tmp):
    myjson = {}
    if os.path.isfile(outdir_tmp): 
        with open(outdir_tmp, 'r') as outfile:
            myjson = json.load(outfile)    
    return myjson


'''
Convert a stored hash (hex, as retrieved from str(Imagehash))
to a bool array object.
'''
def hex_to_hash(hexstr, hash_size=8):

    l = []
    count = hash_size * (hash_size // 4)
    if len(hexstr) != count:
        emsg = 'Expected hex string size of {}.'
        raise ValueError(emsg.format(count))
    for i in range(count // 2):
        h = hexstr[i*2:i*2+2]
        v = int("0x" + h, 16)
        l.append([v & 2**i > 0 for i in range(8)])
    return np.array(l).flatten()#.astype(int)

def read_phashes_manifest(phashes_path):
    phashes = {}
    with open(phashes_path) as infile:
        for line in infile.readlines():
            split = line.split('\t')
            hashid = split[0].strip()
            hash_str = split[1].strip()
            phashes[hashid] = hash_str
    print('[i] processed', len(phashes))
    return phashes

def read_phashes_diff(phash_path):
    print('[i] computing diffs in', phash_path)
    hashes = None
    with open(phash_path) as data_file:    
        hashes = json.load(data_file)
    return hashes

def precompute_vectors(hashes, phases_path):
    pickle_file = phases_path + '.pickle'
    if os.path.isfile(pickle_file): 
        with open(pickle_file, 'rb') as fo:
            hashes = pickle.load(fo)
        print('[w] fetch precomputed vectors from ', pickle_file, 'new processed', len(hashes))
        return hashes
    else:
        hashes = np.array(list(hashes.values()))
        hashes2 = []
        for hex_hash in hashes:
            try:
                hashes2.append(hex_to_hash(hex_hash))
            except Exception as e:
                print(hex_hash)
                print(str(e))
        with open(pickle_file, 'wb') as fo:
            pickle.dump(hashes2, fo)
    return hashes2
   

'''
    Re-implementation of seek_sequential making batches of samples. 
    Use isntead if memory issues with the size of the vectors as the dataset grows. 
'''
def seek_sequential_batch(hashes, outdir):
    # ----------

    len_hashes = len(hashes)
    pbar = tf.contrib.keras.utils.Progbar(len_hashes)
    cprogress = tf.constant(0)

    # One shot iterator through all images in the dataset 
    dataset_i = tf.data.Dataset.range(len_hashes)
    iterator_i = dataset_i.make_one_shot_iterator()
    next_element_i = iterator_i.get_next()

    hash_i = tf.placeholder(tf.bool, shape=[64])
    hashes_j = tf.placeholder(tf.bool, shape=[None, 64])

    diff_op = tf.count_nonzero(tf.not_equal(hash_i, hashes_j), 1) 

    nz_op = tf.count_nonzero(hashes_j, 1) 

    pbar.update(0)

    with tf.train.MonitoredSession() as sess: 

        for _ in range(len_hashes-1):
            i = sess.run(next_element_i)
            for batch in range(i+1, len_hashes, FLAGS.batch_size):
                diff = sess.run(diff_op, feed_dict={hash_i: hashes[i], hashes_j: hashes[batch:batch+FLAGS.batch_size]})
            pbar.update(i)


'''
    Doesn't use queues and makes batches of samples. 
    Performance-wise, this method manages to run 100% 
    of the GPU at intervals (feed_dict slows things up).
'''
def seek_sequential(hashes, outdir):

    # ----------

    len_hashes = len(hashes)
    pbar = tf.contrib.keras.utils.Progbar(len_hashes)
    cprogress = tf.constant(0)

    # One shot iterator through all images in the dataset 
    dataset_i = tf.data.Dataset.range(len_hashes)
    iterator_i = dataset_i.make_one_shot_iterator()
    next_element_i = iterator_i.get_next()

    hash_i = tf.placeholder(tf.bool, shape=[64])
    hashes_j = tf.placeholder(tf.bool, shape=[None, 64])

    diff_op = tf.count_nonzero(tf.not_equal(hash_i, hashes_j), 1) 
    pbar.update(0)

    with tf.train.MonitoredSession() as sess: 

        for _ in range(len_hashes-1):
            i = sess.run(next_element_i)
            diff = sess.run(diff_op, feed_dict={hash_i: hashes[i], hashes_j: hashes[i+1:]})
            pbar.update(i)


''' 
    Iterates over our data puts small junks into our queue.
'''
def check_batch_pair(sess, hashes, enqueue_op, init_i, batch_size, queue_hash_i, queue_hash_j):

    len_hashes = len(hashes)

    for i in range(init_i, init_i+batch_size):

        x = []
        y = []

        for hash_j in hashes[i+1:]:  
            hash_i = hashes[i]
            x.append(hash_i)
            y.append(hash_j)

        if len(x) > 0 and len(y) > 0:
            sess.run(enqueue_op, feed_dict={queue_hash_i: x,
                                            queue_hash_j: y})

def seek_queue_pair(hashes, outdir):

    len_hashes = len(hashes)

    last_index = 0
    num_threads = 10
    batch_size = len_hashes/num_threads

    # are used to feed data into our queue
    queue_hash_i = tf.placeholder(tf.bool, shape=[None, 64])
    queue_hash_j = tf.placeholder(tf.bool, shape=[None, 64])

    queue = tf.FIFOQueue(capacity=batch_size, dtypes=[tf.bool, tf.bool], shapes=[[64], [64]])

    enqueue_pair_op = queue.enqueue_many([queue_hash_i, queue_hash_j])
    dequeue_pair_op = queue.dequeue()

    diff_hash_i = tf.placeholder(tf.bool, shape=[64])
    diff_hash_j = tf.placeholder(tf.bool, shape=[64])
    diff_hashes_j = tf.placeholder(tf.bool, shape=[None, 64])
    diff_op_pair = tf.count_nonzero(tf.not_equal(diff_hash_i, diff_hash_j)) 

    # ------------------------- #

    # start the threads for our FIFOQueue and batch
    config=tf.ConfigProto()
    sess = tf.Session(config=config)
    
    enqueue_threads = [threading.Thread(target=check_batch_pair, args=(sess, hashes, enqueue_pair_op, init_i, batch_size, queue_hash_i, queue_hash_j)) for init_i in range(last_index, len_hashes, batch_size)]
    # Start the threads and wait for all of them to stop.
    for t in enqueue_threads: 
        t.isDaemon()
        t.start()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Fetch the data from the pipeline and put it where it belongs (into your model)
    for _ in range((len_hashes*len_hashes)/2-len_hashes/2):
        # Computing diff
        hash_i, hash_j = sess.run(dequeue_pair_op)
        diff = sess.run(diff_op_pair, feed_dict={diff_hash_i: hash_i, diff_hash_j: hash_j})


    # shutdown everything to avoid zombies
    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(enqueue_threads)
    coord.join(threads)
    sess.close()

def default(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def check_batch_many(sess, hashes, enqueue_op, init_i, batch_size, queue_i, queue_hash_i, blacklist=[], num_devices=1):
    
    x = []
    y = []

    len_hashes = len(hashes)
    candidates = range(init_i, init_i+batch_size, num_devices)

    #for i in set(candidates) - set(blacklist):
    for i in candidates:
        if i in blacklist:
            continue

        if i < len_hashes:
            x.append(i)
            y.append(hashes[i])

    if len(x) > 0 and len(y) > 0:
        sess.run(enqueue_op, feed_dict={queue_i: x,
                                        queue_hash_i: y})


def op_runner(sess, hashes, dequeue_op, init_i, batch_size, diff_op_many, diff_hash_i, diff_hashes_j, pbar, num_devices=1):

    len_hashes = len(hashes)
    for _ in range(init_i, init_i+batch_size, num_devices):
        if _ < len_hashes - 1:
            run_options = tf.RunOptions(timeout_in_ms=400000)
            i, hash_i = sess.run(dequeue_op, options=run_options)
            diff = sess.run(diff_op_many, feed_dict={diff_hash_i: hash_i, diff_hashes_j: hashes[i+1:]})    
            pbar.add(num_devices)

'''
    #for d in ['/gpu:0', '/gpu:1']: 
    #    with tf.device(d):
    Can't use /gpu:1 -- https://github.com/tensorflow/tensorflow/issues/9506
'''
def seek_queue_many(ids, hashes, outdir, blacklist, hashes_diff):

    len_hashes = len(hashes)

    last_index = 0
    num_threads = 5
    batch_size = int(len_hashes/num_threads)
    total_tasks = len_hashes - len(blacklist)
    print(batch_size)
    print(total_tasks)
    pbar = tf.contrib.keras.utils.Progbar(total_tasks)

    # are used to feed data into our queue
    queue_i = tf.placeholder(tf.int32, shape=[None])
    queue_hash_i = tf.placeholder(tf.bool, shape=[None, 64])
    queue_hashes_j = tf.placeholder(tf.bool, shape=[batch_size, None]) #shape=[None, 64] [len_hashes]

    queue = tf.FIFOQueue(capacity=50, dtypes=[tf.int32, tf.bool], shapes=[[], [64]])

    enqueue_op = queue.enqueue_many([queue_i, queue_hash_i])
    dequeue_op = queue.dequeue()

    diff_hash_i = tf.placeholder(tf.bool, shape=[64])
    diff_hashes_j = tf.placeholder(tf.bool, shape=[None, 64])
    diff_op_many = tf.count_nonzero(tf.not_equal(diff_hash_i, diff_hashes_j), 1) 

    filter_op = tf.less_equal(diff_op_many, DISTANCE_THRESHOLD)

    where_op = tf.where(filter_op)

    # start the threads for our FIFOQueue and batch
    sess = tf.Session(config=config)
    
    enqueue_threads = [threading.Thread(target=check_batch_many, args=[sess, hashes, enqueue_op, init_i, batch_size, queue_i, queue_hash_i, blacklist]) for init_i in range(last_index, len_hashes, batch_size)]
    # Start the threads and wait for all of them to stop.
    for t in enqueue_threads: 
        t.isDaemon()
        t.start()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    pbar.update(0)

    seen_images = []
    outdir_tmp = outdir + '.tmp' + '.' + str(settings.distributed_machine)
    # Fetch the data from the pipeline and put it where it belongs (into your model)
    for _ in range(total_tasks):
        # Computing diff
        i, hash_i = sess.run(dequeue_op)
        diff, filter, where = sess.run([diff_op_many, filter_op, where_op], feed_dict={diff_hash_i: hash_i, diff_hashes_j: hashes[i:]})
        for j in where:
            j_rel = j[0] 
            j_abs = i+j_rel
            key_id = ids[i] + '-' + ids[j_abs]
            hashes_diff[key_id] = diff[j_rel]

        seen_images.append(i)

        if _ % 100000 == 0:
            with open(outdir_tmp, 'w') as outfile:
                json.dump(hashes_diff, outfile, default=default)
            progress_file = 'progress.' + outdir_tmp
            with open(progress_file + '.txt', 'w') as outfile:
                outfile.write(str(i)+'\n')
            with open(progress_file + '.json', 'w') as outfile:
                json.dump(str(seen_images), outfile, default=default)

        pbar.update(_)

    with open(outdir, 'w') as outfile:
        json.dump(hashes_diff, outfile, default=default)         

    # shutdown everything to avoid zombies
    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(enqueue_threads)
    coord.join(threads)
    #coord.join(operation_threads)
    sess.close()

    os.remove(outdir_tmp)
    os.remove(progress_file+'.txt')
    os.remove(progress_file+'.json')

'''
    #for d in ['/gpu:0', '/gpu:1']: 
    #    with tf.device(d):
    Can't use /gpu:1 -- https://github.com/tensorflow/tensorflow/issues/9506
'''
def seek_queue_many_device(ids, hashes, outdir, blacklist, hashes_diff, devices, device):

    len_hashes = len(hashes)
    num_devices = len(devices)

    last_index = 0
    num_threads = 8
    batch_size = int(len_hashes/num_threads)
    total_tasks = len_hashes - 1 - len(blacklist)
    pbar = tf.contrib.keras.utils.Progbar(total_tasks)

    # Feed data into our queue
    queue_i = tf.placeholder(tf.int32, shape=[None])
    queue_hash_i = tf.placeholder(tf.bool, shape=[None, 64])
    queue_hashes_j = tf.placeholder(tf.bool, shape=[batch_size, None]) #shape=[None, 64] [len_hashes]

    queue = tf.FIFOQueue(capacity=100, dtypes=[tf.int32, tf.bool], shapes=[[], [64]])

    enqueue_op = queue.enqueue_many([queue_i, queue_hash_i])
    dequeue_op = queue.dequeue()

    diff_hash_i = tf.placeholder(tf.bool, shape=[64])
    diff_hashes_j = tf.placeholder(tf.bool, shape=[None, 64])
    diff_op_many = tf.count_nonzero(tf.not_equal(diff_hash_i, diff_hashes_j), 1) 

    filter_op = tf.less_equal(diff_op_many, DISTANCE_THRESHOLD)
    where_op = tf.where(filter_op)

    # start the threads for our FIFOQueue and batch
    config=tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    if devices.index(device) == 0:
        last_index += last_index % 2
    elif devices.index(device) == 1:
        last_index += (last_index+1) % 2

    enqueue_threads = [threading.Thread(target=check_batch_many, args=[sess, hashes, enqueue_op, init_i, batch_size, queue_i, queue_hash_i, blacklist, num_devices]) for init_i in range(last_index, len_hashes, batch_size)]
    # Start the threads and wait for all of them to stop.
    for t in enqueue_threads: 
        t.isDaemon()
        t.start()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    pbar.update(0)

    seen_images = []
    outdir_tmp = outdir + '.tmp' + '.' + str(settings.distributed_machine) + '.' + str(devices.index(device))
    # Fetch the data from the pipeline and put it where it belongs (into your model)
    for _ in range(devices.index(device), len_hashes - 1 - len(blacklist), num_devices):
        # Computing diff
        i, hash_i = sess.run(dequeue_op)
        diff, filter, where = sess.run([diff_op_many, filter_op, where_op], feed_dict={diff_hash_i: hash_i, diff_hashes_j: hashes[i+1:]})
        for j in where:
            j_rel = j[0] 
            j_abs = i+j_rel+1
            key_id = ids[i] + '-' + ids[j_abs]
            hashes_diff[key_id] = diff[j_rel]

        seen_images.append(i)

        # Store progress
        if _ % 1000 == 0:
            with open(outdir_tmp, 'w') as outfile:
                json.dump(hashes_diff, outfile, default=default)
            progress_file = 'progress.' + outdir_tmp
            with open(progress_file + '.txt', 'w') as outfile:
                outfile.write(str(i)+'\n')
            with open(progress_file + '.json', 'w') as outfile:
                json.dump(str(seen_images), outfile, default=default)

        pbar.update(_)

    # Consolidate results
    with open(outdir + '.' + str(settings.distributed_machine) + '.' + str(devices.index(device)), 'w') as outfile:
        json.dump(hashes_diff, outfile, default=default)      

    # Reset progress
    with open(progress_file, 'w') as outfile:
        outfile.write('0\n')

    # Shutdown everything to avoid zombies
    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(enqueue_threads)
    coord.join(threads)
    sess.close()

def convert_vectors(hashes):
    thashes = []
    for h in hashes:
        thashes.append(tf.convert_to_tensor(h, dtype=tf.int64))
    return thashes

def read_blacklist(phash_path):
    blacklist = []
    with open(phash_path + '.new.progress') as data_file:    
        blacklist = json.load(data_file)
    blacklist_dic = {}
    for b in blacklist:
        if b not in list(blacklist_dic.keys()):
            blacklist_dic[b] = None

    print('[i] blacklisting', len(blacklist_dic))
    return blacklist_dic

def read_blacklist_dict(phash_path):
    blacklist_dic = {}
    if not os.path.exists(phash_path + '.new_dict.progress'):
        with open(phash_path + '.new_dict.progress', 'w') as f:
            f.write("{}")
    with open(phash_path + '.new_dict.progress') as data_file:    
        blacklist_dic = json.load(data_file)

    print('[i] blacklisting', len(blacklist_dic))
    return blacklist_dic

def main(options, arguments):

    global previous_time
    previous_time = time.time()

    phases_path = options.input
    if options.output==None:
        outfile = phases_path.replace('.txt', '-diffs.json')
    else:
        outfile = options.output

    ## - Pre-computation
    hashes_dic = read_phashes_manifest(phases_path)
    hashes = precompute_vectors(hashes_dic, phases_path)

    hashes_diff = {}
    blacklist = read_blacklist_dict(phases_path)

    if options.device == None:
        seek_queue_many(list(hashes_dic.keys()), hashes, outfile, blacklist, hashes_diff)

    else:

        devices = ['/gpu:0', '/gpu:1']
        device = devices[int(options.device)]

        with tf.device(device):
            seek_queue_many_device(list(hashes_dic.keys()), hashes, outfile, blacklist, hashes_diff, devices, device)

    os.remove(phases_path + '.new_dict.progress')
    os.remove(phases_path + '.pickle')

if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("-d", "--device", dest='device', help="GPU device ID", default=None)
    parser.add_option("-i", "--input", dest='input', default='phashes.txt',help="phashes file")
    parser.add_option("-o", "--output", dest='output', default=None ,help="file that we store the phashes distances")

    (options, arguments) = parser.parse_args()

    main(options, arguments)

