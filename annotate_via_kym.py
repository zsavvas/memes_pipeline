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
import pickle
import ast

import tensorflow as tf
import numpy as np


DISTANCE_THRESHOLD = 8
DEBUG = False

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


def process_clusters_file(clusters_file):
    
    clusters = {}

    print('[i] process_clusters_file', clusters_file)

    with open(clusters_file) as fd:
        for idx, line in enumerate(fd.readlines()):

            # for each cluster, get the medroid 
            dic = ast.literal_eval(line)
            cluster_no = int(dic['cluster_no'])

            if cluster_no == -1: 
                continue

            images = dic['images']
            medroid_phash = dic['medroid_phash']

            clusters[cluster_no] = medroid_phash

    return clusters


def process_kym_files(kym_phashes_file):
    
    kym_phashes_by_meme_dic = {}
    kym_images_dic = {}
    kym_images_dic_reverse = {}
    kym_meme_name = {}

    print('[i] process_kym_files', kym_phashes_file)

    with open(kym_phashes_file) as fd:
        for idx, line in enumerate(fd.readlines()):

            if idx == 0:
                continue

            split = line.split()

            image = split[0]
            m_class = split[1]
            proba = split[2]
            phash = split[3]

            image_name = image[image.index('/'):]
            meme_name = image_name.split('_')[0]

            if not meme_name:
                continue
            # take into account only the images that are predicted to not be screenshots from social networks
            if m_class == 'relevant' and float(proba) > 0.8:
                continue

            if meme_name not in kym_phashes_by_meme_dic:
                kym_phashes_by_meme_dic[meme_name] = []
            kym_phashes_by_meme_dic[meme_name].append(phash)

            kym_images_dic[phash] = image
            kym_images_dic_reverse[image] = phash
            kym_meme_name[phash] = meme_name

    return kym_phashes_by_meme_dic, kym_images_dic, kym_images_dic_reverse, kym_meme_name

def read_phashes_manifest(phashes_path):
    phashes = {}
    print('[i] read_phashes_manifest', phashes_path)
    with open(phashes_path) as infile:
        for line in infile.readlines():
            split = line.split()
            hashid = split[0].strip()
            hash_str = split[1].strip()
            phashes[hashid] = hash_str
    print('[i] processed', len(phashes))
    return phashes


def default(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def precompute_vectors(hashes_dic, phases_path):
    pickle_file = phases_path + '.pickle'
    if os.path.isfile(pickle_file): 
        with open(pickle_file, 'rb') as fo:
            hashes = pickle.load(fo)
        print('[w] fetched precomputed vectors from ', pickle_file, 'new processed', len(hashes))
    else:
        hashes = np.array(list(hashes_dic.values()))
        hashes = [np.array(hex_to_hash(hex_hash)) for hex_hash in hashes]
        with open(pickle_file, 'wb') as fo:
            pickle.dump(hashes, fo)
        print('[w] generated vectors and stored them in ', pickle_file, 'new processed', len(hashes))

    return hashes

    
def fetch_info_kym(phash, kym_phashes_by_meme_dic, kym_images_dic, kym_images_dic_reverse, kym_meme_name, kym_phashes):
    
    phashes = list(kym_images_dic_reverse.values())
    index = phashes.index(phash) 

    meme_name = kym_meme_name[phash]
    image_name = kym_images_dic[phash]
    phash_np = kym_phashes[index]

    return index, meme_name, image_name, phash_np



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



'''
    #for d in ['/gpu:0', '/gpu:1']: 
    #    with tf.device(d):
    Can't use /gpu:1 -- https://github.com/tensorflow/tensorflow/issues/9506
'''
def seek_queue_many(ids_i, ids_j, hashes_i, hashes_j, outdir, blacklist, hashes_diff):
    len_hashes = len(hashes_i)

    last_index = 0
    num_threads = 5
    batch_size = int(len_hashes/num_threads)+1
    total_tasks = len_hashes - len(blacklist)
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
    config=tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    
    enqueue_threads = [threading.Thread(target=check_batch_many, args=[sess, hashes_i, enqueue_op, init_i, batch_size, queue_i, queue_hash_i, blacklist]) for init_i in range(last_index, len_hashes, batch_size)]
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
        diff, filter, where = sess.run([diff_op_many, filter_op, where_op], feed_dict={diff_hash_i: hash_i, diff_hashes_j: hashes_j})
        for j in where:
            j_pos = j[0] 
            key_id = str(ids_i[i]) + '-' + str(ids_j[j_pos])
            hashes_diff[key_id] = diff[j_pos]

        seen_images.append(i)

        if _ % 1000 == 0:
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




def main(options, arguments):
    devices = ['/gpu:0', '/gpu:1']
    if options.device == None:
        device = devices[0]
    else:
        device = devices[int(options.device)]

    if options.distance == None:
        distance = 8
    else: 
        distance = int(options.distance)
    global DISTANCE_THRESHOLD 
    DISTANCE_THRESHOLD = distance
    phases_path = options.phashes

    clusters_file = options.clustering
    #if distance != 8:
    #    clusters_file = clusters_file.replace('.txt', '_' + str(distance) + '.txt')
        
    kym_phashes_file = "kym_phashes_classes.txt"
    outfile = options.output

    ''' 
        Process Clusters 
    '''
    clusters = process_clusters_file(clusters_file)
    src_hashes_dic = read_phashes_manifest(phases_path)
    src_hashes = precompute_vectors(src_hashes_dic, phases_path)

    src_values = list(src_hashes_dic.values())
    cluster_hashes = []
    for cid in clusters:
        medroid = clusters[cid]
        index = src_values.index(medroid)
        cluster_hashes.append(src_hashes[index])

    print('[i] computed cluster backnone with #hashes', len(cluster_hashes))

    ''' 
        Process KYM 
    '''
    kym_phashes_by_meme_dic, kym_images_dic, kym_images_dic_reverse, kym_meme_name = process_kym_files(kym_phashes_file)
    kym_phashes = precompute_vectors(kym_images_dic_reverse, kym_phashes_file)

    # --- Tag Clusters
    hashes_i = cluster_hashes
    hashes_j = kym_phashes

    hashes_diff = {}#load_json(outfile)
    blacklist = []#read_blacklist_dict(phases_path)

    print('[i] seek_queue_many init')
    with tf.device(device):
        seek_queue_many(list(clusters.keys()), list(kym_images_dic_reverse.keys()), hashes_i, hashes_j, outfile, blacklist, hashes_diff)
    print('[i] seek_queue_many end', outfile)

    #sys.exit()



if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("-d", "--device", dest='device', help="GPU device ID", default=None)
    parser.add_option("-p", "--phashes", dest='phashes', default='phashes.txt',help="phashes file")
    parser.add_option("-c", "--clustering", dest='clustering', default='clustering_output.txt', help="clustering output file")
    parser.add_option("-o", "--output", dest='output', default='clustering_selection.json' ,help="output file with kym annotations")
    parser.add_option("-k", "--distance", dest='distance', default=8, help="threshold for selecting annotations")

    (options, arguments) = parser.parse_args()
    main(options, arguments)


