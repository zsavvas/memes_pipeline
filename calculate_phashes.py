#!/usr/bin/env python

'''
This code is part of the publication "On the Origins of Memes by Means of Fringe Web Communities" at IMC 2018.
If you use this code please cite the publication.
'''

import os
from operator import itemgetter
from PIL import Image
import math, operator
from time import sleep
from multiprocessing import Process, Manager, Queue
#import Queue
import itertools
import imagehash
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-d", "--directory", dest='directory', help="directory that contains the images to calculate phashes")
parser.add_option("-o", "--output", dest='output', default='phashes.txt',help="file to store the phashes")

(options, args) = parser.parse_args()

IMAGE_DIR = options.directory
HASHES = options.output

num_workers = 8
try:
    imagehash_version = float(imagehash.__version__)
except AttributeError:
    imagehash_version=3.5
def list_files(images_dir):
        # reading filanmes of the images from directory
        file_names = []
        count = 0
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                file_names.append('%s/%s' %(root, file))
                count +=1
        return file_names


def calculate_hash(in_queue, out_list):

    while True:
        try:
            item = in_queue.get(True, 5)
        except Queue.Empty: 
            break
        line_no, line = item
        if line_no % 10000 == 0:
            output = open(HASHES, 'w')
            for result in out_list:
                output.write(result + '\n')
            output.close()
        if line == None:
            return
        try:
            # because of a change in imagehash library at version 4.0 and to be backward compatible with phashes from Memes paper
            if imagehash_version > 3.5:
                image_phash = imagehash.old_hex_to_hash(str(imagehash.phash(Image.open(line))))
            else:
                image_phash = imagehash.phash(Image.open(line))
            out_list.append(str(line) + '\t' + str(image_phash))
        except Exception as e:
            print(str(e))
            pass

if __name__ == "__main__":
    filenames = list_files(IMAGE_DIR)
                
    manager = Manager()
    results = manager.list()
    work = manager.Queue(num_workers)

    # start for workers    
    pool = []
    for i in range(num_workers):
        p = Process(target=calculate_hash, args=(work, results))
        p.start()
        pool.append(p)


    iters = itertools.chain(filenames, (None,)*num_workers)
    for num_and_line in enumerate(iters):
        work.put(num_and_line)
       

    for p in pool:
        p.join()

    # get the results
    print("Done. Writing to file %d phashes" %(len(results)))
    output = open(HASHES, 'w')
    for result in results:
        output.write(result + '\n')
    output.close()
