import pickle
import random

from code.dataset_generator import generate_train_public_data
from code.utils import get_border_and_garbage_images, relabel_me, relabel_image_metadata, relabel_me6, relabel_eleven

import argparse

def load_data(TRAIN_DATA, EXTRA_DATA):
    train_metadata, public_metadata = generate_train_public_data(TRAIN_DATA, EXTRA_DATA)

    print(len(train_metadata), len(public_metadata))
    print(len(train_metadata) + len(public_metadata))

    #remove bad images
    remove_images = set(get_border_and_garbage_images())
    train_metadata = [x for x in train_metadata if not x.image_name in remove_images or 11 in x.image_labels]
    public_metadata = [x for x in public_metadata if not x.image_name in remove_images or 11 in x.image_labels]

    print(len(train_metadata) + len(public_metadata))
    print(len(train_metadata), len(public_metadata))

    pickle.dump(train_metadata, open( "pckl/train_metadata.p", "wb" ))
    pickle.dump(public_metadata, open( "pckl/public_metadata.p", "wb" ))

def get_validation_list(label, relabel = {'0':0.0, '1':0.0, '2':0.25, '3':0.5, '4':0.75, '5':1.0}):

    result = []

    with open('validation/' + str(label) + '.txt') as f:
        for x in f:
            x = x.strip().split(' ')

            who = x[0]
            val = relabel[x[1]]

            result.append((who, val))
    
    return result

def positive_relabel_auto(label, image_metadata):
    st = pickle.load(open('auto-positive-relabel/' + str(label) + '.p', 'rb'))
    
    print("POSITIVE:", label, len(st))
    cnt = 0

    for metadata in image_metadata:
        if metadata.image_name in st:
            cnt += 1
            metadata.relabel[label] = 0.99

            if not label in metadata.image_labels:
                metadata.image_labels.add(label)

def relabel_auto(label, image_metadata, folder='auto-relabel/'):
    st = pickle.load(open(folder + str(label) + '.p', 'rb'))
    
    print("NEGATIVE:", label, len(st))
    cnt = 0

    for metadata in image_metadata:
        if metadata.image_name in st:
            cnt += 1
            metadata.relabel[label] = 0.01

    print(cnt)

def get_train_validation_data():

    train_metadata = pickle.load(open( "pckl/train_metadata.p", "rb" ))
    public_metadata = pickle.load(open( "pckl/public_metadata.p", "rb" ))

    random.seed()
    random.shuffle(public_metadata)
    
    train_metadata.extend(public_metadata)

    relabel_me6('relabel/6.txt', train_metadata)
    positive_relabel_auto(6, train_metadata)

    for label in [0,2,3,4,5,6,7,10,13,14,17]:
        relabel_auto(label, train_metadata)

    relabel_me(1, 'relabel/1.txt', train_metadata, {'0':0.0, '1':0.0, '2':0.4, '3':0.5, '4':0.85, '5':1.0})
    relabel_me(12, 'relabel/12.txt', train_metadata, {'0':0.0, '1':0.0, '2':0.3, '3':0.75, '4':1.0, '5':1.0})
    relabel_me(8, 'relabel/8.txt', train_metadata, {'1': 0.0, '2':0.5, '3':1.0})
    relabel_me(9, 'relabel/9.txt', train_metadata, {'0':0.0, '1':0.0, '2':0.5, '3':0.75, '4':1.0, '5':1.0})
    relabel_me(15, 'relabel/15.txt', train_metadata, {'0':0.0, '1':0.0, '2':0.0, '3':0.6, '4':1.0, '5':1.0})

    for label in [1, 6, 12]:
        relabel_auto(label, train_metadata, 'auto-relabel2/')

    relabel_eleven('relabel/11.txt', train_metadata)

    d = {}

    with open('data/default_sort.txt') as df:
        cnt = 0
        for x in df:
            d[x.strip()] = cnt
            cnt += 1
    
    train_metadata.sort(key=lambda x : d[x.image_name])

    validation_set = set()
    class_counter = {}
    class_counter[0] = 20000
    #class_counter[1] = 3000
    #class_counter[2] = 3000
    #class_counter[3] = 3000
    #class_counter[4] = 5000
    #class_counter[5] = 2400
    #class_counter[6] = 2000
    #class_counter[7] = 2000
    #class_counter[8] = 2000
    #class_counter[9] = 2000
    #class_counter[10] = 2000
    #before 2500
    #class_counter[11] = 1500
    #class_counter[12] = 2000
    class_counter[13] = 5000
    class_counter[14] = 5000
    #class_counter[15] = 2000
    class_counter[16] = 7000
    #class_counter[17] = 5000
    class_counter[18] = 500

    non_custom_validation = [0,6,13,14,16,18]

    validation_set = set()

    # add positives
    for label in [11]:

        label_counter = class_counter.get(label, 0)
        if label_counter <= 0:
            continue

        for metadata in train_metadata[500000:]:
            if label in metadata.image_labels:
                validation_set.add(metadata)
                metadata.metric_labels.add(label)
                label_counter -= 1

                if label_counter <= 0:
                    break

    # add manually labeled positives
    custom_val = {}

    relabel_vals = {}
    relabel_vals[1] = {'0':0.0, '1':0.0, '2':0.5, '3':0.8, '4':1.0, '5':1.0}
    relabel_vals[2] = {'0':0.0, '1':0.0, '2':0.5, '3':0.8, '4':1.0, '5':1.0}
    relabel_vals[3] = {'0':0.0, '1':0.0, '2':0.5, '3':0.8, '4':1.0, '5':1.0}
    relabel_vals[4] = {'0':0.0, '1':0.0, '2':0.5, '3':0.8, '4':1.0, '5':1.0}
    relabel_vals[5] = {'0':0.0, '1':0.0, '2':0.5, '3':0.8, '4':1.0, '5':1.0}

    relabel_vals[6] = {'0':0.0, '1':0.0, '2':0.5, '3':0.8, '4':1.0, '5':1.0}

    relabel_vals[7] = {'0':0.0, '1':0.0, '2':0.2, '3':0.75, '4':1.0, '5':1.0}

    relabel_vals[8] = {'0':0.0, '1':0.0, '2':0.5, '3':0.8, '4':1.0, '5':1.0}
    relabel_vals[9] = {'0':0.0, '1':0.0, '2':0.00, '3':0.5, '4':1.0, '5':1.0}

    relabel_vals[10] = {'0':0.0, '1':0.0, '2':0.00, '3':0.8, '4':1.0, '5':1.0}

    relabel_vals[12] = {'0':0.0, '1':0.0, '2':0.5, '3':0.8, '4':1.0, '5':1.0}
    relabel_vals[15] = {'0':0.0, '1':0.0, '2':0.5, '3':0.8, '4':1.0, '5':1.0}

    relabel_vals[17] = {'0':0.0, '1':0.0, '2':0.5, '3':1.0, '4':1.0, '5':1.0}


    for label in [1,2,3,4,5,6,7,8,9,10,12,15,17]:
        if label in relabel_vals:
            relabeler = relabel_vals[label]
            l = get_validation_list(label, relabeler)
        else:
            l = get_validation_list(label)
        
        for x in l:
            who = x[0]
            value = x[1]

            items = custom_val.get(who, [])
            items.append((label, value))
            custom_val[who] = items

    for metadata in train_metadata:
        who = metadata.image_name 
        if who in custom_val:
            items = custom_val[who]
            for item in items:
                label, value = item[0], item[1]

                validation_set.add(metadata)

                if not (label in [8,9,12,15] and value == 0.5):
                    # TODO remove
                    metadata.metric_labels.add(label)

                metadata.relabel[label] = value

    for metadata in train_metadata[-55000:]:
        if 11 not in metadata.image_labels and 15 not in metadata.image_labels:
            validation_set.add(metadata)

            for label in non_custom_validation:
                if label in metadata.image_labels:
                    metadata.metric_labels.add(label)

    for metadata in train_metadata[:30000]:
        if 11 not in metadata.image_labels and 15 not in metadata.image_labels:
            validation_set.add(metadata)

            for label in non_custom_validation:
                if label in metadata.image_labels:
                    metadata.metric_labels.add(label)

    for metadata in validation_set:
        for label in range(18):
            if not label in metadata.image_labels:
                metadata.metric_labels.add(label)
    
    train_metadata = [x for x in train_metadata if x not in validation_set]

    validation_metadata = list(validation_set)

    return train_metadata, validation_metadata