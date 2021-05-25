import numpy as np
import pickle
import csv

LABEL_COUNT = 19

tsv_file = 'data/kaggle_2021.tsv'
relabel_mapper = {'0':0.0, '1':0.0, '2':0.25, '3':0.5, '4':0.9, '5':1.0}
mapper_11 = {'1':0.0, '2':0.05, '3':0.2, '4':0.65, '5':1.0}
mapper_8 = {'1': 0.0, '2':0.5, '3':1.0}

def relabel_me6(filename, image_metadata):

    r = get_latest_antibody_relabel(filename)
    label = 6
    for metadata in image_metadata:
        if label in metadata.image_labels:
            if metadata.image_name.rsplit('_',1)[0] in r:
                value = 0.51
            else:
                value = 1.0
            metadata.relabel[label] = value

def relabel_me(label, filename, image_metadata, rmapper = relabel_mapper):

    relabel = get_latest_relabel(filename, rmapper)
    c = 0
    for metadata in image_metadata:
        if metadata.image_name in relabel:
            if not label in metadata.image_labels:
                metadata.image_labels.add(label)
            value = relabel[metadata.image_name]
            metadata.relabel[label] = value

        elif label in metadata.image_labels:
            c += 1
            value = 0.69
            metadata.relabel[label] = value
    print("UNLABELED:", label, c)


def relabel_image_metadata(label, filename, image_metadata):
    if label == 11:
        raise Exception('class 11')
        relabel = get_latest_relabel(filename, rmapper = mapper_11)
    else:    
        relabel = get_latest_relabel(filename)

    for metadata in image_metadata:
        if metadata.image_name in relabel:
            value = relabel[metadata.image_name]
            metadata.relabel[label] = value

            if value <= 0.5:
                metadata.image_labels.remove(label)

                if metadata.metric_labels:
                    metadata.metric_labels.remove(label)
            else:
                if label == 11:
                    metadata.image_labels = set([11])
                    if metadata.metric_labels:
                        metadata.metric_labels = set([11])

def relabel_eleven(filename, image_metadata):
    relabel = get_latest_relabel(filename, rmapper = mapper_11)
    cnt = 0
    for metadata in image_metadata:
        if metadata.image_name in relabel:
            value = relabel[metadata.image_name]
            metadata.relabel[11] = value

            if value >= 0.2:
                metadata.image_labels = set([11])

        elif 11 in metadata.image_labels:
            print(metadata.image_name)
            cnt += 1
    print(cnt)

def get_latest_relabel(filename, rmapper = None):

    result = {}

    with open(filename) as f:
        for row in f:
            row = row.strip().split(' ')
            if rmapper:
                result[row[0]] = rmapper[row[1]]
            else:
                result[row[0]] = relabel_mapper[row[1]]
    
    return result

def get_latest_relabel_old(filename):

    result = set()

    with open(filename) as f:
        for row in f:
            row = row.strip().split(' ')
            if row[1] != '3':
                result.add(row[0])
            else:
                result.discard(row[0])
    
    return list(result)

def get_latest_antibody_relabel(filename):

    publichpa_antibody_mapper = {row[0].split('/')[-1] : row[0].split('/')[-2] for row in csv.reader(open(tsv_file, 'r')) if row[0] != 'Image' and row[3] == 'False'}
    rev_antibody_mapper = [(row[0].split('/')[-2], row[0].split('/')[-1]) for row in csv.reader(open(tsv_file, 'r')) if row[0] != 'Image' and row[3] == 'False']
    result = get_latest_relabel_old(filename)
    
    result = set(publichpa_antibody_mapper.get(r, r) for r in result)
    result = set(x[1] for x in rev_antibody_mapper if x[0] in result)

    return result

def parse_classes(classes):
    if not classes:
        return [18]
    
    return list(set(map(int, classes.split('|'))))

def create_data_from_train(csv_file = 'data/train.csv'):
    train = [(row[0], parse_classes(row[1])) for row in csv.reader(open(csv_file, 'r')) if row[0] != 'ID']
    return train

def create_data_from_publichpa_full_length(tsv_file = 'data/kaggle_2021.tsv'):
    publichpa = {row[0].split('/')[-1] : parse_classes(row[4]) for row in csv.reader(open(tsv_file, 'r')) if row[0] != 'Image' and row[3] == 'False'}
    return list(publichpa.items())

def get_border_and_garbage_images():

    mp = {}

    images = create_data_from_train()
    images.extend(create_data_from_publichpa_full_length())

    for x in pickle.load(open( "pckl/train-values.p", "rb" ) ):
        mp[x[0]] = x[1]

    for x in pickle.load(open( "pckl/publichpa-values.p", "rb" ) ):
        mp[x[0]] = x[1]

    result = []
    cnt = 0

    d = {}

    for img in images:
        vals = mp.get(img[0], [])
        cnt += len(vals)
        for i in range(len(vals)):
            if vals[i] < 1.0:
                d[vals[i]] = d.get(vals[i], 0) + 1
                result.append(img[0] + '_' + str(i))
        
    result.sort()

    return result

def get_labels_to_indices_map(label_ids):
    label_ids = sorted(set(label_ids))
    labels_to_incides_map = {label : index for index, label in enumerate(label_ids, 0)}

    return labels_to_incides_map

def get_mask_bounding_box(mask):
    true_points = np.argwhere(mask)
    
    if not true_points.any():
        return np.array([0, 0]), np.array([0, 0])
    
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)

    return top_left, bottom_right

def get_cropped_mask(mask, cell_id):
    top_left, bottom_right = get_mask_bounding_box((mask == cell_id))

    mask = mask[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1].copy()
    mask[mask != cell_id] = 0
    mask[mask != 0] = 1

    return mask

def validation_count(validation_metadata):
    count_map = {label:0 for label in range(LABEL_COUNT)}

    for metadata in validation_metadata:
        for label in metadata.image_labels:
            if label in metadata.metric_labels:
                count_map[label] = count_map[label] + 1
    
    return count_map

def train_count(train_metadata):
    count_map = {label:0 for label in range(LABEL_COUNT)}

    for metadata in train_metadata:
        for label in metadata.image_labels:
            count_map[label] = count_map[label] + 1
    
    return count_map

def train_validation_print(train_metadata, validation_metadata):
    train_map = train_count(train_metadata)
    validation_map = validation_count(validation_metadata)

    train_sum = sum(train_map.values())
    validation_sum = sum(validation_map.values())

    print('%3s\t%10s\t%10s\t%10s' % ('ID', 'TRAIN', 'VAL', '%'))
    print('-' * 50)
    for label in range(LABEL_COUNT):
        print('%3d\t%10d\t%10d\t%10d' % (label, train_map[label], validation_map[label], validation_map[label] * 100 // (train_map[label] + validation_map[label])))
    print('-' * 50)
    print('%3s\t%10d\t%10d\t%10d' % ('ALL', train_sum, validation_sum, validation_sum * 100 // (train_sum + validation_sum)))

    train_sum = len(train_metadata)
    validation_sum = len(validation_metadata)

    print('%3s\t%10d\t%10d\t%10d' % ('ALL', train_sum, validation_sum, validation_sum * 100 // (train_sum + validation_sum)))