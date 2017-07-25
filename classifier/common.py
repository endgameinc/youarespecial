import numpy as np
###########################################################
def schedule(epoch, start=0.2, decay=0.5, every=5):
    ''' a learning rate schedule function helper for keras
    Returns a learning rate after "epoch" epochs.
    The learning rate begins at "start", and decays every "every" epochs by "decay".
    That is learning_rate = start * decay ** (epoch // every )
    '''
    # start at "start", and multiply by "decay" every "every" epochs
    lr = start * decay**(epoch // every)
    if epoch % every == 0:
        print("learning rate = {}".format(lr))
    return lr

###########################################################
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
def summarize_performance(y_pred, y_test, name, target_fpr=0.01):
    ''' summarize model performance for a model
    Specify a targeted FP rate via "targeted_fpr".
    The function returns 
    * AUC: the area under the ROC curve, independant of thershold    
    * thresh : threshold that achieves at most targeted_fpr (if possible)
    * fixed_fpr: the FPR actually achieved at "thresh" fixed threshold
    * fixed_tpr: the TPR acutally achieved
    * confusion: confusion matrix at threshold "thesh"
    * accuracy: accuracy at threshold "thresh"'''

    print('** {} **'.format(name))
    roc_auc = roc_auc_score(np.round(y_test), y_pred)
    print('ROC AUC = {}'.format( roc_auc ) )

    fpr, tpr, thresholds = roc_curve( np.round(y_test), y_pred, drop_intermediate=False )
    ixs = np.where( fpr <= target_fpr )[0]
    if len(ixs)==0:
        print('Unable to achieve target_fpr={}, using target_fpr={} instead'.format( target_fpr, fpr[0] ) )
        ix = 0
    else:
        ix = ixs[-1] # the last threshold that achieves <= target_fpr

    thresh = thresholds[ ix ]
    fixed_fpr = fpr[ix]
    fixed_tpr = tpr[ix]
    print('threshold={}: {} TP rate @ {} FP rate'.format( thresh, fixed_tpr, fixed_fpr ) )

    confusion = confusion_matrix(np.round(y_test), y_pred > thresh )
    print('confusion matrix @ threshold:\n{}'.format( confusion ) )

    accuracy = accuracy_score( np.round(y_test), y_pred > thresh )
    print('accuracy @ threshold = {}'.format( accuracy ) )

    return roc_auc, thresh, fixed_fpr, fixed_tpr, confusion, accuracy

###########################################################
import os
def fetch_samples():
    ''' fetches malicious and benign samples form disk
    For simplicity, it assumes that samples are named by their sha256 hash, and 
    are stored in benign/ and malicious/ subdirectories.
    Returns a dict "labels", where 
      labels[sha256] = 0 for a benign sha256, or 
      labels[sha256] = 1 for a malicious sha256'''
    labels = [(sha256, 0) for sha256 in os.listdir('benign')]
    labels += [(sha256, 1) for sha256 in os.listdir('malicious')]
    labels = dict(labels)
    return labels

###########################################################
def extract_features_and_persist():
    ''' extract features and cache in sample_index.json and X.dat
    Uses fetch_samples to enumerate through samples on disk and store features.
    If features have already been extracted, reloads them from their persisted files.
    Returns:
    * X : numpy array of features (1 row per sample)
    * sample_index : dict that returns index (row) for a sample, given a sha256 key
    '''
    # extact features and persist
    # this takes about 40 minutes per 10K samples on my machine
    # enumerate samples on disk
    import time
    start_time = time.time()

    import json
    import numpy as np

    labels = fetch_samples()

    try:
        with open('sample_index.json', 'r') as infile:
            sample_index = json.load(infile)

        with open('X.dat', 'rb') as infile:
            X = np.fromfile(infile, dtype=np.float32).reshape(
                len(sample_index), -1)

        features = {sha256: X[i]
                    for sha256, i in sample_index.items() if sha256 in labels}
        # if sha26 in labels ensures that we don't store features that have been removed from disk
    except FileNotFoundError:
        sample_index = dict()
        features = {}

    import os
    from pefeatures import PEFeatureExtractor
    extractor = PEFeatureExtractor()
    for i, (sha256, ismalicious) in enumerate(labels.items()):
        if sha256 in features:
            continue
        print('{}: {} / {}'.format(sha256, i, len(labels)))
        try:
            bytez = open(os.path.join(
                'malicious' if ismalicious else 'benign', sha256), 'rb').read()
        except FileNotFoundError:
            continue  # file is missing...skip it
        features[sha256] = extractor.extract(bytez)

    # reduce labels to include only features we were able to extract
    labels = {sha256: labels[sha256] for sha256 in features.keys()}

    assert np.all(np.asarray(list(labels.keys())) == np.asarray(
        list(features.keys()))), "samples on disk and persisted features out of sync!"

    X = np.asarray(list(features.values()), dtype=np.float32)
    sample_index = {k: i for i, k in enumerate(features.keys())}

    # persist features and sample index to disk
    with open('X.dat', 'wb') as outfile:
        X.tofile(outfile)

    with open('sample_index.json', 'w') as outfile:
        json.dump(sample_index, outfile, indent=1)

    elapsed_time = time.time() - start_time
    print('took {} seconds'.format(elapsed_time))

    y = np.empty((X.shape[0],), dtype=np.float32)
    sha256list = np.empty((X.shape[0],), dtype='<U64')
    for sha256, i in sample_index.items():
        y[i] = labels[sha256]
        sha256list[i] = sha256

    return X, y, sha256list

###########################################################
def get_file_data(sha256, lab, max_file_length, EOF_TOKEN = 256):
    ''' get a chunked file from disk
    Grabs the first max_file_length samples from disk, given a sha256 and a label
    (which tells the function whether to look in "benign/" (lab=0) or "malicious/" (lab=1) subdirectories).
    If a file is smaller than max_file_length bytez, it is padded with EOF_TOKEN (default=256).
    Returns:
    * barray : numpy array of np.uint16 containing first max_file_length of file bytes requested'''
    fn = os.path.join('benign' if lab == 0 else 'malicious', sha256)
    # read up to first max_file_len bytes and turn into numpy array
    byte_vals = np.fromstring(open(fn, 'rb').read(max_file_length), dtype=np.uint8)
    barray = np.ones((max_file_length), dtype=np.uint16) * EOF_TOKEN  # special token to denote end-of-file
    barray[:len(byte_vals)] = byte_vals
    return barray

###########################################################
def generator(sample_label_pairs, samples_per_step, file_chunks, file_chunk_size):
    ''' generator helper function
    So you can train a keras model without sucking all the files into memory.
    Note that file bytes are delievered in separate chunks of file_chunks x file_chunk_size.
    This is because some GPUs may not have enough linear memory layout to hold the entire batch
    of samples in GPU memory at a time.'''
    max_file_length = file_chunks * file_chunk_size
    X_batch = []
    y_batch = []
    while True:  # loop indefinitely
        visit_order = np.random.permutation(len(sample_label_pairs))
        for i in visit_order:
            sha256, lab = sample_label_pairs[i]
            X_batch.append(get_file_data(sha256, lab, max_file_length))
            y_batch.append(lab)
            if len(y_batch) == samples_per_step:
                yield np.asarray(X_batch).reshape((-1, file_chunks, file_chunk_size)), np.asarray(y_batch)
                X_batch = []
                y_batch = []
