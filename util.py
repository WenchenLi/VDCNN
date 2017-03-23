import wget
import os
import pandas as pd
import numpy as np


def download_file(url):
    # Create file-name
    local_filename = url.split('/')[-1]
    if os.path.isfile(local_filename):
        print("The file %s already exist in the current directory\n" % local_filename)
    else:
        # Download
        print("downloading ...\n")
        wget.download(url)
        print('saved data\n')


def load_file(infile):
    """
    Takes .csv and returns loaded data along with labels
    """
    print("processing data frame: %s" % infile)
    # Get data from windows blob
    download_file('https://%s.blob.core.windows.net/%s/%s' % (AZ_ACC, AZ_CONTAINER, infile))
    # load data into dataframe
    df = pd.read_csv(infile,
                     header=None,
                     names=['sentiment', 'summary', 'text'])
    # concat summary, review; trim to 1014 char; reverse; lower
    df['rev'] = df.apply(lambda x: "%s %s" % (x['summary'], x['text']), axis=1)
    df.rev = df.rev.str[:FEATURE_LEN].str[::-1].str.lower()
    # store class as nparray
    df.sentiment -= 1
    y_split = np.asarray(df.sentiment, dtype='bool')
    print("finished processing data frame: %s" % infile)
    print("data contains %d obs, each epoch will contain %d batches" % (df.shape[0], df.shape[0] // BATCH_SIZE))
    return df.rev, y_split


def load_data_frame(X_data, y_data, batch_size=128, shuffle=False):
    """
    For low RAM this methods allows us to keep only the original data
    in RAM and calculate the features (which are orders of magnitude bigger
    on the fly). This keeps only 10 batches worth of features in RAM using
    asynchronous programing and yields one DataBatch() at a time.
    """

    if shuffle:
        idx = X_data.index
        assert len(idx) == len(y_data)
        rnd = np.random.permutation(idx)
        X_data = X_data.reindex(rnd)
        y_data = y_data[rnd]

    # Dictionary to create character vectors
    char_index = dict((c, i + 2) for i, c in enumerate(ALPHABET))

    # Yield processed batches asynchronously
    # Buffy 'batches' at a time
    def async_prefetch_wrp(iterable, buffy=30):
        poison_pill = object()

        def worker(q, it):
            for item in it:
                q.put(item)
            q.put(poison_pill)

        queue = Queue.Queue(buffy)
        it = iter(iterable)
        thread = threading.Thread(target=worker, args=(queue, it))
        thread.daemon = True
        thread.start()
        while True:
            item = queue.get()
            if item == poison_pill:
                return
            else:
                yield item

    # Async wrapper around
    def async_prefetch(func):
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            return async_prefetch_wrp(func(*args, **kwds))

        return wrapper

    @async_prefetch
    def feature_extractor(dta, val):
        # Yield mini-batch amount of character vectors
        # X_split = np.zeros([batch_size, 1, FEATURE_LEN, len(ALPHABET)], dtype='bool')
        X_split = np.zeros([batch_size, 1, FEATURE_LEN, 1], dtype='int')
        for ti, tx in enumerate(dta):
            chars = list(tx)
            for ci, ch in enumerate(chars):
                if ch in ALPHABET:
                    X_split[ti % batch_size][0][ci] = char_index[ch]
                    # X_split[ti % batch_size][0][ci] = np.array(character_hash[ch], dtype='bool')

            # No padding -> only complete batches processed
            if (ti + 1) % batch_size == 0:
                yield mx.nd.array(X_split), mx.nd.array(val[ti + 1 - batch_size:ti + 1])
                # X_split = np.zeros([batch_size, 1, FEATURE_LEN, len(ALPHABET)], dtype='bool')
                X_split = np.zeros([batch_size, 1, FEATURE_LEN, 1], dtype='int')

    # Yield one mini-batch at a time and asynchronously process to keep 4 in queue
    for Xsplit, ysplit in feature_extractor(X_data, y_data):
        yield DataBatch(data=[Xsplit], label=[ysplit])