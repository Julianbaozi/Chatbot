import numpy as np
from random import sample

'''
 split data into train (70%), test (15%) and valid(15%)
    return tuple( (trainX, trainY), (testX,testY), (validX,validY) )

'''
def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]
    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)


'''
 generate batches from dataset
    yield (x_gen, y_gen)

    TODO : fix needed

'''
def batch_generator(data,target,batch_size,shuffle = True):
    if shuffle:
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]
        target = target[idx]
    base = 0
    offset = batch_size
    maximum = len(data)
    length = int(maximum/offset)+1
    for batch_id in range(length):
        if base+offset < maximum:
            #yild data and target(left shift by 1)
            features = data[base:base+offset]
            labels = target[base:base+offset]
            features_len,lables = sort_batch(features,labels)
            yield features_len,labels
            base += offset
#         else:
#             yield data[base:],target[base:]
def sort_batch(batch,labels):
    batch_size = batch.shape[0]
    mask = np.zeros_like(batch)
    mask[batch!=0] = 1
    lengths = np.sum(mask,axis =1)
    idx = np.arange(batch_size)
    idx_len = dict(zip(idx,lengths))
    idx_len = sorted(idx_len.items() ,key = lambda x:x[1],reverse = True)
    idx = [x for x,_ in idx_len]
    batch = batch[idx]
    labels = labels[idx]
    lengths = lengths[idx]
    return(batch,lengths),labels
    
    
    
def batch_gen(x, y, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T

'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)

'''
def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

#'''
# convert indices of alphabets into a string (word)
#    return str(word)
#
#'''
#def decode_word(alpha_seq, idx2alpha):
#    return ''.join([ idx2alpha[alpha] for alpha in alpha_seq if alpha ])
#
#
#'''
# convert indices of phonemes into list of phonemes (as string)
#    return str(phoneme_list)
#
#'''
#def decode_phonemes(pho_seq, idx2pho):
#    return ' '.join( [ idx2pho[pho] for pho in pho_seq if pho ])


'''
 a generic decode function 
    inputs : sequence, lookup

'''
def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])
