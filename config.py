"""
created by VinnyHu
used for parameters

"""

class DefaultConfig(object):


    model = 'PCNN'


    """
    some paths
    """
    train_data_path = ''
    test_data_path = ''
    val_data_path = ''
    """
    parameters for CNN
    """
    kernel_sizes = 3
    kernel_num = 230



    pos_embedding_dim = 5
    word_embedding_dim = 50
    vocab_size = None
    pos_size = 5
    batch_size = 64
    dropout = 0.5
    padding_size = 1
    max_length = 4

    blank_padding = True
    entpair_bag = False
    bag_size = None
    mode = None
    shuffle = True

    num_classes = 52
    num_workers = 8

    loss_weight = False
    lr = 0.5
    weight_decay = 0
    max_epoch = 10