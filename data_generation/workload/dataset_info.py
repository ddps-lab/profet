dataset_16 = {
    'num_classes': 10,
    'img_rows': 16,
    'img_cols': 16,
    'img_channels': 3,
    'num_data': 50000,
    'num_test': 10000
}

dataset_32 = {
    'num_classes': 10,
    'img_rows': 32,
    'img_cols': 32,
    'img_channels': 3,
    'num_data': 50000,
    'num_test': 10000
}

dataset_64 = {
    'num_classes': 100,
    'img_rows': 64,
    'img_cols': 64,
    'img_channels': 3,
    'num_data': 5000,
    'num_test': 1000
} 

dataset_128 = {
    'num_classes': 100,
    'img_rows': 128,
    'img_cols': 128,
    'img_channels': 3,
    'num_data': 5000,
    'num_test': 1000
}

dataset_224 = {
    'num_classes': 1000,
    'img_rows': 224,
    'img_cols': 224,
    'img_channels': 3,
    'num_data': 2000,
    'num_test': 500
}

dataset_256 = {
    'num_classes': 1000,
    'img_rows': 256,
    'img_cols': 256,
    'img_channels': 3,
    'num_data': 2000,
    'num_test': 500
}

def select_dataset(dataset_name):
    if dataset_name == 16:
        return dataset_16
    elif dataset_name == 32:
        return dataset_32
    elif dataset_name == 64:
        return dataset_64
    elif dataset_name == 128:
        return dataset_128
    elif dataset_name == 224:
        return dataset_224
    elif dataset_name == 256:
        return dataset_256
