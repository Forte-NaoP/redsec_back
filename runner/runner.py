from service import *
from seal import *
from torchvision import datasets, transforms
import numpy as np
import torch
import h5py, os
import time
import math
import pathlib
import sys

if __name__ == '__main__':

    cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../', os.environ['WORK_DIR'])
    sys.path.append(cwd)
    print(os.path.join(cwd, 'ckks_parms'))
    parms = EncryptionParameters(scheme_type.ckks)
    parms.load(os.path.join(cwd, 'ckks_parms'))
    context = SEALContext(parms)
    ckks_encoder = CKKSEncoder(context)
    public_key = PublicKey()
    public_key.load(context, os.path.join(cwd, 'pub_key'))

    galois_key = GaloisKeys()
    galois_key.load(context, os.path.join(cwd, 'galois_key'))

    relin_keys = RelinKeys()
    relin_keys.load(context, os.path.join(cwd, 'relin_key'))

    encryptor = Encryptor(context, public_key)
    evaluator = Evaluator(context)

    import model
    model_cnn = torch.load(os.path.join(cwd, 'weight'), map_location=torch.device('cpu'))
    model_runner = model.CNN()
    model_runner.load_state_dict(model_cnn)

    conv_weights, conv_bias, fc_weights, fc_bias = [], [], [], []

    layers = []

    for name, _ in model_cnn.items():
        layer_type, value_type = name.split('.')
        layers.append(name)

    print('\n'.join(layers))
    image_prefix = os.environ['IMAGE_PATH']
    input_path = os.path.join(cwd, 'images', f'{image_prefix}_input')
    output_path = os.path.join(cwd, 'images', f'{image_prefix}_output')

    import shutil
    with open(input_path, "rb") as im, open(output_path, "wb") as om:
        shutil.copyfileobj(im, om)

    import time
    time.sleep(60)
    print('inference done')