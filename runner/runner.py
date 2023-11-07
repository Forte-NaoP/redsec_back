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


def HE_inference(
        evaluator, ckks_encoder, galois_key, relin_keys, ctxt,
        conv_weights, conv_biases,
        fc_weights, fc_biases,
        image_size, paddings, strides, data_size
):

    result = [ctxt]

    result, OH, S, const_param = conv2d_layer_converter_(evaluator, ckks_encoder, galois_key, relin_keys, result,
                                                         conv_weights[0], conv_biases[0],
                                                         input_size=image_size, real_input_size=image_size,
                                                         padding=paddings[0], stride=strides[0],
                                                         data_size=data_size, const_param=1)

    result, OH, S, const_param = average_pooling_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys,
                                                                 result,
                                                                 kernel_size=2,
                                                                 input_size=image_size, real_input_size=OH,
                                                                 padding=0, stride=2, tmp_param=S,
                                                                 data_size=data_size, const_param=const_param)

    result = flatten(evaluator, ckks_encoder, galois_key, relin_keys, result, OH, OH, S,
                     input_size=image_size, data_size=data_size, const_param=const_param)

    result = fc_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys, result,
                                fc_weights[0], fc_biases[0], data_size=data_size)

    result, const_param = approximated_ReLU_converter(evaluator, ckks_encoder, data_size, 1024,
                                                      relin_keys, result, 0, 1)

    result = fc_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys, result,
                                fc_weights[1], fc_biases[1], data_size=data_size)

    return result


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

    with open(input_path, 'rb') as f:
        ctxt = f.read()

    strides = [1, 1, 1]
    paddings = [0, 0, 0]
    result = HE_inference(evaluator, ckks_encoder, galois_key, relin_keys, ctxt,
                          conv_weights, conv_bias, fc_weights, fc_bias,
                          28, paddings, strides, 1080)

    with open(output_path, 'wb') as f:
        f.write(result)
