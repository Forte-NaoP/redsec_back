import torch.nn.modules.conv

header = """
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))

from service import *
from seal import *
import torch
import re
"""

main = r"""
if __name__ == '__main__':

    cwd = sys.path[0]
    sys.path.append(cwd)

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
    
    conv_strides, conv_paddings, pool_strides, pool_paddings, fc_outputs = [], [], [], [], []
    import inspect
    forward = list(map(lambda x: x.split('=')[1].strip(),
                       filter(lambda x: not x.startswith('#'),
                              inspect.getsource(model.CNN.forward).splitlines()[1:-1])))
                              
    member_pattern = re.compile(r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)')                          
    for line in forward:
        member_name = member_pattern.findall(line)
        if member_name:
            member = getattr(model_runner, member_name[0][0])
            if isinstance(member, torch.nn.modules.conv.Conv2d):
                conv_strides.append(member.stride[0])
                conv_paddings.append(member.padding[0])
            elif isinstance(member, torch.nn.modules.pooling.AvgPool2d):
                pool_strides.append(member.stride)
                pool_paddings.append(member.padding)
            elif isinstance(member, torch.nn.modules.linear.Linear):
                fc_outputs.append(member.out_features)
            else:
                raise NotImplementedError
    
    conv_weights, conv_bias, fc_weights, fc_bias = [], [], [], []
    
    for name, value in model_cnn.items():
        layer_type, value_type = name.split('.')
        if layer_type.startswith('Conv'):
            if value_type == 'weight':
                conv_weights.append(value)
            elif value_type == 'bias':
                conv_bias.append(value)
        elif layer_type.startswith('FC'):
            if value_type == 'weight':
                fc_weights.append(value)
            elif value_type == 'bias':
                fc_bias.append(value)
                
    image_prefix = os.environ['IMAGE_PATH']
    input_path = os.path.join(cwd, 'images', f'{image_prefix}_input')
    output_path = os.path.join(cwd, 'images', f'{image_prefix}_output')
    
    ctxt = seal.Ciphertext()
    ctxt.load(context, input_path)
    
    result = HE_inference(evaluator, ckks_encoder, galois_key, relin_keys, ctxt,
                          conv_weights, conv_bias, fc_weights, fc_bias,
                          28, model.data_size, 
                          conv_strides, conv_paddings, pool_strides, pool_paddings, fc_outputs)
                          
    result.save(output_path)
"""

HE_inference = """

def HE_inference(
        evaluator, ckks_encoder, galois_key, relin_keys, ctxt,
        conv_weights, conv_biases,
        fc_weights, fc_biases,
        image_size, data_size,
        conv_strides, conv_paddings, pool_strides, pool_paddings, fc_outputs
):
    result = [ctxt]
    const_param = 1
    
    {body}
    
    return result
"""

conv = """
    result, OH, S, const_param = conv2d_layer_converter_(evaluator, ckks_encoder, galois_key, relin_keys, result,
                                                         conv_weights[{idx}], conv_biases[{idx}],
                                                         input_size=image_size, real_input_size=image_size,
                                                         padding=conv_paddings[{idx}], stride=conv_strides[{idx}],
                                                         data_size=data_size, const_param=const_param)
"""

pool = """
    result, OH, S, const_param = average_pooling_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys,
                                                                 result,
                                                                 kernel_size={kernel_size},
                                                                 input_size=image_size, real_input_size=OH,
                                                                 padding=pool_paddings[{idx}], 
                                                                 stride=pool_strides[{idx}], 
                                                                 tmp_param=S, data_size=data_size,
                                                                 const_param=const_param)
"""

flatten = """
    result = flatten(evaluator, ckks_encoder, galois_key, relin_keys, result, OH, OH, S,
                        input_size=image_size, data_size=data_size, const_param=const_param)
"""

fc = """
    result = fc_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys, result,
                                    fc_weights[{idx}], fc_biases[{idx}], data_size=data_size)
"""

relu = """
    result, const_param = approximated_ReLU_converter(evaluator, ckks_encoder, data_size, {real_size}, 
                                                        relin_keys, result, 0, {const_param})
"""

import sys
import re
import os

if __name__ == "__main__":

    cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../', os.environ['WORK_DIR'])
    sys.path.append(cwd)

    import model

    model_cnn = model.CNN()
    model_repr = repr(model_cnn)

    conv_pattern = re.compile(
        r'Conv2d\(\d+, \d+, kernel_size=\(\d+, \d+\), stride=\((\d+, \d+)\)(?:, padding=\((\d+, \d+)\))?\)')
    pool_pattern = re.compile(r'AvgPool2d\(kernel_size=\d+, stride=(\d+), padding=(\d+)\)')
    fc_pattern = re.compile(r'Linear\(in_features=(\d+), out_features=(\d+), bias=True\)')

    conv_matches = conv_pattern.findall(model_repr)
    pool_matches = pool_pattern.findall(model_repr)
    fc_matches = fc_pattern.findall(model_repr)

    conv_strides = [int(match[0].split(',')[0]) for match in conv_matches]
    conv_paddings = [int(match[1]) if match[1] else 0 for match in conv_matches]
    pool_strides = [int(match[0]) for match in pool_matches]
    pool_paddings = [int(match[1]) for match in pool_matches]
    fc_outputs = [int(match[1]) for match in fc_matches]

    conv_strides, conv_paddings, pool_strides, pool_paddings, fc_outputs = [], [], [], [], []

    import inspect
    forward = list(map(lambda x: x.split('=')[1].strip(),
                       filter(lambda x: not x.startswith('#'),
                              inspect.getsource(model.CNN.forward).splitlines()[1:-1])))
    print(forward)

    body = ""

    member_pattern = re.compile(r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)')
    conv_idx, pool_idx, fc_idx = 0, 0, 0
    after_flatten = False
    for line in forward:
        member_name = member_pattern.findall(line)
        if not member_name:
            if line.startswith('torch.flatten'):
                body += flatten
                after_flatten = True
            elif line.startswith('apporximate_relu'):
                if after_flatten:
                    body += relu.format(real_size=fc_outputs[fc_idx-1], const_param=1)
                else:
                    body += relu.format(real_size='data_size', const_param='const_param')
        else:
            member = getattr(model_cnn, member_name[0][0])
            print(type(member))
            if isinstance(member, torch.nn.modules.conv.Conv2d):
                body += conv.format(idx=conv_idx)
                conv_strides.append(member.stride[0])
                conv_paddings.append(member.padding[0])
                conv_idx += 1
            elif isinstance(member, torch.nn.modules.pooling.AvgPool2d):
                body += pool.format(kernel_size=member.kernel_size, idx=pool_idx)
                pool_strides.append(member.stride)
                pool_paddings.append(member.padding)
                pool_idx += 1
            elif isinstance(member, torch.nn.modules.linear.Linear):
                body += fc.format(idx=fc_idx)
                fc_outputs.append(member.out_features)
                fc_idx += 1
            else:
                raise NotImplementedError

    code = header + HE_inference.format(body=body) + main
    print(conv_strides, conv_paddings, pool_strides, pool_paddings, fc_outputs)
    with open(os.path.join(cwd, 'inference.py'), 'w') as f:
        f.write(code)






