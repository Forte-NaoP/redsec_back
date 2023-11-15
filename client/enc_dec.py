import seal
from seal import *
import os

import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

import cnn_model

if __name__ == "__main__":

    mode = -1
    while mode != 0 and mode != 1:
        mode = int(input("enter the mode (enc: 0, dec: 1): "))

    file_name = input("enter the file name: ")
    while not os.path.isfile(file_name):
        file_name = input("file does not exist\nenter the file name: ")

    cwd = './key'

    parms = EncryptionParameters(scheme_type.ckks)
    parms.load(os.path.join(cwd, 'ckks_parms'))
    context = SEALContext(parms)
    ckks_encoder = CKKSEncoder(context)

    scale = context.key_context_data().parms().coeff_modulus()[1].bit_count()
    print('scale: ', scale)

    if mode == 0:
        public_key = PublicKey()
        public_key.load(context, os.path.join(cwd, 'pub_key'))
        encryptor = Encryptor(context, public_key)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        img = Image.open(file_name).convert('L')
        width, height = img.size
        img_data = transform(img)
        img_data = [np.array(img_data)]

        data = []
        data.extend(img_data[0].flatten())
        data.extend([0] * (cnn_model.data_size - width*height))
        data = torch.Tensor(data)

        ctxt = encryptor.encrypt(ckks_encoder.encode(data, 2**scale))
        ctxt.save(file_name+'_ctxt')

    else:
        secret_key = SecretKey()
        secret_key.load(context, os.path.join(cwd, 'secret_key'))
        decryptor = Decryptor(context, secret_key)

        ctxt = seal.Ciphertext()
        ctxt.load(context, file_name)
        ptxt = ckks_encoder.decode(decryptor.decrypt(ctxt))[:10]

        print(ptxt.index(max(ptxt)))


