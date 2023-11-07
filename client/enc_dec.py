from seal import *
import os
from PIL import Image
import torchvision.transforms as transforms

if __name__ == "__main__":

    mode = -1
    while mode != 0 and mode != 1:
        mode = int(input("enter the mode (enc: 0, dec: 1): "))

    file_name = input("enter the file name: ")
    while not os.path.isfile(file_name):
        file_name = input("file does not exist\nenter the file name: ")

    cwd = './'

    parms = EncryptionParameters(scheme_type.ckks)
    parms.load(os.path.join(cwd, 'ckks_parms'))
    context = SEALContext(parms)
    ckks_encoder = CKKSEncoder(context)

    if mode == 0:
        public_key = PublicKey()
        public_key.load(context, os.path.join(cwd, 'pub_key'))
        encryptor = Encryptor(context, public_key)
        with open(file_name+'_ctxt', "wb") as _o:

            transform = transforms.Compose([
                transforms.ToTensor()
            ])

            data = transform(Image.open(file_name))
            ctxt = encryptor.encrypt(ckks_encoder.encode(data))
            _o.write(ctxt)
    else:
        secret_key = SecretKey()
        secret_key.load(context, os.path.join(cwd, 'secret_key'))
        decryptor = Decryptor(context, secret_key)

        with open(file_name, "rb") as _i, open(file_name+'_ctxt', "wb") as _o:
            ctxt = _i.read()
            ptxt = ckks_encoder.decode(decryptor.decrypt(ctxt))
            _o.write(ptxt)
