from seal import *
import os

if __name__ == "__main__":

    poly_modulus_degree = 0
    while poly_modulus_degree not in [16384, 32768]:
        poly_modulus_degree = int(input("Enter poly_modulus_degree (16384, 32768): "))

    parms = EncryptionParameters(scheme_type.ckks)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    bits_scale1 = int(input("bits_scale1 (Enc/Dec bit size): "))
    bits_scale2 = int(input("bits_scale2 (scaling bit size): "))

    max_bit_count = {16384: 438, 32768: 881}
    max_medium_scale = (max_bit_count[poly_modulus_degree] - 2 * bits_scale1) // bits_scale2
    modulus_chain_length = 9999
    while modulus_chain_length > max_medium_scale or modulus_chain_length < 3:
        modulus_chain_length = int(input(f"Enter modulus_chain_length (3 ~ {max_medium_scale}): "))

    coeff_mod_bit_sizes = [bits_scale1] + [bits_scale2] * modulus_chain_length + [bits_scale1]
    parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, coeff_mod_bit_sizes))

    scale = 2.0 ** bits_scale2
    context = SEALContext(parms)
    ckks_encoder = CKKSEncoder(context)
    slot_count = ckks_encoder.slot_count()

    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    galois_key = keygen.create_galois_keys()
    relin_keys = keygen.create_relin_keys()

    if not os.path.isdir('key'):
        os.mkdir('key')
    parms.save('key/ckks_parms')
    public_key.save('key/pub_key')
    secret_key.save('key/secret_key')
    galois_key.save('key/galois_key')
    relin_keys.save('key/relin_key')

