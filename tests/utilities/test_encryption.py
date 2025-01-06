import os

import pytest

from litdata.utilities.encryption import FernetEncryption, RSAEncryption


def test_fernet_encryption(tmpdir):
    password = "password"
    data = b"test data"
    fernet = FernetEncryption(password)
    encrypted_data = fernet.encrypt(data)
    decrypted_data = fernet.decrypt(encrypted_data)
    assert data == decrypted_data
    assert data != encrypted_data
    assert decrypted_data != encrypted_data
    assert isinstance(encrypted_data, bytes)
    assert isinstance(decrypted_data, bytes)
    assert isinstance(fernet.algorithm, str)
    assert fernet.algorithm == "fernet"
    assert fernet.password == password
    assert fernet.key == fernet._derive_key(password, fernet.salt)
    assert isinstance(fernet._derive_key(password, os.urandom(16)), bytes)

    # ------ Test for ValueError ------
    with pytest.raises(ValueError, match="The provided `level` should be either `sample` or `chunk`"):
        fernet = FernetEncryption(password, level="test")

    # ------ Test for saving and loading fernet instance------
    file_path = tmpdir.join("fernet.txt")
    fernet.save(file_path)
    fernet_loaded = FernetEncryption.load(file_path, password)
    assert fernet_loaded.password == password
    assert fernet_loaded.level == fernet.level
    assert fernet_loaded.salt == fernet.salt
    assert fernet_loaded.key == fernet.key

    decrypted_data_loaded = fernet_loaded.decrypt(encrypted_data)
    assert data == decrypted_data_loaded


def test_rsa_encryption(tmpdir):
    password = "password"
    data = b"test data"
    rsa = RSAEncryption(password)
    encrypted_data = rsa.encrypt(data)
    decrypted_data = rsa.decrypt(encrypted_data)
    assert data == decrypted_data
    assert data != encrypted_data
    assert decrypted_data != encrypted_data
    assert isinstance(encrypted_data, bytes)
    assert isinstance(decrypted_data, bytes)
    assert isinstance(rsa.algorithm, str)
    assert rsa.algorithm == "rsa"

    # ------ Test for ValueError ------
    with pytest.raises(ValueError, match="The provided `level` should be either `sample` or `chunk`"):
        rsa = RSAEncryption(password, level="test")

    # ------ Test for saving and loading rsa instance------
    file_path = tmpdir.join("rsa.txt")
    rsa.save(file_path)

    rsa_loaded = RSAEncryption.load(file_path, password)
    assert rsa_loaded.level == rsa.level
    assert rsa_loaded.password == rsa.password

    decrypted_data_loaded = rsa_loaded.decrypt(encrypted_data)
    assert data == decrypted_data_loaded

    # ------ Test for saving and loading rsa instance with password------
    private_key_path = tmpdir.join("rsa_private.pem")
    public_key_path = tmpdir.join("rsa_public.pem")
    rsa.save_keys(private_key_path, public_key_path)

    rsa_keys_loaded = RSAEncryption(password)
    rsa_keys_loaded.load_keys(private_key_path, public_key_path, password)

    decrypted_data_loaded = rsa_keys_loaded.decrypt(encrypted_data)
    assert data == decrypted_data_loaded
