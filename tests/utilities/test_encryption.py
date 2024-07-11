from litdata.utilities.encryption import FernetEncryption, RSAEncryption


def test_fernet_encryption():
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
    assert isinstance(fernet.extension, str)
    assert fernet.extension == "fernet"
    assert fernet.password == password
    assert isinstance(fernet._derive_key(password), bytes)
    assert isinstance(fernet._derive_key(password), bytes)


def test_rsa_encryption():
    data = b"test data"
    rsa = RSAEncryption()
    encrypted_data = rsa.encrypt(data)
    decrypted_data = rsa.decrypt(encrypted_data)
    assert data == decrypted_data
    assert data != encrypted_data
    assert decrypted_data != encrypted_data
    assert isinstance(encrypted_data, bytes)
    assert isinstance(decrypted_data, bytes)
    assert isinstance(rsa.extension, str)
    assert rsa.extension == "rsa"
