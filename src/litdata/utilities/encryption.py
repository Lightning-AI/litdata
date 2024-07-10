import base64
import os
from abc import ABC, abstractmethod

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class Encryption(ABC):
    """Base class for encryption algorithm."""

    @abstractmethod
    def encrypt(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def decrypt(self, data: bytes) -> bytes:
        pass


class FernetEncryption(Encryption):
    """Encryption for the Fernet package.

    Adapted from: https://cryptography.io/en/latest/fernet/

    """

    def __init__(self, passsword: str) -> None:
        super().__init__()
        self.passsword = passsword
        self.fernet = Fernet(self._derive_key(passsword))
        self.extension = "fernet"

    def encrypt(self, data: bytes) -> bytes:
        return self.fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        return self.fernet.decrypt(data)

    def _derive_key(self, password: str) -> bytes:
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))


class RSAEncryption:
    """Encryption for the RSA package.

    Adapted from: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/rsa/

    """

    def __init__(self, private_key_path: str = None, public_key_path: str = None) -> None:
        if private_key_path:
            self.private_key = self._load_private_key(private_key_path)
        else:
            self.private_key = None

        if public_key_path:
            self.public_key = self._load_public_key(public_key_path)
        else:
            self.public_key = None

        if not private_key_path and not public_key_path:
            self.private_key, self.public_key = self._generate_keys()
        self.extension = "rsa"

    def _load_private_key(self, path: str):
        with open(path, "rb") as key_file:
            return serialization.load_pem_private_key(
                key_file.read(),
                password=None,
            )

    def _load_public_key(self, path: str):
        with open(path, "rb") as key_file:
            return serialization.load_pem_public_key(key_file.read())

    def _generate_keys(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()
        return private_key, public_key

    def encrypt(self, data: bytes) -> bytes:
        if not self.public_key:
            raise AttributeError("Public key not found.")
        return self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def decrypt(self, data: bytes) -> bytes:
        if not self.private_key:
            raise AttributeError("Private key not found.")
        return self.private_key.decrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def save_keys(self, private_key_path: str, public_key_path: str, password: str = None) -> None:
        encryption_algorithm = (
            serialization.BestAvailableEncryption(password.encode()) if password else serialization.NoEncryption()
        )

        with open(private_key_path, "wb") as f:
            f.write(
                self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=encryption_algorithm,
                )
            )
        with open(public_key_path, "wb") as f:
            f.write(
                self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
