import base64
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, Union, get_args

from litdata.constants import _CRYPTOGRAPHY_AVAILABLE

if _CRYPTOGRAPHY_AVAILABLE:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


EncryptionLevel = Literal["sample", "chunk"]


class Encryption(ABC):
    """Base class for encryption algorithm."""

    @abstractmethod
    def encrypt(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def decrypt(self, data: bytes) -> bytes:
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        pass


class FernetEncryption(Encryption):
    """Encryption for the Fernet package.

    Adapted from: https://cryptography.io/en/latest/fernet/

    """

    def __init__(
        self,
        password: str,
        level: EncryptionLevel = "sample",
    ) -> None:
        super().__init__()
        if not _CRYPTOGRAPHY_AVAILABLE:
            raise ModuleNotFoundError(str(_CRYPTOGRAPHY_AVAILABLE))

        if level not in get_args(EncryptionLevel):
            raise ValueError("The provided `level` should be either `sample` or `chunk`")

        self.password = password
        self.key = self._derive_key(password)
        self.fernet = Fernet(self.key)
        self.level = level
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

    def state_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "password": self.password, "level": self.level}


class RSAEncryption(Encryption):
    """Encryption for the RSA package.

    Adapted from: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/rsa/

    """

    def __init__(
        self,
        private_key_path: Optional[str] = None,
        public_key_path: Optional[str] = None,
        password: Optional[str] = None,
        level: EncryptionLevel = "sample",
    ):
        if not _CRYPTOGRAPHY_AVAILABLE:
            raise ModuleNotFoundError(str(_CRYPTOGRAPHY_AVAILABLE))

        if level not in get_args(EncryptionLevel):
            raise ValueError("The provided `level` should be either `sample` or `chunk`")

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

        self.password = password
        self.level = level
        self.extension = "rsa"

    def _load_private_key(self, path: str) -> Any:  # TODO: Fix Any
        with open(path, "rb") as key_file:
            return serialization.load_pem_private_key(
                key_file.read(),
                password=self.password.encode() if self.password else None,
            )

    def _load_public_key(self, path: str) -> Any:  # TODO: Fix Any
        with open(path, "rb") as key_file:
            return serialization.load_pem_public_key(key_file.read())

    def _generate_keys(self) -> Tuple[Any, Any]:  # TODO: Fix Any
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

    def save_keys(self, private_key_path: str, public_key_path: str) -> None:
        state = self.state_dict()
        with open(private_key_path, "wb") as key_file:
            key_file.write(state["private_key"].encode("utf-8"))

        with open(public_key_path, "wb") as key_file:
            key_file.write(state["public_key"].encode("utf-8"))

    def state_dict(self) -> Dict[str, Union[str, None]]:
        encryption_algorithm = (
            serialization.BestAvailableEncryption(self.password.encode())
            if self.password
            else serialization.NoEncryption()
        )
        return {
            "private_key": self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=encryption_algorithm,
            ).decode("utf-8")
            if self.private_key
            else None,
            "public_key": self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("utf-8")
            if self.public_key
            else None,
            "password": self.password,
            "level": self.level,
            "extension": self.extension,
        }

    def __getstate__(self) -> Dict[str, Union[str, Any]]:
        return self.state_dict()

    def __setstate__(self, state: Dict[str, Union[str, Any]]) -> None:
        # Restore the state from the serialized data
        self.password = state["password"]
        self.level = state["level"]
        self.extension = state["extension"]

        if state["private_key"]:
            self.private_key = serialization.load_pem_private_key(
                state["private_key"].encode("utf-8"),
                password=self.password.encode() if self.password else None,
            )
        else:
            self.private_key = None

        if state["public_key"]:
            self.public_key = serialization.load_pem_public_key(
                state["public_key"].encode("utf-8"),
            )
        else:
            self.public_key = None
