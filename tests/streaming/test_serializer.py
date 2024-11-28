# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import random
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest
import tifffile
import torch
from lightning_utilities.core.imports import RequirementCache
from litdata.streaming.serializers import (
    _AV_AVAILABLE,
    _NUMPY_DTYPES_MAPPING,
    _SERIALIZERS,
    _TORCH_DTYPES_MAPPING,
    _TORCH_VISION_AVAILABLE,
    IntegerSerializer,
    JPEGSerializer,
    NoHeaderNumpySerializer,
    NoHeaderTensorSerializer,
    NumpySerializer,
    PILSerializer,
    TensorSerializer,
    TIFFSerializer,
    VideoSerializer,
    _get_serializers,
)


def seed_everything(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


_PIL_AVAILABLE = RequirementCache("PIL")
_TIFFFILE_AVAILABLE = RequirementCache("tifffile")


def test_serializers():
    keys = list(_SERIALIZERS.keys())
    assert keys == [
        "str",
        "int",
        "float",
        "video",
        "tifffile",
        "file",
        "pil",
        "jpeg",
        "bytes",
        "no_header_numpy",
        "numpy",
        "no_header_tensor",
        "tensor",
        "pickle",
    ]


def test_int_serializer():
    serializer = IntegerSerializer()

    for i in range(100):
        data, _ = serializer.serialize(i)
        assert isinstance(data, bytes)
        assert i == serializer.deserialize(data)


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
@pytest.mark.parametrize("mode", ["I", "L", "RGB"])
def test_pil_serializer(mode):
    serializer = PILSerializer()

    from PIL import Image

    np_data = np.random.randint(255, size=(28, 28), dtype=np.uint32)
    img = Image.fromarray(np_data).convert(mode)

    data, _ = serializer.serialize(img)
    assert isinstance(data, bytes)

    deserialized_img = serializer.deserialize(data)
    deserialized_img = deserialized_img.convert("I")
    np_dec_data = np.asarray(deserialized_img, dtype=np.uint32)
    assert isinstance(deserialized_img, Image.Image)

    # Validate data content
    assert np.array_equal(np_data, np_dec_data)


def test_pil_serializer_available():
    serializer = PILSerializer()
    with mock.patch("litdata.streaming.serializers._PIL_AVAILABLE", False):
        assert not serializer.can_serialize(None)


@pytest.mark.skipif(condition=not _PIL_AVAILABLE, reason="Requires: ['pil']")
def test_jpeg_serializer():
    serializer = JPEGSerializer()

    from PIL import Image

    array = np.random.randint(255, size=(28, 28, 3), dtype=np.uint8)
    img = Image.fromarray(array)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    img = Image.open(io.BytesIO(img_bytes))

    data, _ = serializer.serialize(img)
    assert isinstance(data, bytes)

    deserialized_img = serializer.deserialize(data)
    assert deserialized_img.shape == torch.Size([3, 28, 28])


def test_jpeg_serializer_available():
    serializer = JPEGSerializer()
    with mock.patch("litdata.streaming.serializers._PIL_AVAILABLE", False):
        assert not serializer.can_serialize(None)


@pytest.mark.flaky(reruns=3)
@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_tensor_serializer():
    seed_everything(42)

    serializer_tensor = TensorSerializer()

    shapes = [(10,), (10, 10), (10, 10, 10), (10, 10, 10, 5), (10, 10, 10, 5, 4)]
    for dtype in _TORCH_DTYPES_MAPPING.values():
        for shape in shapes:
            # Not serializable for some reasons
            if dtype in [torch.bfloat16]:
                continue
            tensor = torch.ones(shape, dtype=dtype)

            data, _ = serializer_tensor.serialize(tensor)
            deserialized_tensor = serializer_tensor.deserialize(data)

            assert deserialized_tensor.dtype == dtype
            assert torch.equal(tensor, deserialized_tensor)


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on windows")
def test_numpy_serializer():
    seed_everything(42)

    serializer_tensor = NumpySerializer()

    shapes = [(10,), (10, 10), (10, 10, 10), (10, 10, 10, 5), (10, 10, 10, 5, 4)]
    for dtype in _NUMPY_DTYPES_MAPPING.values():
        # Those types aren't supported
        if dtype.name in ["object", "bytes", "str", "void"]:
            continue
        for shape in shapes:
            tensor = np.ones(shape, dtype=dtype)
            data, _ = serializer_tensor.serialize(tensor)
            deserialized_tensor = serializer_tensor.deserialize(data)
            assert deserialized_tensor.dtype == dtype
            np.testing.assert_equal(tensor, deserialized_tensor)


def test_assert_bfloat16_tensor_serializer():
    serializer = TensorSerializer()
    tensor = torch.ones((10,), dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="Got unsupported ScalarType BFloat16"):
        serializer.serialize(tensor)


def test_assert_no_header_tensor_serializer():
    serializer = NoHeaderTensorSerializer()
    t = torch.ones((10,))
    data, name = serializer.serialize(t)
    assert name == "no_header_tensor:1"
    assert serializer._dtype is None
    serializer.setup(name)
    assert serializer._dtype == torch.float32
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)


def test_assert_no_header_numpy_serializer():
    serializer = NoHeaderNumpySerializer()
    t = np.ones((10,), dtype=np.float64)
    assert serializer.can_serialize(t)
    data, name = serializer.serialize(t)
    try:
        assert name == "no_header_numpy:10"
    except AssertionError as e:  # debug what np.core.sctypes looks like on Windows
        raise ValueError(np.core.sctypes) from e
    assert serializer._dtype is None
    serializer.setup(name)
    assert serializer._dtype == np.dtype("float64")
    new_t = serializer.deserialize(data)
    np.testing.assert_equal(t, new_t)


@pytest.mark.skipif(
    condition=not _TORCH_VISION_AVAILABLE or not _AV_AVAILABLE, reason="Requires: ['torchvision', 'av']"
)
def test_wav_deserialization(tmpdir):
    from torch.hub import download_url_to_file

    video_file = os.path.join(tmpdir, "video.mp4")
    key = "tutorial-assets/mptestsrc.mp4"  # E501
    download_url_to_file(f"https://download.pytorch.org/torchaudio/{key}", video_file)

    serializer = VideoSerializer()
    assert serializer.can_serialize(video_file)
    data, name = serializer.serialize(video_file)
    assert len(data) / 1024 / 1024 == 0.2262248992919922
    assert name == "video:mp4"
    vframes, aframes, info = serializer.deserialize(data)
    assert vframes.shape == torch.Size([301, 512, 512, 3])
    assert aframes.shape == torch.Size([1, 0])
    assert info == {"video_fps": 25.0}


def test_get_serializers():
    class CustomSerializer(NoHeaderTensorSerializer):
        pass

    serializers = _get_serializers({"no_header_tensor": CustomSerializer(), "custom": CustomSerializer()})

    assert isinstance(serializers["no_header_tensor"], CustomSerializer)
    assert isinstance(serializers["custom"], CustomSerializer)


def test_deserialize_empty_tensor():
    serializer = TensorSerializer()
    t = torch.ones((0, 3)).int()
    data, _ = serializer.serialize(t)
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)

    t = torch.ones((0, 3)).float()
    data, _ = serializer.serialize(t)
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)


def test_deserialize_empty_no_header_tensor():
    serializer = NoHeaderTensorSerializer()
    t = torch.ones((0,)).int()
    data, name = serializer.serialize(t)
    serializer.setup(name)
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)

    t = torch.ones((0,)).float()
    data, name = serializer.serialize(t)
    serializer.setup(name)
    new_t = serializer.deserialize(data)
    assert torch.equal(t, new_t)


@pytest.mark.skipif(not _TIFFFILE_AVAILABLE, reason="Requires: ['tifffile']")
def test_tiff_serializer():
    serializer = TIFFSerializer()

    # Create a synthetic multispectral image
    height, width, bands = 28, 28, 12
    np_data = np.random.randint(0, 65535, size=(height, width, bands), dtype=np.uint16)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        tifffile.imwrite(tmp_file.name, np_data)
        file_path = tmp_file.name

    # Test can_serialize
    assert serializer.can_serialize(file_path)

    # Serialize
    data, metadata = serializer.serialize(file_path)
    assert isinstance(data, bytes)
    assert metadata == f"tiff:{os.path.splitext(file_path)[1].lower()}"

    # Deserialize
    serializer.setup(metadata)
    deserialized_data = serializer.deserialize(data)
    assert isinstance(deserialized_data, np.ndarray)
    assert deserialized_data.shape == (height, width, bands)
    assert deserialized_data.dtype == np.uint16

    # Validate data content
    assert np.array_equal(np_data, deserialized_data)

    # Clean up
    os.remove(file_path)


def test_tiff_serializer_can_serialize():
    serializer = TIFFSerializer()
    assert serializer.can_serialize("image.tif")
    assert serializer.can_serialize("image.tiff")
    assert not serializer.can_serialize("image.jpg")
    assert not serializer.can_serialize(None)
    assert not serializer.can_serialize(123)
