import os
from unittest import mock

from litdata.utilities import broadcast as broadcast_module
from litdata.utilities.broadcast import broadcast_object, requests


@mock.patch.dict(
    os.environ, {"LIGHTNING_APP_EXTERNAL_URL": "http://", "LIGHTNING_APP_STATE_URL": "http://"}, clear=True
)
def test_broadcast(monkeypatch):
    session = mock.MagicMock()
    resp = requests.Response()
    resp.status_code = 200

    def fn(*args, **kwargs):
        nonlocal session
        return {"value": session.post._mock_call_args_list[0].kwargs["json"]["value"]}

    resp.json = fn
    session.post.return_value = resp
    monkeypatch.setattr(requests, "Session", mock.MagicMock(return_value=session))
    assert broadcast_object("key", "value") == "value"


@mock.patch.dict(
    os.environ, {"LIGHTNING_APP_EXTERNAL_URL": "http://", "LIGHTNING_APP_STATE_URL": "http://"}, clear=True
)
def test_broadcast_with_rank(monkeypatch):
    session = mock.MagicMock()
    resp = requests.Response()
    resp.status_code = 200

    counter = 0

    def fn(*args, **kwargs):
        nonlocal session
        nonlocal counter
        counter += 1

        if counter == 3:
            return {"value": session.post._mock_call_args_list[0].kwargs["json"]["value"]}
        return {"value": None}

    resp.json = fn
    session.post.return_value = resp
    monkeypatch.setattr(requests, "Session", mock.MagicMock(return_value=session))
    monkeypatch.setattr(broadcast_module, "sleep", mock.MagicMock())
    assert broadcast_object("key", "value", rank=1) == "value"
    assert counter == 3
