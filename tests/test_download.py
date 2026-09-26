import io
import zipfile

import pytest

from bearing_attention import download as dl


class FakeResponse(io.BytesIO):
    def __init__(self, data, status, drop_after=None):
        super().__init__(data)
        self.status = status
        self.drop_after = drop_after

    def read(self, size=-1):
        if self.drop_after is not None and self.tell() >= self.drop_after:
            raise ConnectionResetError("simulated drop")
        return super().read(size)


def test_download_resumes_after_dropped_connection(tmp_path, monkeypatch):
    payload = bytes(range(256)) * 8
    requests = []

    def fake_urlopen(request, timeout):
        byte_range = request.get_header("Range")
        requests.append(byte_range)
        if byte_range is None:
            return FakeResponse(payload, 200, drop_after=700)
        start = int(byte_range.removeprefix("bytes=").rstrip("-"))
        return FakeResponse(payload[start:], 206)

    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    monkeypatch.setattr(dl, "CHUNK_BYTES", 100)

    dest = dl.download("http://example.invalid/a.zip", tmp_path / "a.zip", expected_bytes=len(payload))

    assert dest.read_bytes() == payload
    assert requests == [None, "bytes=700-"]
    assert not (tmp_path / "a.zip.part").exists()


def test_download_skips_complete_file(tmp_path, monkeypatch):
    dest = tmp_path / "a.zip"
    dest.write_bytes(b"x" * 10)

    def no_network(*args, **kwargs):
        raise AssertionError("should not touch the network")

    monkeypatch.setattr(dl.urllib.request, "urlopen", no_network)
    assert dl.download("http://example.invalid/a.zip", dest, expected_bytes=10) == dest


def _zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_extract_member_then_nested_zips(tmp_path):
    full_test = _zip_bytes({"Full_Test_Set/Bearing1_3/acc_00001.csv": "0,0,0,0,0.1,0.2\n"})
    inner = _zip_bytes({
        "Learning_set/Bearing1_1/acc_00001.csv": "9,39,39,65438,0.552,-0.146\n",
        "Full_Test_Set.zip": full_test,
    })
    outer = tmp_path / "outer.zip"
    outer.write_bytes(_zip_bytes({"10. FEMTO Bearing/": b"", "10. FEMTO Bearing/Inner.zip": inner}))

    inner_path = dl.extract_member(outer, "Inner.zip", tmp_path / "archives")
    assert inner_path.read_bytes() == inner

    out = tmp_path / "raw"
    dl.extract_zip(inner_path, out)
    assert (out / "Learning_set/Bearing1_1/acc_00001.csv").exists()
    assert (out / "Full_Test_Set/Bearing1_3/acc_00001.csv").exists()
    assert not (out / "Full_Test_Set.zip").exists()


def test_extract_member_rejects_ambiguous_name(tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(_zip_bytes({"x/data.zip": b"1", "y/data.zip": b"2"}))
    with pytest.raises(RuntimeError, match="found 2"):
        dl.extract_member(archive, "data.zip", tmp_path)


@pytest.mark.skipif(dl.find_archive_tool() is None, reason="needs bsdtar or 7-Zip")
def test_archive_tool_extracts_selected_member(tmp_path):
    archive = tmp_path / "pack.zip"
    archive.write_bytes(_zip_bytes({"2nd_test.rar": b"a", "1st_test.rar": b"b", "Readme.pdf": b"c"}))
    tool = dl.find_archive_tool()

    assert sorted(dl.list_archive(tool, archive)) == ["1st_test.rar", "2nd_test.rar", "Readme.pdf"]
    dl.extract_archive(tool, archive, tmp_path / "out", ["2nd_test.rar"])
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["2nd_test.rar"]
