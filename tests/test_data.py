import pytest

from sql_analyst import data


def test_existing_database_is_not_redownloaded(tmp_path, monkeypatch):
    db = tmp_path / "chinook.sqlite"
    db.write_bytes(b"x")
    monkeypatch.setattr(data.urllib.request, "urlretrieve", lambda *a: pytest.fail("downloaded"))
    assert data.ensure_database(db) == db


def test_missing_chinook_is_downloaded_and_verified(tmp_path, monkeypatch, chinook_db):
    dest = tmp_path / "chinook.sqlite"
    monkeypatch.setattr(
        data.urllib.request,
        "urlretrieve",
        lambda url, out: out.write_bytes(chinook_db.read_bytes()),
    )
    assert data.ensure_database(dest).exists()


def test_checksum_mismatch_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data.urllib.request, "urlretrieve", lambda url, out: out.write_bytes(b"evil")
    )
    with pytest.raises(data.ChecksumError):
        data.download_chinook(tmp_path / "chinook.sqlite")
    assert not (tmp_path / "chinook.sqlite").exists()
    assert not (tmp_path / "chinook.part").exists()


def test_other_databases_are_never_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(data.urllib.request, "urlretrieve", lambda *a: pytest.fail("downloaded"))
    data.ensure_database(tmp_path / "custom.sqlite")
