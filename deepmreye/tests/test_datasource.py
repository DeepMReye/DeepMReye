"""Tests for corpus resolution across machines.

The same commands run on the cluster (data on scratch, no network wanted) and
on a laptop (data pulled from HuggingFace). These pin down which location wins,
and that a download is never attempted when a local copy exists.
"""
import h5py
import pytest

from deepmreye import datasource


def _make_corpus(path):
    path.mkdir(parents=True, exist_ok=True)
    with h5py.File(path / datasource.REGISTRY_NAME, "w") as f:
        f.create_group("ds000001")
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DEEPMREYE_DATA", raising=False)
    monkeypatch.delenv("DEEPMREYE_CACHE", raising=False)


def test_explicit_path_wins_over_everything(tmp_path, monkeypatch):
    env = _make_corpus(tmp_path / "env")
    monkeypatch.setenv("DEEPMREYE_DATA", str(env))
    explicit = tmp_path / "explicit"

    assert datasource.resolve(explicit, quiet=True) == explicit


def test_explicit_path_is_created_not_downloaded(tmp_path, monkeypatch):
    """An explicit path states intent; it must not silently become a download."""
    monkeypatch.setattr(datasource, "fetch", lambda **kw: pytest.fail("downloaded"))
    target = tmp_path / "new"
    assert datasource.resolve(target, quiet=True) == target
    assert target.is_dir()


def test_env_var_used_when_set(tmp_path, monkeypatch):
    env = _make_corpus(tmp_path / "scratch")
    monkeypatch.setenv("DEEPMREYE_DATA", str(env))
    monkeypatch.setattr(datasource, "fetch", lambda **kw: pytest.fail("downloaded"))

    assert datasource.resolve(quiet=True) == env


def test_cwd_data_dir_used_when_it_is_a_corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = _make_corpus(tmp_path / "data")
    monkeypatch.setattr(datasource, "fetch", lambda **kw: pytest.fail("downloaded"))

    assert datasource.resolve(quiet=True) == local


def test_empty_data_dir_is_not_mistaken_for_a_corpus(tmp_path, monkeypatch):
    """A bare ./data with no registry must not shadow the real copy."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cache = _make_corpus(tmp_path / "cache")
    monkeypatch.setenv("DEEPMREYE_CACHE", str(cache))

    assert datasource.resolve(quiet=True) == cache


def test_cached_download_is_reused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = _make_corpus(tmp_path / "cache")
    monkeypatch.setenv("DEEPMREYE_CACHE", str(cache))
    monkeypatch.setattr(datasource, "fetch", lambda **kw: pytest.fail("re-downloaded"))

    assert datasource.resolve(quiet=True) == cache


def test_cache_is_topped_up_for_the_stage_that_needs_more(tmp_path, monkeypatch):
    """Labeling pulls only the registry, so `evaluate` must not find an empty cache."""
    monkeypatch.chdir(tmp_path)
    cache = _make_corpus(tmp_path / "cache")
    monkeypatch.setenv("DEEPMREYE_CACHE", str(cache))

    asked = {}
    monkeypatch.setattr(datasource, "fetch",
                        lambda **kw: asked.update(kw) or kw["target"])

    datasource.resolve(patterns=["*/*.h5"], quiet=True)
    assert asked["patterns"] == ["*/*.h5"]
    assert asked["target"] == cache


def test_a_local_corpus_is_never_topped_up(tmp_path, monkeypatch):
    """A directory you pointed at is yours; only the cache is ours to complete."""
    env = _make_corpus(tmp_path / "scratch")
    monkeypatch.setenv("DEEPMREYE_DATA", str(env))
    monkeypatch.setattr(datasource, "fetch", lambda **kw: pytest.fail("touched network"))

    assert datasource.resolve(patterns=["*/*.h5"], quiet=True) == env


def test_stage_patterns_keep_labeling_small(tmp_path):
    """The point of the table: `qa` must not drag down blocks or reports.

    Thumbnails are the exception and the reason this is not simply "no globs":
    at ~20 KB each they are ~30 MB over the whole QA sample, so labeling can
    take them in one go rather than streaming 5 MB reports per dataset.
    """
    qa = datasource.STAGE_PATTERNS["qa"]
    assert "datasets.h5" in qa
    assert not any(p.endswith(".h5") and "*" in p for p in qa)
    assert not any("html" in p for p in qa)
    assert datasource.THUMBNAIL_GLOB in qa


def test_evaluate_stage_takes_only_the_labeled_datasets():
    """`dsL*` is what makes evaluation cheap: labels without the corpus."""
    stage = datasource.STAGE_PATTERNS["evaluate"]
    assert datasource.LABELED_GLOB in stage
    assert datasource.LABELED_GLOB.startswith("dsL")
    # A bare `*/*.h5` here would pull the whole unlabeled corpus instead.
    assert "*/*.h5" not in stage
    assert not any("html" in p for p in stage)
    assert not any("png" in p for p in stage)


def test_no_download_raises_with_a_useful_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPMREYE_CACHE", str(tmp_path / "empty"))

    with pytest.raises(FileNotFoundError, match="DEEPMREYE_DATA"):
        datasource.resolve(download=False, quiet=True)


def test_download_is_the_last_resort(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setenv("DEEPMREYE_CACHE", str(cache))

    called = {}

    def fake_fetch(repo_id=None, target=None, patterns=None, quiet=False):
        called["target"] = target
        return _make_corpus(target)

    monkeypatch.setattr(datasource, "fetch", fake_fetch)
    assert datasource.resolve(quiet=True) == cache
    assert called["target"] == cache


def test_ensure_reports_skips_datasets_already_present(tmp_path, monkeypatch):
    """Reports are big; re-fetching ones already on disk would be costly."""
    have = tmp_path / "ds000001" / "sub-01"
    have.mkdir(parents=True)
    (have / "report_sub-01.html").write_text("<html></html>")

    requested = {}

    def fake_fetch(repo_id=None, target=None, patterns=None, quiet=False):
        requested["patterns"] = patterns

    monkeypatch.setattr(datasource, "fetch", fake_fetch)

    missing = datasource.ensure_reports(tmp_path, ["ds000001", "ds000002"], quiet=True)
    assert missing == ["ds000002"]
    assert requested["patterns"] == ["ds000002/*/*.html"]


def test_ensure_reports_is_a_noop_when_all_present(tmp_path, monkeypatch):
    have = tmp_path / "ds000001" / "sub-01"
    have.mkdir(parents=True)
    (have / "report_sub-01.html").write_text("<html></html>")
    monkeypatch.setattr(datasource, "fetch", lambda **kw: pytest.fail("fetched"))

    assert datasource.ensure_reports(tmp_path, ["ds000001"], quiet=True) == []
