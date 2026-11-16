import pytest

from ragkit.governance import ABRouter, CanaryRouter, PromptRegistry, ReleaseConfig, bucket, parse_semver

A = ReleaseConfig("stable", "p", "1.0.0", "small", "snap-1")
B = ReleaseConfig("canary", "p", "1.1.0", "large", "snap-2")


def test_semver_parse_and_reject():
    assert parse_semver("1.10.2") == (1, 10, 2)
    with pytest.raises(ValueError):
        parse_semver("1.0")


def test_registry_latest_is_semver_not_alphabetical():
    r = PromptRegistry()
    r.register("p", "1.9.0", "a")
    r.register("p", "1.10.0", "b")
    assert r.get("p").version == "1.10.0"


def test_registry_idempotent_and_immutable():
    r = PromptRegistry()
    v = r.register("p", "1.0.0", "text")
    assert r.register("p", "1.0.0", "text") is v
    with pytest.raises(ValueError, match="bump the version"):
        r.register("p", "1.0.0", "edited")


def test_hash_depends_on_content():
    r = PromptRegistry()
    assert r.register("p", "1.0.0", "a").content_hash != r.register("p", "1.0.1", "a").content_hash


def test_activate_and_rollback():
    r = PromptRegistry()
    r.register("p", "1.0.0", "a")
    r.register("p", "1.1.0", "b")
    r.activate("p", "1.0.0")
    r.activate("p", "1.1.0")
    assert r.active("p").version == "1.1.0"
    assert r.rollback("p").version == "1.0.0"
    with pytest.raises(ValueError):
        r.rollback("p")


def test_fingerprint_changes_with_any_pinned_field():
    assert A.fingerprint != ReleaseConfig("stable", "p", "1.0.0", "small", "snap-2").fingerprint
    assert A.fingerprint == ReleaseConfig("stable", "p", "1.0.0", "small", "snap-1").fingerprint


def test_bucket_deterministic_and_in_range():
    assert bucket("u1", "s") == bucket("u1", "s")
    assert all(0 <= bucket(f"u{i}", "s") < 1 for i in range(200))
    assert bucket("u1", "s") != bucket("u1", "other-salt")


def test_canary_share_is_close_to_percent_and_sticky():
    router = CanaryRouter(A, B, percent=10)
    users = [f"u{i}" for i in range(2000)]
    share = sum(router.route(u) is B for u in users) / len(users)
    assert 0.07 < share < 0.13
    assert all(router.route(u) is router.route(u) for u in users[:50])


def test_canary_ramp_only_adds_users_and_zero_is_rollback():
    router = CanaryRouter(A, B, percent=10)
    users = [f"u{i}" for i in range(1000)]
    at10 = {u for u in users if router.route(u) is B}
    router.set_percent(40)
    at40 = {u for u in users if router.route(u) is B}
    assert at10 < at40
    router.set_percent(0)
    assert all(router.route(u) is A for u in users)
    router.set_percent(100)
    assert all(router.route(u) is B for u in users)


def test_ab_router_three_way_weights():
    c = ReleaseConfig("c", "p", "2.0.0", "small", "s")
    router = ABRouter([(A, 1), (B, 1), (c, 2)], salt="x")
    counts = {}
    for i in range(4000):
        n = router.route(f"u{i}").name
        counts[n] = counts.get(n, 0) + 1
    assert 0.2 < counts["stable"] / 4000 < 0.3 and 0.4 < counts["c"] / 4000 < 0.6
    with pytest.raises(ValueError):
        ABRouter([(A, 0)])
