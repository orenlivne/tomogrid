"""The guardrails on the GPU fleet driver.

None of these tests touches the network or spends anything.  What they check
is the part that decides whether money can be lost: that an instance cannot be
created without a deletion deadline Compute Engine enforces, that only the
listed machine types can be asked for, that the ledger refuses to exceed the
budget, and that the private key does not outlive the signature.
"""

from __future__ import annotations

import base64
import json
import pathlib
import subprocess

import pytest

from tools import gcpfleet as gf


def _ledger(tmp_path):
    return gf.Ledger(path=tmp_path / "ledger.json")


# --- the budget ------------------------------------------------------------


def test_budget_refuses_the_run_that_would_break_it(tmp_path):
    led = _ledger(tmp_path)
    led.check(gf.MACHINES["a100-80"], 60, budget=15.0)  # $5.10, fine
    with pytest.raises(gf.BudgetExceeded):
        led.check(gf.MACHINES["a100-80"], 60 * 4, budget=15.0)


def test_budget_counts_what_has_already_been_committed(tmp_path):
    led = _ledger(tmp_path)
    for _ in range(2):
        m = gf.MACHINES["a100-80"]
        led.check(m, 60, budget=15.0)
        led.runs.append(gf.Run("a100-80", "z", "n", 0.0, 60, m.usd_hour))
    assert led.committed() == pytest.approx(10.20)
    with pytest.raises(gf.BudgetExceeded):
        led.check(gf.MACHINES["a100-80"], 60, budget=15.0)


def test_the_ledger_survives_the_process(tmp_path):
    led = _ledger(tmp_path)
    led.runs.append(gf.Run("l4", "z", "n", 0.0, 25, 0.93))
    led.save()
    assert gf.Ledger(path=led.path).load().committed() == pytest.approx(0.3875)


def test_spot_is_budgeted_at_the_on_demand_rate():
    """The ledger must over-estimate, never under-estimate."""
    body = gf.instance_body(gf.MACHINES["a100-80"], "n", "z", "img", "#!/bin/sh",
                            25, spot=True)
    assert body["scheduling"]["provisioningModel"] == "SPOT"
    assert gf.MACHINES["a100-80"].usd_hour > 4.0  # the on-demand figure


# --- the instance cannot outlive its deadline -------------------------------


@pytest.mark.parametrize("key", sorted(gf.MACHINES))
@pytest.mark.parametrize("spot", [True, False])
def test_every_instance_deletes_itself(key, spot):
    body = gf.instance_body(gf.MACHINES[key], "n", "us-central1-a", "img",
                            "#!/bin/sh", 25, spot=spot)
    sched = body["scheduling"]
    assert sched["instanceTerminationAction"] == "DELETE"
    assert sched["maxRunDuration"] == {"seconds": 1500}
    assert sched["automaticRestart"] is False
    assert body["labels"] == {gf.LABEL: "1"}
    assert body["disks"][0]["autoDelete"] is True


def test_the_guest_timer_fires_before_the_api_deadline():
    """A hung benchmark should shut down cleanly, not be killed."""
    script = gf.startup_script("129", 2, 4, 25, "")
    assert "shutdown -h +22" in script
    assert script.rstrip().endswith("poweroff")


def test_no_inbound_anything():
    body = gf.instance_body(gf.MACHINES["l4"], "n", "z", "img", "#!/bin/sh", 25)
    assert "firewall" not in json.dumps(body).lower()
    assert "sshKeys" not in json.dumps(body)


# --- only the listed machines ----------------------------------------------

def test_machine_keys_are_the_only_way_in():
    assert set(gf.MACHINES) == {"t4", "l4", "a100-40", "a100-80"}
    for m in gf.MACHINES.values():
        assert m.count == 1, "multi-GPU nodes are out of budget by design"
        assert m.usd_hour < 6.0


def test_plan_needs_no_credential_and_no_network(tmp_path, monkeypatch):
    monkeypatch.delenv("GCP_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setattr(gf, "LEDGER", tmp_path / "l.json")
    text = gf.plan(["l4", "a100-80"], 25)
    assert "$0.39" in text and "maxRunDuration" in text


# --- the payload ------------------------------------------------------------


def test_payload_fits_the_metadata_limit():
    script = gf.startup_script("129 257 513", 2, 4, 25, gf.payload())
    assert len(script.encode()) < gf.METADATA_LIMIT


def test_payload_carries_the_benchmark_and_the_kernels():
    import io
    import tarfile

    tar = tarfile.open(fileobj=io.BytesIO(base64.b64decode(gf.payload())))
    names = tar.getnames()
    assert "tomogrid/bd.py" in names
    assert "tomogrid/backend.py" in names
    assert "experiments/gpu_bench.py" in names
    assert not [n for n in names if "__pycache__" in n]


# --- talking to the API -----------------------------------------------------


class FakeHttp:
    def __init__(self, listing):
        self.listing, self.calls = listing, []

    def __call__(self, method, url, token, body=None):
        self.calls.append((method, url))
        if method == "GET" and "instances?" in url:
            return {"items": self.listing}
        return {"status": "DONE"}


def test_reap_deletes_only_our_instances():
    http = FakeHttp([{"name": "tomogrid-l4-1"}, {"name": "tomogrid-l4-2"}])
    fleet = gf.Fleet("proj", "zone", token="t", http=http)
    assert fleet.reap() == ["tomogrid-l4-1", "tomogrid-l4-2"]
    import urllib.parse
    assert all(f"labels.{gf.LABEL}=1" in urllib.parse.unquote(u)
               for m, u in http.calls if m == "GET")
    assert sum(1 for m, _ in http.calls if m == "DELETE") == 2


def test_reap_is_a_noop_when_nothing_is_up():
    fleet = gf.Fleet("proj", "zone", token="t", http=FakeHttp([]))
    assert fleet.reap() == []


def test_harvest_stops_at_the_end_marker(capsys):
    chunks = ["boot\n", f"{gf.BEGIN}\nn=129 ok\n{gf.END}\n"]

    def http(method, url, token, body=None):
        return {"contents": chunks.pop(0) if chunks else "", "next": "0"}

    fleet = gf.Fleet("p", "z", token="t", http=http)
    seen = gf.harvest(fleet, "n", deadline=9e18, poll=0.0)
    assert gf.END in seen and "n=129 ok" in seen
    assert chunks == []


def test_harvest_gives_up_when_the_instance_is_gone():
    def http(method, url, token, body=None):
        raise RuntimeError("GET ... -> 404: not found")

    fleet = gf.Fleet("p", "z", token="t", http=http)
    assert gf.harvest(fleet, "n", deadline=9e18, poll=0.0) == ""


# --- the key ----------------------------------------------------------------


def test_signing_works_and_leaves_no_key_behind(tmp_path, monkeypatch):
    pem = tmp_path / "k.pem"
    subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-out",
                    str(pem), "-pkeyopt", "rsa_keygen_bits:2048"],
                   check=True, capture_output=True)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    before = set(tmp_path.iterdir())

    tok = gf.assertion({"client_email": "a@b.iam.gserviceaccount.com",
                        "private_key": pem.read_text()}, now=1700000000)

    assert set(tmp_path.iterdir()) == before, "the key file was left on disk"
    head, claims, sig = tok.split(".")
    payload = json.loads(base64.urlsafe_b64decode(claims + "=="))
    assert payload["scope"] == gf.SCOPE
    assert payload["exp"] - payload["iat"] == 3600
    assert len(base64.urlsafe_b64decode(sig + "==")) == 256  # RS256


def test_missing_credential_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GCP_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with pytest.raises(RuntimeError, match="GCP_SERVICE_ACCOUNT_JSON"):
        gf._service_account()


def test_quota_prefers_the_larger_of_the_spot_and_standard_limits():
    def http(method, url, token, body=None):
        assert "/regions/us-central1" in url
        return {"quotas": [
            {"metric": "NVIDIA_A100_80GB_GPUS", "usage": 0, "limit": 0},
            {"metric": "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS", "usage": 1,
             "limit": 8},
            {"metric": "CPUS", "usage": 3, "limit": 100},
        ]}

    fleet = gf.Fleet("p", "us-central1-a", token="t", http=http)
    assert fleet.quota(gf.MACHINES["a100-80"]) == (1, 8)


def test_quota_of_a_project_that_never_asked_is_zero():
    fleet = gf.Fleet("p", "us-central1-a", token="t",
                     http=lambda *a, **k: {"quotas": []})
    assert fleet.quota(gf.MACHINES["l4"]) == (0.0, 0.0)


def test_every_machine_has_a_quota_metric():
    assert all(m.accelerator in gf.QUOTA for m in gf.MACHINES.values())
