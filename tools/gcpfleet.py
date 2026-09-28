"""Rent one GPU, run the benchmark, read the answer back, delete the instance.

The point of this file is the guardrails, not the API calls.  Money is spent by
a process that can be killed at any moment -- this container is reclaimed after
a period of inactivity -- so nothing here may depend on the process living long
enough to clean up.

Three defences, in order of how much they are trusted:

1. Compute Engine deletes the instance itself.  Every instance is created with
   ``scheduling.maxRunDuration`` and ``instanceTerminationAction: DELETE``, so
   the deletion is enforced by Google on a wall clock, not by this script.  If
   the session dies one second after the instance boots, the instance still
   disappears on schedule.
2. Only the machine types in :data:`MACHINES` can be asked for.  There is no
   code path that accepts an arbitrary machine type, so a typo cannot launch an
   eight-GPU node.
3. A ledger on disk accumulates the cost of every launch and refuses the next
   one if it would pass the budget.  Spot instances are charged at the
   *on-demand* rate in the ledger, so the running total is always an
   over-estimate of the real bill.

Authentication is a service-account JSON key, read from the environment and
never written to the repository.  The JWT is signed by shelling out to
``openssl``, so this file needs nothing outside the standard library.

Results come back through the serial console (``getSerialPortOutput``), which
is a plain API read.  No SSH key, no firewall rule, no bucket, no inbound
anything.

  python -m tools.gcpfleet --plan                 # costs nothing, needs no key
  python -m tools.gcpfleet --run l4 --yes         # cheap smoke test
  python -m tools.gcpfleet --run a100-80 --yes    # the real measurement
  python -m tools.gcpfleet --reap                 # delete anything still up
  python -m tools.gcpfleet --status               # what is running, what it cost
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = pathlib.Path(__file__).resolve().parent / "gcp_ledger.json"
LABEL = "tomogrid-bench"
API = "https://compute.googleapis.com/compute/v1"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"

#: Hard budget in US dollars.  Raise it deliberately, not by accident.
BUDGET_USD = float(os.environ.get("TOMOGRID_GPU_BUDGET_USD", "15"))

#: Metadata values are capped at 256 KiB; the payload is checked against this.
METADATA_LIMIT = 256 * 1024

BEGIN, END = "===TOMOGRID-BEGIN===", "===TOMOGRID-END==="


@dataclass(frozen=True)
class Machine:
    """One allowed configuration.

    ``usd_hour`` is the on-demand list price, used for budgeting even when the
    instance is a spot instance, so the ledger over-estimates.  Prices move and
    differ by region; they are here to bound spending, not to quote a bill.
    """

    key: str
    machine_type: str
    accelerator: str
    count: int
    usd_hour: float
    note: str = ""


#: The regional quota that has to be non-zero before any of this works.  A new
#: project has zero GPU quota, which is the usual reason a launch fails.
QUOTA = {"nvidia-tesla-t4": "NVIDIA_T4_GPUS", "nvidia-l4": "NVIDIA_L4_GPUS",
         "nvidia-tesla-a100": "NVIDIA_A100_GPUS",
         "nvidia-a100-80gb": "NVIDIA_A100_80GB_GPUS"}

MACHINES = {m.key: m for m in (
    Machine("t4", "n1-standard-8", "nvidia-tesla-t4", 1, 0.95,
            "oldest and cheapest; fp64 is 1/32, use for plumbing only"),
    Machine("l4", "g2-standard-8", "nvidia-l4", 1, 0.93,
            "300 GB/s, fp64 1/64: the cheap way to test the fp32 path"),
    Machine("a100-40", "a2-highgpu-1g", "nvidia-tesla-a100", 1, 3.75,
            "1555 GB/s, fp64 9.7 Tflop/s"),
    Machine("a100-80", "a2-ultragpu-1g", "nvidia-a100-80gb", 1, 5.10,
            "2039 GB/s; the device in the paper's table"),
)}


@dataclass
class Run:
    key: str
    zone: str
    name: str
    started: float
    minutes: float
    usd_hour: float

    @property
    def budgeted_usd(self) -> float:
        return self.usd_hour * self.minutes / 60.0


@dataclass
class Ledger:
    """What has been committed so far, as an over-estimate."""

    path: pathlib.Path = LEDGER
    runs: list[Run] = field(default_factory=list)

    def load(self) -> "Ledger":
        if self.path.exists():
            self.runs = [Run(**r) for r in json.loads(self.path.read_text())]
        return self

    def save(self) -> None:
        self.path.write_text(json.dumps([r.__dict__ for r in self.runs],
                                        indent=2) + "\n")

    def committed(self) -> float:
        return sum(r.budgeted_usd for r in self.runs)

    def check(self, machine: Machine, minutes: float,
              budget: float = BUDGET_USD) -> float:
        """Cost of the proposed run, or raise if it would break the budget."""
        cost = machine.usd_hour * minutes / 60.0
        total = self.committed() + cost
        if total > budget:
            raise BudgetExceeded(
                f"{machine.key} for {minutes:g} min budgets ${cost:.2f}; "
                f"${self.committed():.2f} already committed, "
                f"${total:.2f} > ${budget:.2f} cap")
        return cost


class BudgetExceeded(RuntimeError):
    pass


# --- credentials -----------------------------------------------------------


def _service_account() -> dict:
    raw = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not raw:
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if path and pathlib.Path(path).exists():
            raw = pathlib.Path(path).read_text()
    if not raw:
        raise RuntimeError(
            "no credential: set GCP_SERVICE_ACCOUNT_JSON to the service-account "
            "key JSON, or GOOGLE_APPLICATION_CREDENTIALS to a path")
    return json.loads(raw)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def assertion(info: dict, now: int | None = None) -> str:
    """The signed JWT a service account trades for a token.

    The private key is written to a file only for as long as openssl needs it,
    with owner-only permissions, and removed in a ``finally``.  It is never
    logged and never returned.
    """
    now = int(time.time()) if now is None else now
    header = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    claims = _b64(json.dumps({
        "iss": info["client_email"], "scope": SCOPE,
        "aud": "https://oauth2.googleapis.com/token",
        "iat": now, "exp": now + 3600,
    }).encode())
    signing_input = f"{header}.{claims}".encode()

    key_path = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f".sa{now}.pem"
    try:
        key_path.touch(mode=0o600)
        key_path.write_text(info["private_key"])
        sig = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", str(key_path)],
            input=signing_input, capture_output=True, check=True).stdout
    finally:
        key_path.unlink(missing_ok=True)

    return f"{header}.{claims}.{_b64(sig)}"


def access_token(info: dict | None = None) -> str:
    """Mint an OAuth access token from a service-account key."""
    body = urllib.parse.urlencode({
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": assertion(info or _service_account()),
    }).encode()
    req = urllib.request.Request("https://oauth2.googleapis.com/token",
                                 data=body)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["access_token"]


def _http(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # surface Google's message, not a 500
        raise RuntimeError(f"{method} {url} -> {e.code}: "
                           f"{e.read().decode()[:600]}") from None


# --- the payload the instance runs -----------------------------------------


def payload(paths=("tomogrid", "experiments", "pyproject.toml")) -> str:
    """The package, as a base64 tar, small enough to ride in the metadata."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in paths:
            src = ROOT / rel
            if not src.exists():
                continue
            tar.add(src, arcname=rel, filter=lambda t: (
                None if "__pycache__" in t.name or t.name.endswith(".csv")
                else t))
    return base64.b64encode(buf.getvalue()).decode()


def startup_script(grids: str, sigma: int, order: int, minutes: float,
                   tarball: str) -> str:
    """Run the benchmark, print it between markers, then power off.

    The ``shutdown`` at the top is a second timer inside the guest, in case the
    benchmark hangs: it fires before Compute Engine's own deadline, so the
    normal path is a clean shutdown and the API deadline is the backstop.
    """
    inner = int(max(5, minutes - 3))
    return f"""#!/bin/bash
shutdown -h +{inner} &
set -x
mkdir -p /opt/tomogrid && cd /opt/tomogrid
echo '{tarball}' | base64 -d | tar xz
pip install --quiet numpy 'cupy-cuda12x' || pip install --quiet numpy cupy-cuda11x
echo "{BEGIN}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import cupy; print('cupy', cupy.__version__)"
PYTHONPATH=/opt/tomogrid python3 -m experiments.gpu_bench \
    --grids {grids} --sigma {sigma} --order {order} --backend cupy 2>&1
PYTHONPATH=/opt/tomogrid python3 -m experiments.gpu_bench \
    --grids {grids} --sigma {sigma} --order {order} --backend cupy --fp32 2>&1
PYTHONPATH=/opt/tomogrid python3 -m experiments.gpu_bench \
    --grids {grids} --sigma {sigma} --order {order} --backend numpy 2>&1
echo "{END}"
poweroff
"""


def instance_body(machine: Machine, name: str, zone: str, image: str,
                  script: str, minutes: float, spot: bool = True) -> dict:
    """The create request.  The guardrails live here and are asserted in tests."""
    seconds = int(minutes * 60)
    return {
        "name": name,
        "machineType": f"zones/{zone}/machineTypes/{machine.machine_type}",
        "labels": {LABEL: "1"},
        "guestAccelerators": [{
            "acceleratorType": f"zones/{zone}/acceleratorTypes/"
                               f"{machine.accelerator}",
            "acceleratorCount": machine.count,
        }],
        "disks": [{
            "boot": True, "autoDelete": True,
            "initializeParams": {"sourceImage": image, "diskSizeGb": 60},
        }],
        "networkInterfaces": [{
            "network": "global/networks/default",
            # A public address only so the instance can reach pypi; nothing
            # listens on it, and no firewall rule is created.
            "accessConfigs": [{"type": "ONE_TO_ONE_NAT", "name": "external"}],
        }],
        "scheduling": {
            "onHostMaintenance": "TERMINATE",
            "automaticRestart": False,
            "provisioningModel": "SPOT" if spot else "STANDARD",
            "instanceTerminationAction": "DELETE",
            "maxRunDuration": {"seconds": seconds},
        },
        "metadata": {"items": [
            {"key": "startup-script", "value": script},
            {"key": "serial-port-logging-enable", "value": "true"},
        ]},
    }


# --- driving it ------------------------------------------------------------


IMAGE_FAMILIES = (
    ("deeplearning-platform-release", "common-cu124-debian-11"),
    ("deeplearning-platform-release", "common-cu123-debian-11"),
    ("deeplearning-platform-release", "common-cu121-debian-11"),
    ("deeplearning-platform-release", "common-cu118-debian-11"),
)


class Fleet:
    def __init__(self, project: str, zone: str, token: str | None = None,
                 http=_http):
        self.project, self.zone, self.http = project, zone, http
        self._token = token

    @property
    def token(self) -> str:
        if self._token is None:
            self._token = access_token()
        return self._token

    def _url(self, path: str) -> str:
        return f"{API}/projects/{self.project}{path}"

    def image(self) -> str:
        """First CUDA image family that resolves, so a rename cannot strand us."""
        for proj, fam in IMAGE_FAMILIES:
            try:
                r = self.http("GET", f"{API}/projects/{proj}/global/images/"
                                     f"family/{fam}", self.token)
                return r["selfLink"]
            except RuntimeError:
                continue
        raise RuntimeError("no CUDA image family resolved; pass --image")

    def create(self, body: dict) -> dict:
        return self.http("POST", self._url(f"/zones/{self.zone}/instances"),
                         self.token, body)

    def delete(self, name: str) -> dict:
        return self.http("DELETE",
                         self._url(f"/zones/{self.zone}/instances/{name}"),
                         self.token)

    def ours(self) -> list[dict]:
        q = urllib.parse.urlencode({"filter": f"labels.{LABEL}=1"})
        r = self.http("GET", self._url(f"/zones/{self.zone}/instances?{q}"),
                      self.token)
        return r.get("items", [])

    def quota(self, machine: Machine) -> tuple[float, float]:
        """``(usage, limit)`` for this accelerator, spot or not, in the region.

        Worth asking before spending: a project that has never requested GPU
        quota has a limit of zero, and the create call fails after the ledger
        has already been written.
        """
        region = self.zone.rsplit("-", 1)[0]
        r = self.http("GET", self._url(f"/regions/{region}"), self.token)
        names = {QUOTA[machine.accelerator],
                 "PREEMPTIBLE_" + QUOTA[machine.accelerator]}
        rows = [q for q in r.get("quotas", []) if q["metric"] in names]
        if not rows:
            return (0.0, 0.0)
        best = max(rows, key=lambda q: q["limit"])
        return (best["usage"], best["limit"])

    def serial(self, name: str, start: int = 0) -> dict:
        return self.http("GET", self._url(
            f"/zones/{self.zone}/instances/{name}/serialPort"
            f"?port=1&start={start}"), self.token)

    def reap(self) -> list[str]:
        """Delete every instance we labelled.  Safe to call at any time."""
        gone = []
        for inst in self.ours():
            self.delete(inst["name"])
            gone.append(inst["name"])
        return gone


def harvest(fleet: Fleet, name: str, deadline: float, poll: float = 20.0):
    """Follow the serial console until the end marker, or the deadline."""
    start, seen = 0, ""
    while time.time() < deadline:
        try:
            r = fleet.serial(name, start)
        except RuntimeError as e:
            if "404" in str(e):  # already gone
                break
            raise
        chunk = r.get("contents", "")
        if chunk:
            print(chunk, end="", flush=True)
            seen += chunk
            start = int(r.get("next", start))
        if END in seen:
            break
        time.sleep(poll)
    return seen


def plan(keys, minutes, budget=BUDGET_USD) -> str:
    led = Ledger().load()
    out = [f"budget ${budget:.2f}, ${led.committed():.2f} already committed",
           f"{'run':>9} {'machine':>16} {'GPU':>18} {'min':>5} "
           f"{'budgeted':>9}  note"]
    total = 0.0
    for k in keys:
        m = MACHINES[k]
        cost = m.usd_hour * minutes / 60.0
        total += cost
        out.append(f"{k:>9} {m.machine_type:>16} {m.accelerator:>18} "
                   f"{minutes:5g} {'$%.2f' % cost:>9}  {m.note}")
    out.append(f"{'':>9} {'':>16} {'':>18} {'':>5} {'$%.2f' % total:>9}  "
               f"at on-demand rates; spot is typically a third of this")
    out.append("every instance carries maxRunDuration="
               f"{int(minutes * 60)}s with terminationAction=DELETE, so "
               "Compute Engine removes it even if this session dies")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true",
                    help="print the cost and the guardrails; no network")
    ap.add_argument("--run", choices=sorted(MACHINES))
    ap.add_argument("--reap", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--yes", action="store_true", help="required to spend")
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--grids", default="129 257")
    ap.add_argument("--sigma", type=int, default=2)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--zone", default=os.environ.get("GCP_ZONE",
                                                     "us-central1-a"))
    ap.add_argument("--project", default=os.environ.get("GCP_PROJECT"))
    ap.add_argument("--image")
    ap.add_argument("--on-demand", action="store_true",
                    help="do not use a spot instance (about 3x the price)")
    args = ap.parse_args()

    if args.plan or not (args.run or args.reap or args.status):
        print(plan(sorted(MACHINES), args.minutes))
        print(f"\npayload {len(payload()) / 1024:.0f} KiB of "
              f"{METADATA_LIMIT // 1024} KiB metadata limit")
        return 0

    if not args.project:
        print("set GCP_PROJECT (or pass --project)")
        return 2
    fleet = Fleet(args.project, args.zone)

    if args.reap:
        gone = fleet.reap()
        print("deleted:", ", ".join(gone) if gone else "nothing was running")
        return 0

    if args.status:
        for inst in fleet.ours():
            print(inst["name"], inst["status"], inst.get("creationTimestamp"))
        led = Ledger().load()
        print(f"{len(led.runs)} runs, ${led.committed():.2f} committed of "
              f"${BUDGET_USD:.2f}")
        return 0

    machine = MACHINES[args.run]
    led = Ledger().load()
    cost = led.check(machine, args.minutes)
    print(plan([args.run], args.minutes))
    usage, limit = fleet.quota(machine)
    print(f"quota {QUOTA[machine.accelerator]} in "
          f"{args.zone.rsplit('-', 1)[0]}: {usage:g} used of {limit:g}")
    if limit < machine.count:
        print("no quota for this accelerator; request an increase under "
              "IAM & Admin > Quotas before spending anything")
        return 3
    if not args.yes:
        print("\nadd --yes to actually spend this")
        return 0

    name = f"tomogrid-{args.run}-{int(time.time())}"
    body = instance_body(machine, name, args.zone,
                         args.image or fleet.image(),
                         startup_script(args.grids, args.sigma, args.order,
                                        args.minutes, payload()),
                         args.minutes, spot=not args.on_demand)
    record = Run(args.run, args.zone, name, time.time(), args.minutes,
                 machine.usd_hour)
    led.runs.append(record)
    led.save()  # record before creating, so a crash still shows the spend
    print(f"creating {name} (${cost:.2f} budgeted)")
    try:
        fleet.create(body)
    except RuntimeError:
        record.minutes = 0.0  # nothing was created, so nothing was spent
        led.save()
        raise
    try:
        harvest(fleet, name, time.time() + args.minutes * 60)
    finally:
        print("\ndeleting", name)
        try:
            fleet.delete(name)
        except RuntimeError as e:
            print("delete failed, Compute Engine will still remove it:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
