# Renting a GPU to measure this

I can do it. The network policy in this container already allows the Compute
Engine and OAuth endpoints — they answer genuine `401` and `400`, so the only
thing missing is a credential. `tools/gcpfleet.py` is the driver; it talks to
the REST API directly and needs nothing but the Python standard library and
`openssl`, so no `gcloud` install is involved (the `gcloud` apt repo *is*
blocked here, which is why it doesn't use it).

## What I need from you

Five things. None of them involves pasting a secret into the chat.

1. **A dedicated GCP project.** Not your main one. A throwaway project with
   its own billing budget means the blast radius of anything I do — or of a
   leaked key — is one project you can delete.
2. **Billing enabled** on it, and the **Compute Engine API** turned on.
3. **GPU quota.** This is the real blocker, not the money. A new project has a
   limit of *zero* GPUs and the launch will just fail. Under IAM & Admin →
   Quotas, request `PREEMPTIBLE_NVIDIA_L4_GPUS` ≥ 1 and
   `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` ≥ 1 in `us-central1`. L4 is usually
   granted within minutes; A100 can take a day and is sometimes refused on a
   new account. The driver checks quota and tells you before it spends
   anything.
4. **A service account with two roles and nothing else** — it must not be able
   to touch billing, IAM, or storage:

   ```
   gcloud iam service-accounts create tomogrid-bench --project "$P"
   gcloud projects add-iam-policy-binding "$P" \
     --member "serviceAccount:tomogrid-bench@$P.iam.gserviceaccount.com" \
     --role roles/compute.instanceAdmin.v1
   gcloud projects add-iam-policy-binding "$P" \
     --member "serviceAccount:tomogrid-bench@$P.iam.gserviceaccount.com" \
     --role roles/compute.viewer
   gcloud iam service-accounts keys create key.json \
     --iam-account "tomogrid-bench@$P.iam.gserviceaccount.com"
   ```

5. **Give me the key through the environment, not the chat.** In the cloud
   environment menu in this session's title bar → Edit, add environment
   variables: `GCP_SERVICE_ACCOUNT_JSON` set to the whole contents of
   `key.json`, and `GCP_PROJECT` set to the project id. Optionally `GCP_ZONE`
   (default `us-central1-a`) and `TOMOGRID_GPU_BUDGET_USD` (default `15`). A
   new session picks them up.

Also worth doing, though it does not stop spending on its own: a billing
budget on the project with alerts at $5 and $10.

## What it would cost

The compute is minutes. Almost all of the money is boot and install time,
which is why the plan uses images that already have CUDA on them.

| step | machine | wall | budgeted | likely actual (spot) |
|---|---|---|---|---|
| plumbing: does CuPy run, does it agree with the CPU | 1×L4 | 25 min | $0.39 | ~$0.13 |
| the measurement | 1×A100 80GB | 25 min | $2.12 | ~$0.71 |
| **total** | | | **$2.51** | **~$0.85** |

"Budgeted" is the on-demand list rate, which is what the ledger charges even
for a spot instance, so the running total always over-states the bill. One
retry of each still lands near $5 of the $15 cap.

H100 is not in the plan. On GCP it comes in eight-GPU nodes at roughly
$88/hour, so even twenty minutes is twice the whole budget. That row of the
paper's table stays a roofline unless you want to spend differently.

## What stops this costing more than that

Three things, in the order I trust them.

1. **Compute Engine deletes the instance, not me.** Every instance is created
   with `scheduling.maxRunDuration` and `instanceTerminationAction: DELETE`.
   This matters because *I am not a daemon* — this container is reclaimed
   after a period of inactivity, and my session can end mid-run. If that
   happens one second after the instance boots, Google still deletes it on
   schedule. Nothing depends on me living long enough to clean up. Inside the
   guest there is a second, earlier `shutdown -h` so the normal path is a
   clean exit and the API deadline is only the backstop.
2. **Only four machine types exist in the code.** There is no path that takes
   an arbitrary machine type, and every entry is a single GPU under $6/hour, so
   no typo can start an eight-GPU node.
3. **A ledger refuses the next run.** `tools/gcp_ledger.json` accumulates
   every launch at the on-demand rate and raises before creating anything that
   would pass the cap.

`tests/test_gcpfleet.py` — 27 tests, no network, nothing spent — asserts all
of that: that every instance body carries the deletion deadline, that the
budget refuses, that the ledger survives a crash, that `--plan` needs no
credential, and that the private key is gone from disk before the signing
function returns.

## What it does

```bash
python -m tools.gcpfleet --plan               # costs nothing, needs no key
python -m tools.gcpfleet --run l4 --yes       # then read the serial console
python -m tools.gcpfleet --run a100-80 --yes
python -m tools.gcpfleet --status             # what is up, what it cost
python -m tools.gcpfleet --reap               # delete anything still running
```

The instance gets the package as a base64 tarball in its metadata (66 KiB of
a 256 KiB limit), so it never clones the repository and the repository can
stay private. It installs CuPy, runs `experiments/gpu_bench.py` in fp64, fp32
and NumPy, prints the results to the serial console between markers, and
powers off. I read them back with `getSerialPortOutput`, which is an ordinary
API read: **no SSH key, no firewall rule, no open port, no bucket.**

## What I will not do

Raise the budget, add a machine type, use an on-demand instance where spot
will do, or leave an instance up between turns. If a run needs more than the
plan above, I will stop and say so rather than spend it.

## Revoking

Delete the key (`gcloud iam service-accounts keys delete`), or delete the
service account, or delete the project. Removing the environment variable is
enough to stop me using it.
