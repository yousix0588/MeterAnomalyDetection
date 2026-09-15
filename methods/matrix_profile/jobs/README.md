# Spartan Matrix Profile deployment

This directory deploys the rolling 30-day detector without modifying the existing VAE code, environment, data, or results.

## 1. Upload from Windows

Run from the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\methods\matrix_profile\jobs\deploy_to_spartan.ps1
```

The SSH and SCP commands prompt for the Spartan password because the current account uses password authentication.

## 2. Create the isolated environment

On the Spartan login node:

```bash
cd /data/projects/punim1257/Group14/methods/matrix_profile/jobs
bash setup_matrix_profile.sh
```

If Spartan exposes Python 3.10.4 under a different module name, set it explicitly before setup and submission:

```bash
export PYTHON_MODULE='the exact module from methods/lstm_vae/jobs/02_run_vae_array.slurm'
```

## 3. Mandatory preflight

```bash
sbatch 00_preflight.slurm
squeue --me
```

After completion, inspect the preflight `.out` and `.err` logs. The `/usr/bin/time -v` report must show no more than 20 minutes wall time and less than 2 GB maximum resident memory. Do not submit the full array if either limit is exceeded.

## 4. Submit the full dependency chain

```bash
bash submit_full.sh
```

This submits the 100-part array, aggregation after all array parts succeed, and selected plotting after aggregation succeeds. Check failures with:

```bash
grep -RniE 'Traceback|Killed|Out of memory|No space' /data/projects/punim1257/Group14/runs/matrix_profile/logs
```

Final results are written under:

```text
/data/projects/punim1257/Group14/runs/matrix_profile/results/matrix_profile_rolling30/aggregate
/data/projects/punim1257/Group14/runs/matrix_profile/results/matrix_profile_rolling30/selected_plots
```
