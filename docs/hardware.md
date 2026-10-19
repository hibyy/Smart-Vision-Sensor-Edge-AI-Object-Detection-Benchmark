# Hardware used for measurements

Fill this in once (run `python scripts/00_env_report.py --tag laptop_cpu`
and copy the key fields). Every results table must state which row below it used.

| Tag | Role | CPU | Cores (phys/logical) | RAM | OS | Notes |
|-----|------|-----|----------------------|-----|----|-------|
| laptop_cpu | Edge-proxy latency | (fill) | (fill) | (fill) | Windows (fill) | plugged in, high-performance power plan |
| kaggle_cpu | Secondary CPU latency | (from env json) | | | Linux | shared cloud VM, noisy |
| kaggle_t4 | Accuracy evaluation, GPU reference | n/a | | | Linux | Tesla T4 |

Laptop measurement checklist: plugged in, high-performance power plan,
other heavy apps closed, same thread count across compared configurations.
