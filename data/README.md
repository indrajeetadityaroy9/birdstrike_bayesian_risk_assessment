# Data

## Structure

```
data/
├── processed/
│   ├── telemetry_partitioned/   # GPS telemetry (<10MB files for GitHub)
│   ├── traits/                   # Species morphological traits
│   └── birdstrike/              # UK CAA strike records
└── raw/                          # Original files (gitignored)
```

## Telemetry Data

2.76M GPS points across 6 species, 48 individuals. Partitioned for GitHub.

| Species | Points | Partitions | Total Size |
|---------|--------|------------|------------|
| Larus argentatus | 871,172 | 5 | 43 MB |
| Circus aeruginosus | 834,688 | 5 | 29 MB |
| Buteo buteo | 631,717 | 4 | 6 MB |
| Circus pygargus | 244,011 | 2 | 5 MB |
| Larus fuscus | 113,698 | 1 | 6 MB |
| Circus cyaneus | 63,412 | 1 | 2 MB |

**Load with:**
```python
from dkrl.data import create_dataset

bundle, audit = create_dataset(seed=42)
```

## Sources

| Dataset | Source | License |
|---------|--------|---------|
| GPS Telemetry | LifeWatch INBO Belgium | CC0 1.0 |
| Traits | AVONET Database | CC BY 4.0 |
| Birdstrike | UK CAA 2023-2024 | Open Government |

**Citations:**
- Stienen et al. (2024). LBBG_ZEEBRUGGE. Zenodo. https://doi.org/10.5281/zenodo.12336021
- Stienen et al. (2022). HG_OOSTENDE. Zenodo. https://doi.org/10.5281/zenodo.6594838
- Klaassen et al. (2022). H_GRONINGEN. Zenodo. https://doi.org/10.5281/zenodo.6574736
- Desmet et al. (2023). BOP_RODENT. Zenodo. https://doi.org/10.5281/zenodo.10055071
- Tobias et al. (2022). AVONET. Ecology Letters. https://doi.org/10.1111/ele.13898
