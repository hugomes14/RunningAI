# Resumo da análise exploratória do dataset

- Amostras: **5376**
- Atividades: **34**
- Zonas representadas: **Z1, Z2, Z3, Z4, Z5**
- Candidatos a outlier por IQR: **657**
- Ritmo mediano: **444.3 s/km**
- Declive entre p05 e p95: **-11.4% a 11.5%**

## Cobertura por zona

- Z1: 80 amostras
- Z2: 1190 amostras
- Z3: 1392 amostras
- Z4: 2084 amostras
- Z5: 630 amostras

## Colinearidade elevada

- `zona_ordem` / `zona_limite_inferior_bpm`: 1.000
- `zona_ordem` / `zona_limite_superior_bpm`: 1.000
- `zona_limite_inferior_bpm` / `zona_limite_superior_bpm`: 1.000

Os outliers foram identificados e preservados no dataset. A cobertura desigual entre zonas e atividades deve ser considerada na interpretação da validação.