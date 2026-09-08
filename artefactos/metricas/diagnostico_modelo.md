# Diagnóstico do modelo

- Modelo vencedor: **GradientBoosting**.
- MAE macro fora do treino: **68.58 s/km**.
- RMSE macro fora do treino: **87.65 s/km**.
- Melhoria face ao Dummy: **30.6%**.
- Zona com maior MAE observado: **Z1**, com 80 amostras.

## Features com maior importância por permutação

- `subida_acumulada_m`: 15.18 s/km
- `declive_pct`: 10.99 s/km
- `distancia_inicio_km`: 4.07 s/km

Estas associações são descritivas e não demonstram causalidade.
