# LRASR – Detecția anomaliilor în imagini hiperspectrale

Reproducere în Python a metodei **LRASR** (Low-Rank and Sparse Representation) din:

> Y. Xu, Z. Wu, J. Li, A. Plaza, Z. Wei, *"Anomaly Detection in Hyperspectral Images Based on Low-Rank and Sparse Representation"*, IEEE TGRS, vol. 54, no. 4, 2016.

Fundalul imaginii este reprezentat printr-un dicționar de pixeli de fundal, iar coeficienții de reprezentare formează o matrice de rang mic cu regularizare de sparsitate. Anomaliile rezultă din reziduul `E`, obținut prin optimizarea LADMAP (Algorithm 1 din articol).

## Notă despre „checkpoint” și „antrenare”

LRASR **nu este o metodă de învățare** cu parametri antrenabili, deci nu există greutăți de model în sens clasic. În acest repo termenii sunt mapați astfel:

- **Antrenare** (`train.py`) = construirea dicționarului `D` (K-means + selecție Mahalanobis) și rezolvarea optimizării LADMAP pe un set de date. Aceasta este partea costisitoare computațional.
- **Checkpoint** (`checkpoints/*.npz`) = artefactele persistate ale unei rulări complete: dicționarul `D`, harta de detecție, hiperparametrii (`K, P, β, λ, seed`) și AUC-ul obținut.
- **Evaluare din checkpoint** (`eval_checkpoint.py`) = reîncărcarea hărții de detecție salvate și recalcularea ROC/AUC + regenerarea figurilor, fără a re-rula optimizarea.

## Cele mai bune checkpoint-uri obținute

Parametri: `K=15, P=20, β=0.1, λ=0.1, seed=0` (valorile empirice din articol).

| Set de date | Fișier checkpoint | AUC obținut | AUC raportat în articol |
|---|---|---|---|
| Simulat (San Diego + ținte implantate) | `checkpoints/simulated.npz` | 0.8240 | 0.9597 |
| San Diego real (avioane) | `checkpoints/sandiego.npz` | 0.9859 | 0.9882 |
| Urban HYDICE (vehicule) | `checkpoints/urban.npz` | 0.9719 | 0.9220 |

Pe seturile reale (San Diego, urban) rezultatele sunt apropiate de articol sau le depășesc. Setul simulat are AUC mai mic deoarece spectrul-țintă și poziția patch-ului implantat diferă de cele din articol; rezultatul rămâne sensibil la aceste alegeri.

## Comparație cu detectoarele de referință (din `evaluate.py`)

Exemplu pe setul urban (AUC):

| Global-RX | SegRX | SRD | RPCA-RX | **LRASR** |
|---|---|---|---|---|
| 0.9853 | 0.9899 | 0.9923 | 0.9632 | **0.9719** |

## Structura repo-ului

```
lrasr-hsi-anomaly/
├── lrasr.py             # nucleu: operatori LADMAP, solver, dicționar, detector + baseline-uri, ROC/AUC, HySime
├── data.py              # încărcarea seturilor + eliminare benzi + patch simulat + implantare ținte
├── train.py             # construiește dicționarul + LADMAP -> salvează checkpoint .npz
├── evaluate.py          # rulează toți cei 5 detectori pe seturi, tabele AUC + figuri (reproducere completă)
├── eval_checkpoint.py   # încarcă un checkpoint, recalculează AUC, regenerează ROC + harta de detecție
├── checkpoints/         # checkpoint-urile salvate (.npz)
├── data/                # aici pui fișierele .mat (neincluse în git)
├── figures/             # figuri generate (PNG)
├── results/             # tabelele AUC (text)
├── requirements.txt
└── README.md
```

## Date necesare

Pune în `data/`:

- `Sandiego.mat` (cheie `Sandiego`, 400×400×224)
- `PlaneGT.mat` (cheie `PlaneGT`, 100×100)
- `HYDICE_urban.mat` (chei `data` 80×100×175 și `map` 80×100)

Pentru San Diego se elimină benzile de apă/SNR scăzut (1–6, 33–35, 94–97, 107–113, 153–166, 221–224) → 186 benzi.

## Instalare

```bash
pip install -r requirements.txt
```

## Utilizare

Antrenare (generează checkpoint-urile) – toate seturile sau unul singur:

```bash
python3 train.py --dataset all
python3 train.py --dataset sandiego --K 15 --P 20 --beta 0.1 --lam 0.1 --seed 0
```

Evaluare completă (toți detectorii + figuri + tabele AUC):

```bash
python3 evaluate.py --dataset all
```

Evaluare din checkpoint (rapidă, fără re-optimizare):

```bash
python3 eval_checkpoint.py --checkpoint all
python3 eval_checkpoint.py --checkpoint checkpoints/sandiego.npz
```

## Detalii de implementare

- `reshape` se face column-major (`order='F'`) pentru a fi identic cu MATLAB.
- Operatorii LADMAP: SVT (prag pe valori singulare), shrinkage și minimizare ℓ₂,₁ pe coloane.
- Dicționarul: K-means peste toți pixelii, apoi cei mai apropiați `P` pixeli de media clusterului (distanță Mahalanobis) ca atomi de fundal.
- Detectoare de referință incluse: Global-RX, SegRX, SRD (cu OMP), RPCA-RX.
