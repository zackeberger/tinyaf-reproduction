# TinyAF

TinyAF detects the composite endpoint atrial fibrillation (AF) or atrial flutter
(AFL) in one 10-second, single-lead ECG. We provide three TinyAF models at
different sizes.

| Model | Parameters | Default probability threshold |
|---|---:|---:|
| TinyAF-S (default) | 1,279 | 0.414806822797144 |
| TinyAF-M | 3,611 | 0.4413809542877673 |
| TinyAF-L | 7,657 | 0.48858124410228104 |

All three predict the same composite endpoint.
The defaults target 85% combined AFIB/AFL sensitivity on BWH-validation. Checkpoints, thresholds, and
source hashes are in `model_assets/tinyaf/provenance.json`. The C weights are generated
from these checkpoints as exact float32 literals.

We provide three implementations of the TinyAF model:

1. **Python.** The original implementation.
2. **C Reference.** A faithful translation of the Python implementation into C, almost achieving end-to-end parity.
3. **C Optimized.** An optimized version of the C implementation. This is what should be deployed on the ECG103.



## Baselines

The repository includes three Python baseline models, all of which predict the
same AF/AFL endpoint.

| Model / CLI name | Parameters | Network input | Probability threshold |
|---|---:|---|---:|
| HR+HRV / `HRHRV` | 3 | `(B, 2)` mean RR and SDNN, in milliseconds | 0.09417233156767857 |
| SmallNN / `SMALLNN` | 8,669 | `(B, 1000)` conditioned ECG | 0.3922985412771895 |
| LargeNN / `LARGENN` | 6,385,041 | `(B, 1000)` conditioned ECG | 0.6911819363466792 |

Assets live under `model_assets/{largenn,smallnn,hrhrv}/`.
Architecture definitions live in `python_model/models/`.
Shared preprocessing is in `python_model/pipeline.py`, with loading/inference in
`python_model/runtime.py`.



## Python Inference

The input to each model is one 10-second, single-lead ECG and its sampling rate.
Conditioning and feature extraction run prior to inference, when appropriate.

```python
from python_model.runtime import configure_reference, load_model, predict_window

configure_reference() # Set deterministic inference
model = load_model("SmallNN", device="cuda:0")  # Use cpu for a CPU run
prediction = predict_window(raw_ecg_mv, sampling_rate_hz, model)
print(prediction.logit, prediction.status, prediction.error)
```

For multiple models on the same raw window, load them once and call
`predict_models(raw_ecg_mv, sampling_rate_hz, {"TINYAF-S": tinyaf, "HRHRV": hrhrv})`.
Each model receives a batch of one and has an independent result.

TinyAF reports status `-4` when it cannot extract a usable beat.
HR+HRV reports status `-12` when mean RR or SDNN is undefined.



## Environment, Build, and Run

### Python
Create a conda environment and run the Python model.
```sh
conda create -n tinyaf python=3.11
conda activate tinyaf
python -m pip install -r requirements.txt
python -m pip check

# Run the Python version over several examples
python scripts/run_quanta_examples.py \
  --implementation python \
  --examples-dir data/quanta-examples/256hz \
  --models TINYAF-S TINYAF-M TINYAF-L
```

### C

From the repository root, compile and run optimized TinyAF-S with your host
C compiler (GCC on Linux or Apple Clang on macOS):

```sh
mkdir -p compile/optimized/S
cc -O3 -std=c99 -fno-fast-math -ffp-contract=off \
  -Ic_model -DTINYAF_OPTIMIZED=1 \
  -DMODEL_WEIGHTS_HEADER='"TINYAF_S_weights.h"' \
  tinyaf_caller.c c_model/tinyaf.c -lm \
  -o compile/optimized/S/tinyaf_caller

./compile/optimized/S/tinyaf_caller data/quanta-examples/256hz/afib.txt
```

For TinyAF-M or TinyAF-L, replace `S` in the weights filename and output
directory with `M` or `L`. The command above runs the optimized version; for
reference C, use `-DTINYAF_OPTIMIZED=0` and write the output under
`compile/reference/`.

Alternatively, `make` builds both the executable and the shared library used by
the Python C wrapper:

```sh
make                              # default optimized TinyAF-S
make RUNG=M MODE=reference        # fp64 preprocessing + fp32 model
make RUNG=L MODE=optimized        # fp32 preprocessing + fp32 model
```

The multi-example runner requires `make` and builds the optimized C libraries
for the requested models:

```sh
python scripts/run_quanta_examples.py \
  --implementation c \
  --examples-dir data/quanta-examples/256hz \
  --models TINYAF-S TINYAF-M TINYAF-L
```

### Run regression tests

Build all six C model/implementation combinations before running the tests:

```sh
for mode in reference optimized; do
  for model in S M L; do
    make MODE="$mode" RUNG="$model"
  done
done

python -m pytest tests -q -rs
```

The suite uses bundled checkpoints and examples plus generated test data. It
checks model input contracts, asset hashes and thresholds, C errors and
preprocessing, batching across all model families, and evaluation/summary CLI
behavior. The CUDA inference test is skipped when CUDA is unavailable.

For a fresh-checkout test, also run both sets of
[bundled examples](#quanta-examples). To exercise the real dataset workflow,
download and verify the data, prepare its manifests, then run the
[sampled evaluation](#run-shdb-af-iridia-af-cpsc2021) before a full evaluation.

### Regenerate C headers

The generated C headers are included in the repository. To regenerate them,
run from the repository root with the Python environment activated:

```sh
python -m src.export_weights
```

This overwrites `c_model/TINYAF_{S,M,L}_weights.h` using the bundled checkpoints
and threshold metadata in `model_assets/tinyaf/`, and regenerates
`c_model/filter_coefficients.h` using SciPy. Rebuild the affected C models after
regeneration.

### Estimate Cortex-M4 memory usage

This requires `make` and the Arm GNU bare-metal toolchain. Add the
installed toolchain's `bin` directory to your shell's `PATH` (replace the path
below with your installation directory):

```sh
export PATH="/path/to/arm-gnu-toolchain/bin:$PATH"
```

With `arm-none-eabi-gcc`, `arm-none-eabi-ar`, and `arm-none-eabi-size` on `PATH`, run:

```sh
bash scripts/measure_arm.sh
```

The script builds optimized S/M/L libraries, links a small inference harness,
and prints Flash and Static RAM totals to the terminal without saving the table.
Flash is calculated as `text + data`; Static RAM as `data + bss`, including
the 5,120-byte ECG input buffer.



## TinyAF C Prediction API

The main entry point uses the following signature:
```c
int predict_c(const uint16_t *raw_ecg, int raw_len,
              float threshold, float *probability);
```

The input is one 10-second 1-lead ECG sampled at 256 Hz in 12-bit Quanta ADC
counts. That is, `raw_ecg` takes 2,560 unsigned ADC counts in `0..4095`.

`TINYAF_USE_DEFAULT_THRESHOLD` is a C macro defined in `tinyaf.h` as `-1.0f`.
Pass it as the inference function's `threshold` argument to select the compiled
model's default threshold:

```c
float probability;
int status = predict_c(raw_ecg, 2560, TINYAF_USE_DEFAULT_THRESHOLD, &probability);
```

Alternatively, a threshold in `[0, 1]` overrides the model default for that call.

The return value is zero on success and negative on failure. `tinyaf.h` lists
status codes. The probability is NaN on failure, and should be treated separately
than a negative AF/AFL prediction.

`predict_c` uses a reusable static workspace. Calls must be serialized;
overlapping calls from multiple threads or interrupts are not safe.



## Prepare Validation Data

To validate our implementation, we use a few examples from Quanta (that are
pushed in this repo). We also use the datasets SHDB-AF, IRIDIA-AF, and CPSC2021.

### Download Data

To download the datasets, run the commands below from the repository root with
the Python environment activated. The downloader uses AWS when possible, but
requires `wget` and `unzip`.

If those tools are missing, install them in the conda environment:
```sh
conda install -c conda-forge wget unzip
```

To enable AWS install it in your active conda environment:
```sh
conda install -c conda-forge awscli
```

Download all three datasets into `data/`:
```sh
bash scripts/download_datasets.sh all --dir ./data
```

The script verifies SHDB/CPSC files against their published SHA-256 manifests
and IRIDIA downloads against MD5 checksums. Missing or damaged downloads are
retried. IRIDIA recordings are extracted automatically after the archive passes
verification.

Useful alternatives:
```sh
# Download one dataset: shdb, iridia, or cpsc
bash scripts/download_datasets.sh iridia

# Use a different data directory
bash scripts/download_datasets.sh all --dir /path/to/data

# Allow more checksum repair attempts (default: 10)
bash scripts/download_datasets.sh all --max-retries 20

# Download and verify IRIDIA without extracting it yet
bash scripts/download_datasets.sh iridia --skip-iridia-unzip
```

If extraction was skipped, extract before preparing the dataset:
```sh
unzip data/iridia-af/iridia-af-records-v1.0.1.zip -d data/iridia-af/records
```

### Set `config.json` File

The repo depends on a `config.json` file to organize where the data is stored.
The `source` entries must point to the downloaded recordings and annotations.
The `metadata` entries point to where processed metadata lies; those directories
are created automatically. ECG signals remain in `source` and are preprocessed
during inference.

```json
{
  "datasets": {
    "SHDB": {
      "source": "data/shdb-af",
      "metadata": "data/manifests/SHDB",
      "patient_metadata": "data/shdb-af/AdditionalData.csv"
    },
    "IRIDIA": {
      "source": "data/iridia-af/records/iridia-af-records-v1.0.1",
      "metadata": "data/manifests/IRIDIA",
      "patient_metadata": "data/iridia-af/iridia-af-metadata-v1.0.1.csv",
      "native_fs": 200
    },
    "CPSC": {
      "source": "data/cpsc_2021",
      "metadata": "data/manifests/CPSC"
    }
  }
}
```

Relative paths are resolved from the repository root when running the commands
shown here. Absolute paths also work.

`patient_metadata` maps recordings to patients for the saved predictions.

`native_fs: 200` specifies the sampling rate of the released IRIDIA recordings,
whose HDF5 files do not supply that field. SHDB and CPSC rates are read from their
WFDB headers. This setting does not request resampling.


### Validate Data Download
To verify an existing download and IRIDIA's extracted files using these paths:

```sh
mkdir -p logs
python scripts/verify_data.py \
  --config config.json \
  --archive data/iridia-af/iridia-af-records-v1.0.1.zip \
  --output logs/data_integrity.json
```

Adjust the archive path if you changed the download directory. This check uses
the PhysioNet checksum manifests, the IRIDIA archive MD5, and the extracted files'
sizes and CRC32 values from that archive.


### Prepare Datasets

Preparation reads recording headers and rhythm annotations to identify eligible
10-second windows. It writes each window's recording identifier, start/end
samples, source sampling rate, and label into Parquet manifests.

Prepare all three datasets:

```sh
python -m src.datasets.prepare_datasets --config config.json --workers 8
```

Or select datasets using their uppercase names:

```sh
python -m src.datasets.prepare_datasets \
  --config config.json --cohorts SHDB CPSC --workers 4
```



## Run Evaluation

### Quanta Examples

Five examples are included under `data/quanta-examples/256hz` in Quanta's unsigned
12-bit ECG format. Run Python and optimized C for TinyAF-S/M/L:

```sh
python scripts/run_quanta_examples.py \
  --implementation both \
  --examples-dir data/quanta-examples/256hz \
  --models TINYAF-S TINYAF-M TINYAF-L
```

Use `--implementation python` for Python only, or `--implementation c` for
optimized C only. If omitted, implementation defaults to `both` and
models default to `TINYAF-S`.

Additional simulator recordings are included under `data/simulated-afib-examples`:

```sh
python scripts/run_quanta_examples.py \
  --implementation both \
  --examples-dir data/simulated-afib-examples/256hz \
  --models TINYAF-S TINYAF-M TINYAF-L
```

Both example directories were resampled from 100 Hz recordings, stored
in the `100hz/`. To regenerate the 256 Hz examples using polyphase resampling:

```sh
python scripts/resample_quanta_examples.py \
  data/quanta-examples/100hz data/quanta-examples/256hz

python scripts/resample_quanta_examples.py \
  data/simulated-afib-examples/100hz data/simulated-afib-examples/256hz
```


### Run SHDB-AF, IRIDIA-AF, CPSC2021

Evaluate TinyAF and the baselines over three datasets. Select models,
implementations, datasets, and inference devices:

| Option | Available values | Example subset |
|---|---|---|
| `--models` | `TINYAF-S`, `TINYAF-M`, `TINYAF-L`, `LARGENN`, `SMALLNN`, `HRHRV` | `--models TINYAF-S HRHRV` |
| `--implementation` | `python`, `reference`, `optimized` | `--implementation python optimized` |
| `--cohorts` | `SHDB`, `IRIDIA`, `CPSC` | `--cohorts SHDB IRIDIA` |
| `--nn-device` | `cpu`, `cuda`, `cuda:N` (one or more) | `--nn-device cuda:0 cuda:1` |
| `--nn-batch-size` | Positive integer (default `1`) | `--nn-batch-size 128` |

Models, implementations, and cohorts accept one or more values, or `all`.
Names are case-insensitive. The default models are `TINYAF-S TINYAF-M TINYAF-L`.
Implementations and cohorts default to all. The same full model names appear in
saved columns, manifests, and summaries, for example `python_TINYAF-S` and
`python_status_TINYAF-S`.

Baselines run only in Python. A mixed Python/C selection runs C for the
TinyAF models only, and a C-only selection containing a baseline is rejected.

`--nn-device` defaults to `cpu` and controls all Python models: TinyAF-S/M/L,
LargeNN, SmallNN, and HR+HRV. Use `cuda` or `cuda:N` to run their forward passes
on GPU. The C implementation runs on CPU. ECG conditioning, peak detection,
and feature extraction run on CPU.

Build the selected C libraries before evaluation;
Python-only evaluation requires no C build. Examples:

**Python only — TinyAF-S and TinyAF-M on SHDB and IRIDIA:**

```sh
python -m scripts.evaluate \
  --config config.json \
  --models TINYAF-S TINYAF-M --implementation python --cohorts SHDB IRIDIA \
  --output logs/shdb-iridia-SM-python --workers 8

python -m scripts.summarize logs/shdb-iridia-SM-python
```

**Optimized C only — TinyAF-S on SHDB:**

```sh
make MODE=optimized RUNG=S

python -m scripts.evaluate \
  --config config.json \
  --models TINYAF-S --implementation optimized --cohorts SHDB \
  --output logs/shdb-S-optimized --workers 8

python -m scripts.summarize logs/shdb-S-optimized
```

**Compare Python and optimized C — TinyAF-S on SHDB:**

```sh
make MODE=optimized RUNG=S

python -m scripts.evaluate \
  --config config.json \
  --models TINYAF-S --implementation python optimized --cohorts SHDB \
  --output logs/shdb-S-comparison --workers 8

python -m scripts.summarize logs/shdb-S-comparison
```

**All models, supported implementations, and datasets:**

Build all six C libraries with the desktop compiler, then select every model:

```sh
for mode in reference optimized; do
  for model in S M L; do
    make MODE="$mode" RUNG="$model"
  done
done

python -m scripts.evaluate \
  --config config.json --models all --nn-device cuda:0 --nn-batch-size 128 \
  --output logs/evaluation --workers 8

python -m scripts.summarize logs/evaluation
```

**Baselines only, on GPU:**

```sh
python -m scripts.evaluate \
  --config config.json --models LARGENN SMALLNN HRHRV --implementation python \
  --nn-device cuda:0 --nn-batch-size 128 --workers 8 --output logs/baselines
python -m scripts.summarize logs/baselines
```

The Quanta example runner also accepts the baseline names and `--nn-device`:

```sh
python scripts/run_quanta_examples.py \
  --examples-dir data/quanta-examples/256hz --implementation python \
  --models LARGENN SMALLNN HRHRV --nn-device cuda:0
```

For a small trial, add `--max-per-record` to any selection. For example, this
runs Python TinyAF-S on at most 20 windows from each SHDB recording:
```sh
python -m scripts.evaluate \
  --config config.json \
  --models TINYAF-S --implementation python --cohorts SHDB \
  --max-per-record 20 --workers 2 --output logs/evaluation-sample

python -m scripts.summarize logs/evaluation-sample
```
`--max-per-record` limits each recording, not the whole dataset. Its default is
`0`, which applies no sampling limit. A sampled run is useful for checking the
workflow; it is not a full dataset evaluation.

The output directory contains:

| Output | Contents |
|---|---|
| `manifest.json` | Model/implementation/dataset selections, selected configuration, window-manifest hashes, code/library/checkpoint hashes, and reference settings. |
| `SHDB/*.parquet`, `IRIDIA/*.parquet`, `CPSC/*.parquet` | Window labels, patient IDs (null for CPSC), and per-model logits, statuses, probabilities, classes, and errors. |
| `summary.json` | Metrics for each selected model and implementation, plus Python/C comparisons when both are selected; written by `scripts.summarize`. |

The summary reports AUROC, average precision, and sensitivity, specificity,
precision, NPV at the saved threshold. It also reports coverage and
failure/validity counts for each selected model/implementation.
When Python and C are both selected, it also reports probability mean absolute
error, maximum absolute logit difference, and classification agreement.

Rerun the same evaluation command to resume: completed recording files are
reused. Use a fresh output directory if the selected configuration, manifests,
recorded code/library/checkpoint hashes, model/implementation/dataset selections,
or sampling limit changes.

## Generate Figures

To generate figures for the paper, run the following.
Each script writes PNG and PDF figures to `fig/outputs/`.

```sh
python fig/scripts/operations.py
python fig/scripts/model_performance_v_ops.py
python fig/scripts/external_validation.py
python fig/scripts/predictive_value.py
python fig/scripts/patient_burden.py
python fig/scripts/endpoint-alignment.py
python fig/scripts/feature_performance_v_ops.py
```

The patient burden figure reads saved TinyAF-S Python predictions from
`logs/evaluation-tinyaf-s-patients/` by default; use `--predictions-dir` to select
another run.

Additional figures read results from the original Weights & Biases projects.
Set `WANDB_API_KEY` for an account with access to those projects. Write these
figures to a separate subdirectory to preserve the external-validation figures
generated above:

```sh
python fig/scripts/external_val.py --output-dir fig/outputs/wandb
python fig/scripts/pareto_frontiers.py --output-dir fig/outputs/wandb
python fig/scripts/performance_profiles.py --output-dir fig/outputs/wandb
```
