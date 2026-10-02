# Explicit input-resolution transfer for a new recognizer experiment

## Contract

`train-recognizer --initialize-from parent/recognizer.pt --initialize-resize`
permits changing only `image_height` and `max_width` while initializing a new
experiment from compatible selected weights. It does not change default input
sizes, the production model, inference preprocessing, or an existing run.

Without `--initialize-resize`, initialization continues to require identical
model configuration. The new switch requires `--initialize-from`; it cannot be
used alone or to relax `--resume-from`. Full training-state initialization is
still refused. A new or empty output directory is required.

All non-geometry model configuration, model identity/version, vocabulary order
and checksum, state keys and tensor shapes must still match. Source and target
geometry must contain integer heights of at least 16 and widths of at least 32.
Weights-only deserialization and finite-tensor checks remain enabled. A malformed
source or incompatible layer configuration fails before output creation.

The current CRNN pools its CNN features over height before the recurrent encoder.
Its trainable layer shapes therefore do not depend on input height or padded
width. That permits loading the same tensors for this controlled experiment;
it does **not** establish equal predictions or improved recognition at new sizes.
In particular, the bidirectional model is padding-sensitive. Scaling both height
and padded width changes the input image, valid sequence length, amount of
padding, memory requirement, and computation.

## Usage

Start a separate run with an explicitly selected checkpoint and unchanged vocabulary:

```bash
lao-ocr train-recognizer \
  --manifest training/private-data/manifest.jsonl \
  --output training/runs/new-resolution-experiment \
  --initialize-from training/runs/parent/final/recognizer.pt \
  --initialize-resize \
  --image-height 64 \
  --max-width 1024 \
  --batch-size 8 \
  --learning-rate 0.0001 \
  --epochs 3 \
  --device mps
```

The paths above are examples, not generated artifacts. Record a data split,
evaluation set and stopping rule before starting the actual experiment. An
input-resolution change is not new training data or a source-rights grant.

For the Python API, use `initialize_from=...` and `initialize_resize=True` with
an appropriate `TrainingConfig`. Only model weights are transferred. Optimizer,
shuffle/RNG state, epoch history and model selection begin afresh under the new
configuration. The data loader, development evaluator and export use that
run's own geometry rather than the parent's input size.

After the first stage, continue with ordinary strict `resume_from` and the
same new geometry/configuration. Do not repeat `--initialize-resize` on resume.
Changing geometry during an exact resume still fails. New geometry remains
subject to the existing CTC-capacity check for every input image.

## Provenance and reporting

Initialization records `resize_authorized`, `input_geometry_changed`,
`source_input_geometry`, `target_input_geometry`, source checkpoint SHA-256,
and the existing fresh-optimizer/history/RNG declarations. This provenance is
retained in resumable state, the selected native checkpoint and run metadata,
including after subsequent exact resumes. Portable export metadata already
records the target model configuration; it does not independently verify the
training lineage.

Compare the same raw evaluation images using each model's declared preprocessing,
and report the input sizes alongside results. Equal update/sample counts are not
equal arithmetic cost, memory, wall-clock time, or necessarily equal optimization
opportunity. Never substitute a development-selected intermediate checkpoint
for a pre-recorded final-epoch result without labeling that separate choice.

## Regression coverage

`test_recognizer_initialization_resize_optional.py` tests exact initial weights
before the first forward pass, fresh optimizer/history, persistent initialization
provenance, strict resume, default rejection, malformed source/target geometry,
architecture/vocabulary/weight faults, and a real CNN/LSTM forward pass at the new
input geometry. CLI tests exercise forwarding and rejection without an
initialization checkpoint. These software checks do not establish OCR accuracy.
