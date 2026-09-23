from pathlib import Path

ROOT = Path(__file__).parents[1]
OWNED_DOCKERFILE = ROOT / "Dockerfile.owned-api"
GPU_COMPOSE = ROOT / "deploy" / "compose.gpu.yml"
GPU_ENV = ROOT / "deploy" / "presets" / "gpu.env.example"
MAIN = ROOT / "services" / "api" / "app" / "main.py"
CLI = ROOT / "src" / "lao_document_ocr" / "cli.py"


def test_owned_api_dockerfile_installs_torch_and_runs_non_root() -> None:
    content = OWNED_DOCKERFILE.read_text(encoding="utf-8")
    assert 'ARG TORCH_INDEX_URL=""' in content
    assert '"torch>=2.5,<3"' in content
    assert "USER 10001:10001" in content
    assert "libgomp1" in content


def test_gpu_compose_selects_owned_engine_and_accelerator() -> None:
    content = GPU_COMPOSE.read_text(encoding="utf-8")
    assert "Dockerfile.owned-api" in content
    assert "OCR_ENGINE: owned" in content
    assert 'OCR_DEVICE: "${OCR_DEVICE:-cuda}"' in content
    assert 'OCR_DECODER: "${OCR_DECODER:-greedy}"' in content
    assert 'OCR_BEAM_WIDTH: "${OCR_BEAM_WIDTH:-10}"' in content
    assert 'OCR_LANGUAGE_MODEL_PATH: "${OCR_LANGUAGE_MODEL_PATH:-}"' in content
    assert 'OCR_LANGUAGE_MODEL_WEIGHT: "${OCR_LANGUAGE_MODEL_WEIGHT:-0}"' in content
    assert (
        'OCR_LANGUAGE_MODEL_TOKEN_BONUS: "${OCR_LANGUAGE_MODEL_TOKEN_BONUS:-0}"'
        in content
    )
    assert 'JOB_MAX_WORKERS: "${GPU_JOB_MAX_WORKERS:-1}"' in content
    assert 'JOB_MAX_ACTIVE: "${GPU_JOB_MAX_ACTIVE:-4}"' in content
    assert 'gpus: all' in content
    assert ':/models:ro"' in content


def test_gpu_preset_defaults_to_one_worker() -> None:
    content = GPU_ENV.read_text(encoding="utf-8")
    assert "OCR_DEVICE=cuda" in content
    assert "OCR_DECODER=greedy" in content
    assert "OCR_BEAM_WIDTH=10" in content
    assert "OCR_LANGUAGE_MODEL_PATH=" in content
    assert "OCR_LANGUAGE_MODEL_WEIGHT=0" in content
    assert "OCR_LANGUAGE_MODEL_TOKEN_BONUS=0" in content
    assert "GPU_JOB_MAX_WORKERS=1" in content
    assert "GPU_JOB_MAX_ACTIVE=4" in content
    assert "TORCH_INDEX_URL=" in content


def test_api_exposes_owned_recognizer_device_setting() -> None:
    content = MAIN.read_text(encoding="utf-8")
    assert 'OCR_DEVICE = os.getenv("OCR_DEVICE", "cpu")' in content
    assert 'OCR_DECODER = os.getenv("OCR_DECODER", "greedy")' in content
    assert 'OCR_BEAM_WIDTH = int(os.getenv("OCR_BEAM_WIDTH", "10"))' in content
    assert 'OCR_LANGUAGE_MODEL_PATH = os.getenv("OCR_LANGUAGE_MODEL_PATH") or None' in content
    assert (
        'OCR_LANGUAGE_MODEL_WEIGHT = float(os.getenv("OCR_LANGUAGE_MODEL_WEIGHT", "0"))'
        in content
    )
    assert "_cached_owned_engine" in content
    assert "device=OCR_DEVICE" not in content
    assert "OCR_DEVICE," in content


def test_cli_exposes_device_choice_for_owned_inference() -> None:
    content = CLI.read_text(encoding="utf-8")
    assert 'choices=["cpu", "cuda", "mps", "auto"]' in content
    assert "device=args.device" in content
