"""#1152 — an explicitly requested FP8 that trained without FP8, past the #835 gates.

Rows 3 to 5 of the issue go through the real ``apply_v028_speed_memory`` on a card
that can run FP8, with torchao stubbed per case: the conversion itself failing
(row 3), ``fp8_attention`` failing partway (row 4) and a model with no attention
projections (row 5) each used to become a yellow line and a bf16 run. They now
end the run at setup, like rows 1 and 2 already do.

The three SFT setup branches that never reached the converter: audio now calls
it after its LoRA attach, the way vision does; unsloth and layer streaming are
refused at config load, because neither can convert what it loads.

Nothing here needs a GPU.
"""

from __future__ import annotations

import sys
import types
from io import StringIO
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from rich.console import Console

from tests.test_issue835_fp8_gate import _Attn, _card

torch = pytest.importorskip("torch")
nn = torch.nn

# Imported before any test patches sys.platform: pydantic builds its schema on
# first import, and reads the platform while doing so.
from soup_cli.config.schema import TrainingConfig  # noqa: E402
from soup_cli.utils.v028_features import apply_v028_speed_memory  # noqa: E402


def _console():
    out = StringIO()
    return out, Console(file=out, width=400, force_terminal=False, color_system=None)


def _stub_torchao(monkeypatch, convert):
    """Install a torchao.float8 whose ``convert_to_float8_training`` is ``convert``."""
    float8 = ModuleType("torchao.float8")
    float8.convert_to_float8_training = convert
    config_mod = ModuleType("torchao.float8.config")
    config_mod.Float8LinearConfig = SimpleNamespace(from_recipe_name=lambda name: name)
    torchao = ModuleType("torchao")
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.float8", float8)
    monkeypatch.setitem(sys.modules, "torchao.float8.config", config_mod)


@pytest.fixture
def converts(monkeypatch):
    """A torchao that converts; returns the recipe of every conversion."""
    calls: list = []
    _stub_torchao(
        monkeypatch, lambda model, config=None, module_filter_fn=None: calls.append(config)
    )
    return calls


def _recipe_config(name: str, top=None, training=None) -> str:
    """A shipped recipe's YAML with top-level / training overrides applied."""
    from soup_cli.recipes.catalog import get_recipe

    doc = yaml.safe_load(get_recipe(name).yaml_str)
    doc.update(top or {})
    doc["training"].update(training or {})
    return yaml.safe_dump(doc)


class TestExplicitFP8NeverDegrades:
    """Rows 3-5: the shared helper, on a Hopper card, torchao present."""

    @pytest.fixture(autouse=True)
    def _hopper(self, monkeypatch):
        _card(monkeypatch, (9, 0))

    def test_row3_a_failed_conversion_stops_the_run(self, monkeypatch):
        """``quantization_aware: fp8``, torchao present, conversion raises. Used to
        print ``(torchao.float8 missing)`` -- false, torchao is here -- and train
        bf16; on the #1154 branch, a different yellow line and still bf16."""

        def _convert(model, config=None, module_filter_fn=None):
            raise RuntimeError("cuBLAS refused the shape")

        _stub_torchao(monkeypatch, _convert)
        out, console = _console()
        with pytest.raises(RuntimeError, match=r"conversion of the model failed.*tensorwise"):
            apply_v028_speed_memory(
                model=_Attn(), tcfg=TrainingConfig(quantization_aware="fp8"),
                base_model="m", console=console, device="cuda",
            )
        assert "missing" not in out.getvalue()
        assert "FP8 training enabled" not in out.getvalue()

    def test_row4_a_partial_attention_conversion_stops_the_run(self, monkeypatch):
        """``fp8_attention``, the attention pass fails partway. The converter's
        "PARTIALLY converted; restart training without the flag" refusal used to
        be caught, printed in yellow, and then training continued on that very
        model."""

        def _convert(model, config=None, module_filter_fn=None):
            if module_filter_fn is not None:  # the attention-only pass
                raise ValueError("unsupported dim")

        _stub_torchao(monkeypatch, _convert)
        out, console = _console()
        with pytest.raises(RuntimeError, match="PARTIALLY converted"):
            apply_v028_speed_memory(
                model=_Attn(),
                tcfg=TrainingConfig(quantization_aware="fp8", fp8_attention=True),
                base_model="m", console=console, device="cuda",
            )
        assert "FP8 attention:" not in out.getvalue()

    def test_row5_no_attention_projections_stops_the_run(self, converts):
        """``fp8_attention`` on a model with no q/k/v/o. The converter refuses the
        silent no-op by design; the helper used to print that refusal and go on
        as the no-op."""

        class _NoAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 4)

        out, console = _console()
        with pytest.raises(ValueError, match="no attention projections"):
            apply_v028_speed_memory(
                model=_NoAttn(),
                tcfg=TrainingConfig(quantization_aware="fp8", fp8_attention=True),
                base_model="m", console=console, device="cuda",
            )
        assert "FP8 attention:" not in out.getvalue()

    def test_control_torchao_present_still_converts(self, converts):
        """The stop is for failures only: a Hopper card with torchao converts
        both passes and reports them."""

        out, console = _console()
        applied = apply_v028_speed_memory(
            model=_Attn(),
            tcfg=TrainingConfig(quantization_aware="fp8", fp8_attention=True),
            base_model="m", console=console, device="cuda",
        )
        assert applied["fp8"] is True and applied["fp8_attention"] is True
        assert converts == ["tensorwise", "tensorwise"]
        assert "FP8 attention enabled (4 projections)" in out.getvalue()

    def test_control_nvfp4_failure_still_degrades(self, monkeypatch):
        """The nvfp4 block is not part of #1152: its converter failing is still a
        yellow line, and the run goes on."""

        def _boom(model):
            raise RuntimeError("no Blackwell")

        monkeypatch.setattr("soup_cli.utils.advanced_precision.apply_nvfp4", _boom)
        out, console = _console()
        applied = apply_v028_speed_memory(
            model=_Attn(), tcfg=TrainingConfig(nvfp4=True),
            base_model="m", console=console, device="cuda",
        )
        assert applied["nvfp4"] is False
        assert "NVFP4: no Blackwell" in out.getvalue()

    def test_control_no_fp8_request_is_untouched(self, monkeypatch):
        """A converter that would fail is never reached by a config that did not
        ask for FP8: nothing raises, nothing is printed, the legacy 3-key dict."""

        def _convert(model, config=None, module_filter_fn=None):
            raise RuntimeError("must not be reached")

        _stub_torchao(monkeypatch, _convert)
        out, console = _console()
        applied = apply_v028_speed_memory(
            model=_Attn(), tcfg=TrainingConfig(), base_model="m",
            console=console, device="cuda",
        )
        assert applied == {"cut_ce": False, "fp8": False, "kernel_auto_compose": False}
        assert out.getvalue() == ""


class TestAudioBranchReachesTheConverter:
    """SFT's audio setup never called ``_apply_quantization_aware``; the shipped
    ``qwen2-audio-7b-sft`` recipe plus ``quantization_aware: fp8`` loaded and
    trained without FP8, with no message. The loaders are faked (as every audio
    setup test here does); the converter path is real, down to a torchao stub
    that records which model it was handed."""

    def _run_audio_setup(self, monkeypatch, training):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from tests.test_issue339_load_dtype import _FakeModel, _FakeProcessor

        cfg = load_config_from_string(_recipe_config("qwen2-audio-7b-sft", training=training))
        loaded = _FakeModel()
        peft_wrapped = _Attn()
        seen: list = []

        def _convert(model, config=None, module_filter_fn=None):
            seen.append((model, config))

        _stub_torchao(monkeypatch, _convert)
        _card(monkeypatch, (9, 0))
        monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: _FakeProcessor()),
            AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k: loaded),
        ))
        monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            get_peft_model=lambda m, _cfg: peft_wrapped if m is loaded else None,
            prepare_model_for_kbit_training=lambda m, **kwargs: m,
        ))
        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **kwargs: None,
        )
        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False
        wrapper._setup_audio_transformers(cfg, cfg.training)
        return wrapper, peft_wrapped, seen

    def test_audio_setup_converts_the_lora_wrapped_model(self, monkeypatch):
        wrapper, peft_wrapped, seen = self._run_audio_setup(
            monkeypatch, {"quantization_aware": "fp8"}
        )
        assert wrapper.model is peft_wrapped
        assert [(m is peft_wrapped, c) for m, c in seen] == [(True, "tensorwise")]

    def test_audio_setup_without_fp8_converts_nothing(self, monkeypatch):
        _wrapper, _peft, seen = self._run_audio_setup(monkeypatch, {})
        assert seen == []


class TestUnslothAndStreamingRefuseFP8AtLoad:
    """Measured on the #1154 branch: ``llama3.1-8b-sft`` + ``backend: unsloth``
    and + ``stream_layers: true`` both LOADED with ``quantization_aware: fp8``
    and with ``fp8_attention: true``, then trained without FP8. Neither branch
    can apply it, so the config is refused with the reason."""

    def _load(self, top=None, training=None):
        from soup_cli.config.loader import load_config_from_string

        return load_config_from_string(_recipe_config("llama3.1-8b-sft", top, training))

    @pytest.mark.parametrize(
        "fp8", [{"quantization_aware": "fp8"},
                {"quantization_aware": "fp8", "fp8_attention": True}],
        ids=["quantization_aware", "plus_fp8_attention"],
    )
    def test_unsloth_refuses_fp8(self, fp8):
        with pytest.raises(ValueError, match=r"unsloth.*fused kernels") as info:
            self._load({"backend": "unsloth"}, fp8)
        message = str(info.value)
        assert "quantization_aware='fp8'" in message
        assert ("fp8_attention" in message) is bool(fp8.get("fp8_attention"))

    @pytest.mark.parametrize(
        "fp8", [{"quantization_aware": "fp8"},
                {"quantization_aware": "fp8", "fp8_attention": True}],
        ids=["quantization_aware", "plus_fp8_attention"],
    )
    def test_streaming_refuses_fp8(self, fp8):
        with pytest.raises(ValueError, match=r"stream_layers.*meta skeleton") as info:
            self._load(training={"stream_layers": True, "batch_size": 1, **fp8})
        message = str(info.value)
        assert "quantization_aware='fp8'" in message
        assert ("fp8_attention" in message) is bool(fp8.get("fp8_attention"))

    def test_the_unsloth_message_is_validate_fp8_configs(self):
        """One message, not a second one: the schema raises the text
        ``utils/fp8.validate_fp8_config`` already carried (and nothing called)."""
        from soup_cli.utils.fp8 import validate_fp8_config

        [expected] = validate_fp8_config("fp8", "unsloth", "cuda")
        with pytest.raises(ValueError) as info:
            self._load({"backend": "unsloth"}, {"quantization_aware": "fp8"})
        assert expected in str(info.value)

    def test_control_without_fp8_both_branches_still_load(self):
        assert self._load({"backend": "unsloth"}).backend == "unsloth"
        assert self._load(training={"stream_layers": True, "batch_size": 1}).training.stream_layers

    def test_control_transformers_still_accepts_fp8(self):
        cfg = self._load(training={"quantization_aware": "fp8", "fp8_attention": True})
        assert cfg.training.quantization_aware == "fp8" and cfg.training.fp8_attention
