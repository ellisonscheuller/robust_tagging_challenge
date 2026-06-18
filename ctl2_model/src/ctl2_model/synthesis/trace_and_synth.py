from pathlib import Path

import keras
from da4ml.converter import trace_model
from da4ml.trace import comb_trace
from da4ml.codegen import RTLModel, HLSModel
from hgq._dais_tracer.main import _registry


def _identity_handler(layer):
    """Passthrough handler for identity Lambda layers."""
    def _forward(*args, **kwargs):
        return args[0] if args else None
    return _forward


def trace_comb(model, x_sample):
    _registry.setdefault(keras.layers.Lambda, _identity_handler)
    inp, out = trace_model(model)
    comb_logic = comb_trace(inp, out)
    return comb_logic


def generate_rtl(
    comb_logic,
    prj_name: str = "embedding_model",
    output_dir: str = "synth/rtl",
    part_name: str = "xcvu13p-flga2577-2-e",
    clock_period: float = 5.556,
    clock_uncertainty: float = 0.0,
    latency_cutoff: int = 2,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rtl_model = RTLModel(
        comb_logic,
        prj_name=prj_name,
        path=str(output_dir),
        part_name=part_name,
        clock_period=clock_period,
        clock_uncertainty=clock_uncertainty,
        latency_cutoff=latency_cutoff,
    )
    rtl_model.write()

    print(f"RTL written to {output_dir}")
    if hasattr(rtl_model, "_pipe"):
        print(f"Estimated LUTs: {rtl_model._pipe.cost}")
        print(f"Estimated FFs:  {rtl_model._pipe.reg_bits}")
    return rtl_model


def generate_hls_da4ml(
    comb_logic,
    prj_name: str = "embedding_model",
    output_dir: str = "synth/hls_da4ml",
    part_name: str = "xcvu13p-flga2577-2-e",
    clock_period: float = 5.556,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hls_model = HLSModel(
        comb_logic,
        prj_name=prj_name,
        path=str(output_dir),
        part_name=part_name,
        clock_period=clock_period,
    )
    hls_model.compile()
    print(f"HLS project written and compiled at {output_dir}")
    return hls_model


def generate_hls_hls4ml(
    classifier,
    output_dir: str = "synth/hls_hls4ml",
    part_name: str = "xcvu13p-flga2577-2-e",
    io_type: str = "io_parallel",
    clock_period: float = 5.556,
):
    import hls4ml

    output_dir = Path(output_dir)
    hls_config = {
        "Model": {
            "Precision": "ap_fixed<1,0>",
            "ReuseFactor": 128,
        }
    }

    model_hls = hls4ml.converters.convert_from_keras_model(
        classifier,
        output_dir=str(output_dir),
        io_type=io_type,
        part=part_name,
        clock_period=clock_period,
        project_name="embedding_model",
        write_weights_txt=False,
        hls_config=hls_config,
        bit_exact=False,
    )
    model_hls.compile()
    print(f"hls4ml project compiled at {output_dir}")
    return model_hls
