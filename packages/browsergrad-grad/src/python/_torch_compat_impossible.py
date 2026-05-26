"""Pile C — physically-impossible-in-browser stubs.

Loud, descriptive NotImplementedError with the architectural reason. Never
silent success — that was greed's mistake (it returned None or 0 from
torch.cuda.* and let labs run "fine" until accuracy collapsed at submission).

Self-contained: doesn't import browsergrad_grad. The factory + every stub
is purely defensive; nothing in here touches real ops.
"""


def _impossible(name, reason):
    """Build a callable that raises NotImplementedError with the architectural
    reason. Pile C uses this for every fake-implementable torch.* symbol so
    failure is loud and the message points the user at WHY it can't work."""
    def _raise(*args, **kwargs):
        raise NotImplementedError(f"{name}: {reason}")
    return _raise


def install_impossible(torch_mod, _types):
    torch_mod.compile = _impossible(
        "torch.compile",
        "requires a compiler runtime not available in browser. Run your model uncompiled.",
    )

    torch_fx = _types.ModuleType("torch.fx")
    torch_fx.symbolic_trace = _impossible(
        "torch.fx.symbolic_trace",
        "FX tracing relies on Python introspection paths we don't model. Use the eager graph.",
    )
    torch_mod.fx = torch_fx

    torch_jit = _types.ModuleType("torch.jit")
    torch_jit.script = _impossible(
        "torch.jit.script",
        "no script compiler in browser. The eager Python path runs the same way.",
    )
    torch_jit.trace = _impossible(
        "torch.jit.trace",
        "no trace compiler in browser. The eager Python path runs the same way.",
    )
    torch_mod.jit = torch_jit

    torch_cuda = _types.ModuleType("torch.cuda")
    torch_cuda.is_available = lambda: False
    torch_cuda.device_count = lambda: 0
    torch_cuda.current_device = _impossible(
        "torch.cuda.current_device",
        "no CUDA runtime in browser; use WebGPU via @unlocalhosted/browsergrad-kernels.",
    )
    torch_mod.cuda = torch_cuda

    torch_distributed = _types.ModuleType("torch.distributed")
    torch_distributed.init_process_group = _impossible(
        "torch.distributed.init_process_group",
        "no multi-machine runtime in browser.",
    )
    torch_distributed.all_reduce = _impossible(
        "torch.distributed.all_reduce",
        "no multi-machine runtime in browser.",
    )
    torch_distributed.is_initialized = lambda: False
    torch_mod.distributed = torch_distributed

    torch_onnx = _types.ModuleType("torch.onnx")
    torch_onnx.export = _impossible(
        "torch.onnx.export",
        "ONNX exporter requires the C++ backend we don't ship.",
    )
    torch_mod.onnx = torch_onnx

    torch_quant = _types.ModuleType("torch.quantization")
    torch_quant.quantize = _impossible(
        "torch.quantization.quantize",
        "quantization toolchain requires backend kernels we don't ship in WASM.",
    )
    torch_mod.quantization = torch_quant
