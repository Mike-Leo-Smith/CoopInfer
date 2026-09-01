from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from typing import Any, Dict, Mapping, Optional, Tuple

from .ir import IRNode, ModelIR, TensorEdge


def canonicalize_aten_target(target: Any) -> str:
    """Map common framework operators to hardware-cost classes.

    The canonical names are intentionally model-agnostic. Cost backends can use
    them without knowing whether the graph came from a VLA, LLM, CNN, or another
    PyTorch model.
    """

    text = str(target)
    lowered = text.lower()
    if any(token in lowered for token in ("aten.mm", "aten.matmul", "aten.addmm", "aten.linear")):
        return "gemm"
    if "aten.bmm" in lowered:
        return "batched_gemm"
    if "scaled_dot_product_attention" in lowered or "attention" in lowered:
        return "attention"
    if any(token in lowered for token in ("convolution", "conv1d", "conv2d", "conv3d")):
        return "conv"
    if any(token in lowered for token in ("layer_norm", "rms_norm", "group_norm", "batch_norm")):
        return "norm"
    if "softmax" in lowered:
        return "softmax"
    if any(token in lowered for token in ("mean", "sum", "amax", "amin", "prod")):
        return "reduction"
    if "embedding" in lowered:
        return "embedding"
    if any(token in lowered for token in ("cat", "clone", "contiguous", "copy", "to_copy")):
        return "memory"
    if any(
        token in lowered
        for token in (
            "reshape",
            "view",
            "permute",
            "transpose",
            "getitem",
            "slice",
            "select",
            "narrow",
            "unsqueeze",
            "squeeze",
            "expand",
            "detach",
        )
    ):
        return "transform"
    if any(
        token in lowered
        for token in (
            "add",
            "sub",
            "mul",
            "div",
            "silu",
            "gelu",
            "relu",
            "tanh",
            "sigmoid",
            "pow",
            "rsqrt",
            "sqrt",
            "exp",
            "sin",
            "cos",
            "where",
        )
    ):
        return "elementwise"
    return text


def capture_model(
    model: Any,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = False,
    dynamic_shapes: Any = None,
) -> ModelIR:
    """Capture an nn.Module with torch.export and convert it to a generic ModelIR.

    Torch remains optional for the core CoopInfer package. Import happens only
    when this frontend is used.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for coopinfer.frontend.torch_export.capture_model"
        ) from exc

    export_kwargs: Dict[str, Any] = {
        "args": tuple(args),
        "kwargs": dict(kwargs or {}),
        "strict": bool(strict),
    }
    if dynamic_shapes is not None:
        export_kwargs["dynamic_shapes"] = dynamic_shapes
    exported_program = torch.export.export(model, **export_kwargs)
    return capture_exported_program(exported_program)


def capture_exported_program(exported_program: Any) -> ModelIR:
    graph = exported_program.graph
    input_kinds = _input_kinds(exported_program)

    nodes = []
    included: Dict[str, IRNode] = {}
    fx_to_id: Dict[Any, str] = {}

    for fx_node in graph.nodes:
        output_tensors = _tensor_descriptors(fx_node.meta.get("val"))
        common_metadata = {
            "output_size_bytes": _value_nbytes(fx_node.meta.get("val")),
            "output_tensors": output_tensors,
        }

        if fx_node.op == "placeholder":
            kind = input_kinds.get(fx_node.name, "USER_INPUT")
            if kind not in {"USER_INPUT", "CUSTOM_OBJ"}:
                continue
            ir_node = IRNode(
                id=str(fx_node.name),
                op="input",
                target=str(fx_node.target),
                module_path=_module_path(fx_node),
                kind="input",
                metadata={
                    **common_metadata,
                    "hard_boundary_after": True,
                    "input_kind": kind,
                },
            )
        elif fx_node.op in {"call_function", "call_method", "call_module"}:
            ir_node = IRNode(
                id=str(fx_node.name),
                op=canonicalize_aten_target(fx_node.target),
                target=str(fx_node.target),
                module_path=_module_path(fx_node),
                kind="op",
                metadata={
                    **common_metadata,
                    "input_tensors": _input_tensor_descriptors(fx_node, input_kinds),
                    "literal_args": _literal_structure(fx_node.args),
                    "literal_kwargs": _literal_structure(fx_node.kwargs),
                    "stack_trace": str(fx_node.meta.get("stack_trace", "")),
                },
            )
        elif fx_node.op == "output":
            ir_node = IRNode(
                id=str(fx_node.name),
                op="output",
                target="output",
                module_path="",
                kind="output",
                metadata={
                    **common_metadata,
                    "hard_boundary_before": True,
                },
            )
        else:
            continue

        nodes.append(ir_node)
        included[ir_node.id] = ir_node
        fx_to_id[fx_node] = ir_node.id

    edges = []
    for fx_node in graph.nodes:
        target_id = fx_to_id.get(fx_node)
        if target_id is None:
            continue
        for input_node in fx_node.all_input_nodes:
            source_id = fx_to_id.get(input_node)
            if source_id is None:
                continue
            size_bytes = float(
                included[source_id].metadata.get("output_size_bytes", 0.0)
            )
            edges.append(
                TensorEdge(
                    source=source_id,
                    target=target_id,
                    size_bytes=size_bytes,
                    tensor_id=source_id,
                    metadata={
                        "tensors": tuple(
                            dict(item)
                            for item in included[source_id].metadata.get(
                                "output_tensors", ()
                            )
                        )
                    },
                )
            )

    return ModelIR.from_parts(
        nodes,
        edges,
        metadata={
            "capture": "torch.export",
            "graph_module": type(exported_program.graph_module).__name__,
            "tensor_metadata": "shape-dtype-numel-v1",
        },
    )


def _input_kinds(exported_program: Any) -> Dict[str, str]:
    result: Dict[str, str] = {}
    signature = getattr(exported_program, "graph_signature", None)
    specs = getattr(signature, "input_specs", ()) if signature is not None else ()
    for spec in specs:
        arg = getattr(spec, "arg", None)
        name = getattr(arg, "name", None)
        if name is None:
            continue
        kind = getattr(spec, "kind", "USER_INPUT")
        kind_name = getattr(kind, "name", str(kind).split(".")[-1])
        result[str(name)] = str(kind_name)
    return result


def _module_path(node: Any) -> str:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, ABCMapping) or not stack:
        return ""
    key = next(reversed(stack))
    value = stack[key]
    if isinstance(value, tuple) and value:
        candidate = value[0]
        if candidate:
            return str(candidate)
    return str(key)


def _shape_values(value: Any) -> list[int | str]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return []
    result: list[int | str] = []
    for dim in shape:
        try:
            result.append(int(dim))
        except (TypeError, ValueError, RuntimeError):
            result.append(str(dim))
    return result


def _tensor_descriptor(value: Any) -> Dict[str, Any] | None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    shape = _shape_values(value)
    numel = None
    if shape and all(isinstance(dim, int) for dim in shape):
        numel_value = 1
        for dim in shape:
            numel_value *= int(dim)
        numel = numel_value
    elif shape == []:
        numel = 1

    try:
        element_size = int(value.element_size())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        element_size = None

    nbytes = None
    if numel is not None and element_size is not None:
        nbytes = int(numel) * int(element_size)

    return {
        "shape": shape,
        "dtype": str(value.dtype),
        "numel": numel,
        "element_size": element_size,
        "nbytes": nbytes,
    }


def _tensor_descriptors(value: Any) -> list[Dict[str, Any]]:
    result: list[Dict[str, Any]] = []

    def visit(item: Any) -> None:
        descriptor = _tensor_descriptor(item)
        if descriptor is not None:
            result.append(descriptor)
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, ABCMapping):
            for child in item.values():
                visit(child)

    visit(value)
    return result


def _input_tensor_descriptors(
    fx_node: Any,
    input_kinds: Mapping[str, str],
) -> list[Dict[str, Any]]:
    result: list[Dict[str, Any]] = []
    for input_node in fx_node.all_input_nodes:
        for descriptor in _tensor_descriptors(input_node.meta.get("val")):
            row = dict(descriptor)
            row["source"] = str(input_node.name)
            row["source_op"] = str(input_node.op)
            if input_node.op == "placeholder":
                row["input_kind"] = input_kinds.get(
                    input_node.name, "UNKNOWN_PLACEHOLDER"
                )
            result.append(row)
    return result


def _literal_structure(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_literal_structure(item) for item in value]
    if isinstance(value, ABCMapping):
        return {str(key): _literal_structure(item) for key, item in value.items()}
    if hasattr(value, "name") and hasattr(value, "op"):
        return {"node": str(value.name)}
    return repr(value)


def _value_nbytes(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (tuple, list)):
        return sum(_value_nbytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_value_nbytes(item) for item in value.values())
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        try:
            return float(int(value.numel()) * int(value.element_size()))
        except (TypeError, ValueError, RuntimeError):
            return 0.0
    return 0.0
