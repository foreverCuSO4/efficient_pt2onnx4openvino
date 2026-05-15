#!/usr/bin/env python3
"""Export anti-dart YOLO weights to the deployed ONNX layout.

This tool is extracted from the latest known exporter used for
`yolo26n_dart_u8_nhwc_notopk_fp32.onnx`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference
from onnxsim import simplify
from ultralytics import YOLO


DEFAULT_MODEL = (
    "/home/jerryzhu/workspace/train_workstation/projects/"
    "anti_dart_noP2_yolo26n_0207/runs/detect/runs/rm_dart/"
    "yolo26n_dart_4090_02072/weights/best.pt"
)
DEFAULT_OUTPUT = "yolo26n_dart_u8_nhwc_notopk_fp32.onnx"
DEFAULT_TARGET_NODE = "/model.23/Transpose"


def parse_imgsz(value: str) -> tuple[int, int]:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("imgsz must look like 384,640 or 384x640")
    try:
        height, width = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("imgsz values must be integers") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("imgsz values must be positive")
    return height, width


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dim.dim_value if dim.dim_value else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def find_target_output(graph: onnx.GraphProto, target_node_name: str) -> str | None:
    for node in graph.node:
        if node.name == target_node_name:
            if not node.output:
                raise RuntimeError(f"target node {target_node_name!r} has no outputs")
            print(f"   found target node: {target_node_name}")
            print(f"   target tensor: {node.output[0]}")
            return node.output[0]

    print(f"   target node {target_node_name!r} not found")
    return None


def infer_output_shape(
    model_proto: onnx.ModelProto,
    target_output_tensor: str,
    input_shape: tuple[int, int],
    num_classes: int,
) -> list[int]:
    try:
        inferred_model = shape_inference.infer_shapes(model_proto)
        for info in inferred_model.graph.value_info:
            if info.name == target_output_tensor:
                shape = tensor_shape(info)
                if shape and all(isinstance(dim, int) and dim > 0 for dim in shape):
                    print(f"   inferred output shape: {shape}")
                    return [int(dim) for dim in shape]
                print(f"   ignored incomplete inferred shape: {shape}")
    except Exception as exc:  # noqa: BLE001 - keep exporter usable if inference fails.
        print(f"   shape inference failed: {exc}")

    height, width = input_shape
    anchors = (height // 8) * (width // 8) + (height // 16) * (width // 16) + (
        height // 32
    ) * (width // 32)
    fallback = [1, 4 + num_classes, anchors]
    print(f"   using fallback output shape: {fallback}")
    return fallback


def inject_uint8_nhwc_preprocess(
    graph: onnx.GraphProto,
    input_shape: tuple[int, int],
    input_name: str,
) -> None:
    if not graph.input:
        raise RuntimeError("ONNX graph has no inputs")

    old_input_node = graph.input[0]
    old_input_name = old_input_node.name
    graph.input.remove(old_input_node)

    height, width = input_shape
    new_input_tensor = helper.make_tensor_value_info(
        input_name,
        TensorProto.UINT8,
        [1, height, width, 3],
    )
    graph.input.insert(0, new_input_tensor)

    cast_out = "prep/cast_out"
    transpose_out = "prep/trans_out"
    div_const_name = "prep/const_255"

    node_cast = helper.make_node(
        "Cast",
        [input_name],
        [cast_out],
        to=TensorProto.FLOAT,
        name="prep/Cast_U8_to_FP32",
    )
    node_transpose = helper.make_node(
        "Transpose",
        [cast_out],
        [transpose_out],
        perm=[0, 3, 1, 2],
        name="prep/Transpose_NHWC_to_NCHW",
    )
    div_const_tensor = helper.make_tensor(
        div_const_name,
        TensorProto.FLOAT,
        [1],
        [255.0],
    )
    node_div = helper.make_node(
        "Div",
        [transpose_out, div_const_name],
        [old_input_name],
        name="prep/Scale_0_255_to_0_1",
    )

    graph.initializer.append(div_const_tensor)
    graph.node.insert(0, node_div)
    graph.node.insert(0, node_transpose)
    graph.node.insert(0, node_cast)


def normalize_output_layout(
    graph: onnx.GraphProto,
    model_proto: onnx.ModelProto,
    num_classes: int,
) -> None:
    if len(graph.output) != 1:
        raise RuntimeError(f"expected one graph output, got {len(graph.output)}")

    output = graph.output[0]
    output_shape = tensor_shape(output)
    expected_attrs = 4 + num_classes

    if len(output_shape) != 3:
        print(f"   keeping original output layout: {output.name} {output_shape}")
        return

    if output_shape[2] == expected_attrs:
        print(f"   original output is already anchors-last: {output.name} {output_shape}")
        return

    if output_shape[1] != expected_attrs:
        print(f"   keeping original output layout: {output.name} {output_shape}")
        return

    anchors = output_shape[2]
    if not isinstance(anchors, int) or anchors <= 0:
        inferred = shape_inference.infer_shapes(model_proto)
        for info in inferred.graph.output:
            if info.name == output.name:
                inferred_shape = tensor_shape(info)
                if len(inferred_shape) == 3:
                    anchors = inferred_shape[2]
                break

    if not isinstance(anchors, int) or anchors <= 0:
        raise RuntimeError(f"cannot infer anchor dimension for output shape {output_shape}")

    original_output_name = output.name
    transposed_output_name = "output_anchors_last"
    node_transpose = helper.make_node(
        "Transpose",
        [original_output_name],
        [transposed_output_name],
        perm=[0, 2, 1],
        name="post/Transpose_Output_to_NxAttrs",
    )
    graph.node.append(node_transpose)

    while len(graph.output) > 0:
        graph.output.pop()
    graph.output.append(
        helper.make_tensor_value_info(
            transposed_output_name,
            TensorProto.FLOAT,
            [1, anchors, expected_attrs],
        )
    )
    print(
        "   output transposed to anchors-last: "
        f"{transposed_output_name} [1, {anchors}, {expected_attrs}]"
    )


def export_base_onnx(
    model_path: Path,
    temp_dir: Path,
    input_shape: tuple[int, int],
    opset: int,
    simplify_model: bool,
) -> Path:
    model = YOLO(str(model_path))
    before = set(model_path.parent.glob("*.onnx"))

    print("[1/3] Exporting base Ultralytics ONNX")
    model.export(
        format="onnx",
        imgsz=list(input_shape),
        half=False,
        simplify=simplify_model,
        opset=opset,
        nms=False,
        dynamic=False,
    )

    expected = model_path.with_suffix(".onnx")
    if expected.exists():
        source = expected
    else:
        created = sorted(
            set(model_path.parent.glob("*.onnx")) - before,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not created:
            raise RuntimeError("Ultralytics export did not create an ONNX file")
        source = created[0]

    temp_onnx = temp_dir / "base_fp32.onnx"
    shutil.move(str(source), temp_onnx)
    print(f"   base ONNX: {temp_onnx}")
    return temp_onnx


def rewrite_model(
    base_onnx: Path,
    output_path: Path,
    input_shape: tuple[int, int],
    num_classes: int,
    target_node_name: str,
    input_name: str,
    run_onnxsim: bool,
) -> None:
    print("[2/3] Rewriting ONNX graph")
    model_proto = onnx.load(base_onnx)
    graph = model_proto.graph

    target_output_tensor = find_target_output(graph, target_node_name)
    if target_output_tensor is not None:
        inferred_shape = infer_output_shape(
            model_proto,
            target_output_tensor,
            input_shape,
            num_classes,
        )

        while len(graph.output) > 0:
            graph.output.pop()
        graph.output.append(
            helper.make_tensor_value_info(
                target_output_tensor,
                TensorProto.FLOAT,
                inferred_shape,
            )
        )
        print(f"   graph output redirected: {target_output_tensor} {inferred_shape}")
    else:
        normalize_output_layout(graph, model_proto, num_classes)

    inject_uint8_nhwc_preprocess(graph, input_shape, input_name)
    print(f"   graph input: {input_name} [1, {input_shape[0]}, {input_shape[1]}, 3] UINT8")

    print("[3/3] Saving final ONNX")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if run_onnxsim:
        try:
            simplified, ok = simplify(model_proto)
            if ok:
                print("   onnxsim check passed")
            else:
                print("   onnxsim check failed; saving simplified graph anyway")
            onnx.save(simplified, output_path)
            return
        except Exception as exc:  # noqa: BLE001 - fallback should still produce a model.
            print(f"   onnxsim failed: {exc}")
            print("   saving unsimplified graph")

    onnx.save(model_proto, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export anti-dart YOLO .pt weights to UINT8 NHWC raw-head ONNX.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="input .pt checkpoint")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output .onnx path")
    parser.add_argument("--imgsz", type=parse_imgsz, default=(384, 640))
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--target-node", default=DEFAULT_TARGET_NODE)
    parser.add_argument("--input-name", default="images_u8")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-initial-simplify", action="store_true")
    parser.add_argument("--no-final-simplify", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = output_path.resolve()

    if not model_path.exists():
        print(f"model does not exist: {model_path}", file=sys.stderr)
        return 2
    if model_path.suffix != ".pt":
        print(f"model should be a .pt checkpoint: {model_path}", file=sys.stderr)
        return 2
    if args.num_classes <= 0:
        print("--num-classes must be positive", file=sys.stderr)
        return 2

    print("Anti-dart ONNX exporter")
    print(f"   model : {model_path}")
    print(f"   output: {output_path}")
    print(f"   imgsz : {args.imgsz[0]},{args.imgsz[1]}")

    temp_parent = output_path.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_dir_obj = tempfile.TemporaryDirectory(prefix=".anti_dart_export_", dir=temp_parent)
    temp_dir = Path(temp_dir_obj.name)

    try:
        base_onnx = export_base_onnx(
            model_path=model_path,
            temp_dir=temp_dir,
            input_shape=args.imgsz,
            opset=args.opset,
            simplify_model=not args.no_initial_simplify,
        )
        rewrite_model(
            base_onnx=base_onnx,
            output_path=output_path,
            input_shape=args.imgsz,
            num_classes=args.num_classes,
            target_node_name=args.target_node,
            input_name=args.input_name,
            run_onnxsim=not args.no_final_simplify,
        )
        print(f"Done: {output_path}")
        return 0
    finally:
        if args.keep_temp:
            print(f"Temporary files kept at: {temp_dir}")
            temp_dir_obj._finalizer.detach()  # noqa: SLF001 - tempfile has no public keep API.
        else:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
