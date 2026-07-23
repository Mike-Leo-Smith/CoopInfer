from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import networkx as nx
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from .evaluator import base_node_id, frame_index
from .model import Environment
from .solver import SolverResult


DEVICE_COLOR = colors.HexColor("#2563eb")
HOST_COLOR = colors.HexColor("#16a34a")
NETWORK_COLOR = colors.HexColor("#7c3aed")
TEXT_COLOR = colors.HexColor("#111827")
MUTED_COLOR = colors.HexColor("#64748b")
GRID_COLOR = colors.HexColor("#cbd5e1")
PAGE_SIZE = landscape(A4)


def placement_stages(
    graph: nx.DiGraph, assignment: Mapping[str, int]
) -> list[Dict[str, Any]]:
    """Collapse consecutive topological operators with equal placement."""
    ordered = [str(node) for node in nx.lexicographical_topological_sort(graph, key=str)]
    stages: list[Dict[str, Any]] = []
    for node in ordered:
        placement = int(assignment[node])
        if stages and stages[-1]["placement"] == placement:
            stages[-1]["nodes"].append(node)
        else:
            stages.append({"placement": placement, "nodes": [node]})
    for index, stage in enumerate(stages):
        stage["index"] = index
        stage["location"] = "Device" if stage["placement"] == 0 else "Host"
        stage["names"] = [
            str(graph.nodes[node].get("name", node)) for node in stage["nodes"]
        ]
    return stages


def result_to_mapping(
    *,
    config_path: Path,
    graph: nx.DiGraph,
    environment: Environment,
    result: SolverResult,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics = result.metrics
    stages = placement_stages(graph, result.assignment)
    return {
        "config": str(config_path),
        "metadata": dict(metadata or {}),
        "environment": asdict(environment),
        "solver": {
            "mode": result.mode,
            "iterations": result.iterations,
        },
        "assignment": dict(result.assignment),
        "placement_stages": stages,
        "metrics": {
            "amortized_pipeline_span_ms": metrics.latency,
            # Compatibility alias for consumers of pre-0.2 result files.
            "average_latency_ms": metrics.latency,
            "mean_frame_latency_ms": metrics.mean_frame_latency,
            "max_frame_latency_ms": metrics.max_frame_latency,
            "initiation_interval_ms": metrics.initiation_interval,
            "device_utilization": metrics.device_utilization,
            "host_utilization": metrics.host_utilization,
            "network_utilization": metrics.network_utilization,
            "loss": metrics.loss,
            "avg_latency_loss": metrics.avg_latency_loss,
            "max_frame_latency_loss": metrics.max_frame_latency_loss,
            "initiation_interval_loss": metrics.initiation_interval_loss,
            "device_utilization_loss": metrics.device_utilization_loss,
            "pipeline_unroll": metrics.pipeline_unroll,
        },
        "schedule": {
            "start_times_ms": dict(metrics.start_times),
            "finish_times_ms": dict(metrics.finish_times),
            "transfers": [
                {
                    "edges": [list(edge) for edge in transfer.edges],
                    "start_ms": transfer.start,
                    "finish_ms": transfer.finish,
                    "size_mb": transfer.size_mb,
                    "batched": transfer.batched,
                }
                for transfer in metrics.transfer_records
            ],
        },
    }


def export_pipeline_pdf(
    path: Path,
    *,
    config_name: str,
    graph: nx.DiGraph,
    environment: Environment,
    result: SolverResult,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Export a solved placement and schedule as a headless PDF report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(path), pagesize=PAGE_SIZE)
    canvas.setTitle(f"CoopInfer solved pipeline - {config_name}")
    stages = placement_stages(graph, result.assignment)
    _draw_summary_page(
        canvas,
        config_name=config_name,
        graph=graph,
        environment=environment,
        result=result,
        stages=stages,
        metadata=metadata or {},
    )
    _draw_operator_pages(canvas, graph, result.assignment)
    _draw_timeline_pages(canvas, graph, result)
    canvas.save()


def _draw_summary_page(
    canvas: Canvas,
    *,
    config_name: str,
    graph: nx.DiGraph,
    environment: Environment,
    result: SolverResult,
    stages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    width, height = PAGE_SIZE
    canvas.setFillColor(TEXT_COLOR)
    canvas.setFont("Helvetica-Bold", 22)
    canvas.drawString(36, height - 42, "CoopInfer solved pipeline")
    canvas.setFont("Helvetica", 11)
    canvas.setFillColor(MUTED_COLOR)
    canvas.drawString(36, height - 61, config_name)

    metrics = result.metrics
    metric_items = [
        ("Scheduler", f"{result.mode} / {result.iterations} evals"),
        ("Bandwidth", f"{environment.bandwidth:.3f} MB/s"),
        ("Amortized span", f"{metrics.latency:.2f} ms/frame"),
        ("Mean frame E2E", f"{metrics.mean_frame_latency:.2f} ms"),
        ("Max frame latency", f"{metrics.max_frame_latency:.2f} ms"),
        ("Initiation interval", f"{metrics.initiation_interval:.2f} ms"),
        ("Utilization D / H / N", (
            f"{metrics.device_utilization * 100:.1f}% / "
            f"{metrics.host_utilization * 100:.1f}% / "
            f"{metrics.network_utilization * 100:.1f}%"
        )),
        ("Loss", f"{metrics.loss:.4f}"),
        ("Graph", f"{graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges"),
    ]
    if metadata:
        compact = ", ".join(
            f"{key}={value}"
            for key, value in metadata.items()
            if key
            in {
                "batch_size",
                "compression",
                "network_profile",
                "nominal_bandwidth_mbps",
                "effective_bandwidth_mbps",
                "target_hz",
            }
        )
        if compact:
            metric_items.append(("Scenario", compact))

    table_y = height - 91
    col_width = (width - 72) / 4
    for index, (label, value) in enumerate(metric_items):
        row = index // 4
        col = index % 4
        x = 36 + col * col_width
        y = table_y - row * 38
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(MUTED_COLOR)
        canvas.drawString(x, y, label.upper())
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(TEXT_COLOR)
        _draw_ellipsized(canvas, value, x, y - 14, col_width - 10, 10)

    stage_top = table_y - ((len(metric_items) - 1) // 4 + 1) * 38 - 18
    canvas.setFont("Helvetica-Bold", 14)
    canvas.setFillColor(TEXT_COLOR)
    canvas.drawString(36, stage_top, "Placement stages")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED_COLOR)
    canvas.drawRightString(
        width - 36,
        stage_top,
        "Blue = device  |  Green = host  |  order is topological",
    )

    columns = 6
    gap_x = 10
    box_width = (width - 72 - gap_x * (columns - 1)) / columns
    box_height = 46
    gap_y = 12
    start_y = stage_top - 61
    for index, stage in enumerate(stages):
        row = index // columns
        col = index % columns
        x = 36 + col * (box_width + gap_x)
        y = start_y - row * (box_height + gap_y)
        placement = int(stage["placement"])
        color = DEVICE_COLOR if placement == 0 else HOST_COLOR
        canvas.setStrokeColor(color)
        canvas.setFillColor(colors.Color(color.red, color.green, color.blue, alpha=0.10))
        canvas.roundRect(x, y, box_width, box_height, 5, stroke=1, fill=1)
        canvas.setFillColor(color)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(x + 7, y + box_height - 13, f"S{index + 1}  {stage['location']}")
        nodes = list(stage["nodes"])
        if len(nodes) == 1:
            node_text = nodes[0]
        else:
            node_text = f"{nodes[0]} ... {nodes[-1]} ({len(nodes)} ops)"
        canvas.setFillColor(TEXT_COLOR)
        _draw_ellipsized(canvas, node_text, x + 7, y + 13, box_width - 14, 8)

    _footer(canvas, "Solved placement summary")
    canvas.showPage()


def _draw_operator_pages(
    canvas: Canvas,
    graph: nx.DiGraph,
    assignment: Mapping[str, int],
) -> None:
    width, height = PAGE_SIZE
    ordered = [str(node) for node in nx.lexicographical_topological_sort(graph, key=str)]
    rows_per_column = 22
    columns = 2
    rows_per_page = rows_per_column * columns
    for page_start in range(0, len(ordered), rows_per_page):
        canvas.setFillColor(TEXT_COLOR)
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawString(36, height - 42, "Operator placement and cost")
        for local_index, node in enumerate(ordered[page_start : page_start + rows_per_page]):
            column = local_index // rows_per_column
            row = local_index % rows_per_column
            x = 36 + column * (width - 72) / 2
            y = height - 78 - row * 22
            attrs = graph.nodes[node]
            placement = int(assignment[node])
            color = DEVICE_COLOR if placement == 0 else HOST_COLOR
            canvas.setFillColor(color)
            canvas.circle(x + 4, y + 3, 3, stroke=0, fill=1)
            canvas.setFillColor(TEXT_COLOR)
            canvas.setFont("Helvetica-Bold", 8)
            _draw_ellipsized(
                canvas,
                f"{node}  {attrs.get('name', node)}",
                x + 12,
                y,
                185,
                8,
            )
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(MUTED_COLOR)
            canvas.drawRightString(
                x + (width - 72) / 2 - 14,
                y,
                (
                    f"{'Device' if placement == 0 else 'Host'}  "
                    f"D {float(attrs['c_dev']):.3f} ms  "
                    f"H {float(attrs['c_host']):.3f} ms"
                ),
            )
            canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
            canvas.line(x, y - 6, x + (width - 72) / 2 - 14, y - 6)
        _footer(canvas, "Complete logical operator placement")
        canvas.showPage()


def _draw_timeline_pages(
    canvas: Canvas,
    graph: nx.DiGraph,
    result: SolverResult,
) -> None:
    metrics = result.metrics
    total_finish = max(
        [0.0, *metrics.finish_times.values(), *[item.finish for item in metrics.transfer_records]]
    )
    if total_finish <= 0:
        return
    target_windows = 4
    window_ms = max(25.0, _nice_window(total_finish / target_windows))
    window_start = 0.0
    page = 1
    while window_start < total_finish - 1e-9:
        window_end = min(total_finish, window_start + window_ms)
        _draw_timeline_window(
            canvas,
            graph=graph,
            result=result,
            window_start=window_start,
            window_end=window_end,
            page=page,
        )
        canvas.showPage()
        page += 1
        window_start = window_end


def _draw_timeline_window(
    canvas: Canvas,
    *,
    graph: nx.DiGraph,
    result: SolverResult,
    window_start: float,
    window_end: float,
    page: int,
) -> None:
    width, height = PAGE_SIZE
    left = 90
    right = 30
    chart_width = width - left - right
    lane_y = {"Host": height - 160, "Network": height - 280, "Device": height - 400}
    canvas.setFillColor(TEXT_COLOR)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(36, height - 42, "Solved schedule timeline")
    canvas.setFont("Helvetica", 10)
    canvas.setFillColor(MUTED_COLOR)
    canvas.drawString(
        36,
        height - 60,
        f"Window {window_start:.1f}-{window_end:.1f} ms  |  page {page}",
    )

    duration = max(1e-9, window_end - window_start)

    def x_at(value: float) -> float:
        return left + (value - window_start) / duration * chart_width

    for lane, y in lane_y.items():
        canvas.setFillColor(TEXT_COLOR)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawRightString(left - 12, y + 3, lane)
        canvas.setStrokeColor(GRID_COLOR)
        canvas.line(left, y, width - right, y)

    for tick_index in range(11):
        value = window_start + duration * tick_index / 10
        x = x_at(value)
        canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
        canvas.line(x, lane_y["Device"] - 34, x, lane_y["Host"] + 34)
        canvas.setFillColor(MUTED_COLOR)
        canvas.setFont("Helvetica", 7)
        canvas.drawCentredString(x, lane_y["Device"] - 48, f"{value:.0f}")

    for op_id, start in sorted(result.metrics.start_times.items(), key=lambda item: item[1]):
        finish = result.metrics.finish_times[op_id]
        if finish <= window_start or start >= window_end or finish <= start:
            continue
        node = base_node_id(op_id)
        if node not in graph:
            continue
        placement = int(result.assignment[node])
        lane = "Device" if placement == 0 else "Host"
        color = DEVICE_COLOR if placement == 0 else HOST_COLOR
        clipped_start = max(start, window_start)
        clipped_finish = min(finish, window_end)
        x = x_at(clipped_start)
        bar_width = max(1.5, x_at(clipped_finish) - x)
        y = lane_y[lane] - 12
        frame_offset = (frame_index(op_id) % 3) * 8
        y += frame_offset - 8
        canvas.setFillColor(color)
        canvas.roundRect(x, y, bar_width, 9, 2, stroke=0, fill=1)
        if bar_width > 24:
            canvas.setFillColor(TEXT_COLOR)
            canvas.setFont("Helvetica", 6)
            _draw_ellipsized(canvas, op_id, x + 2, y + 1.5, bar_width - 4, 6)

    for transfer in result.metrics.transfer_records:
        if transfer.finish <= window_start or transfer.start >= window_end:
            continue
        clipped_start = max(transfer.start, window_start)
        clipped_finish = min(transfer.finish, window_end)
        x = x_at(clipped_start)
        bar_width = max(1.5, x_at(clipped_finish) - x)
        y = lane_y["Network"] - 6
        canvas.setFillColor(NETWORK_COLOR)
        canvas.roundRect(x, y, bar_width, 12, 2, stroke=0, fill=1)
        if bar_width > 30:
            label = (
                f"{len(transfer.edges)} edges / {transfer.size_mb:.2f} MB"
                if transfer.batched
                else f"{transfer.edges[0][0]}->{transfer.edges[0][1]}"
            )
            canvas.setFillColor(colors.white)
            canvas.setFont("Helvetica", 6)
            _draw_ellipsized(canvas, label, x + 2, y + 2, bar_width - 4, 6)

    canvas.setFillColor(MUTED_COLOR)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        left,
        70,
        (
            f"Amortized span {result.metrics.latency:.2f} ms/frame | "
            f"mean frame {result.metrics.mean_frame_latency:.2f} ms | "
            f"max frame {result.metrics.max_frame_latency:.2f} ms | "
            f"II {result.metrics.initiation_interval:.2f} ms"
        ),
    )
    _footer(canvas, "Actual native-core compute and transfer records")


def _draw_ellipsized(
    canvas: Canvas,
    text: object,
    x: float,
    y: float,
    max_width: float,
    font_size: float,
) -> None:
    value = str(text)
    if stringWidth(value, "Helvetica", font_size) <= max_width:
        canvas.drawString(x, y, value)
        return
    suffix = "..."
    while value and stringWidth(value + suffix, "Helvetica", font_size) > max_width:
        value = value[:-1]
    canvas.drawString(x, y, value + suffix)


def _nice_window(value: float) -> float:
    choices: Iterable[float] = (25, 50, 100, 200, 400, 800, 1600, 3200)
    for choice in choices:
        if choice >= value:
            return choice
    return 6400


def _footer(canvas: Canvas, label: str) -> None:
    width, _ = PAGE_SIZE
    canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
    canvas.line(36, 30, width - 36, 30)
    canvas.setFillColor(MUTED_COLOR)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(36, 18, label)
    canvas.drawRightString(width - 36, 18, "Generated by CoopInfer headless CLI")
