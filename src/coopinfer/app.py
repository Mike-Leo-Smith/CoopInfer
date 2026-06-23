from __future__ import annotations

import html
import sys
from typing import Dict, List, Optional, Tuple

import networkx as nx
from PyQt6.QtCore import Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QComboBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

from .evaluator import edge_transfer_ms
from .model import Environment, ProjectState, graph_from_records, load_from_json, save_to_json
from .solver import SolverResult, solve


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Host-Device Cooperative Inference Scheduler")
        self.resize(1280, 800)
        self.graph = nx.DiGraph()
        self.solver_result: Optional[SolverResult] = None
        self._loading_tables = False

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._seed_example()
        self.graph = self._graph_from_tables()
        self._draw_graph()

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(self._build_nodes_group())
        layout.addWidget(self._build_edges_group())
        layout.addWidget(self._build_environment_group())
        layout.addLayout(self._build_actions())
        layout.addStretch(1)
        return panel

    def _build_nodes_group(self) -> QGroupBox:
        group = QGroupBox("节点配置 (Nodes)")
        layout = QVBoxLayout(group)

        self.nodes_table = QTableWidget(0, 5)
        self.nodes_table.setHorizontalHeaderLabels(
            ["Node ID", "Name", "C_dev (ms)", "C_host (ms)", "Fixed on Dev"]
        )
        self.nodes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.nodes_table.itemChanged.connect(self._preview_from_tables)
        layout.addWidget(self.nodes_table)

        buttons = QHBoxLayout()
        add_button = QPushButton("添加节点")
        delete_button = QPushButton("删除选中节点")
        add_button.clicked.connect(self._add_node_row)
        delete_button.clicked.connect(lambda: self._delete_selected_rows(self.nodes_table))
        buttons.addWidget(add_button)
        buttons.addWidget(delete_button)
        layout.addLayout(buttons)
        return group

    def _build_edges_group(self) -> QGroupBox:
        group = QGroupBox("边配置 (Edges)")
        layout = QVBoxLayout(group)

        self.edges_table = QTableWidget(0, 3)
        self.edges_table.setHorizontalHeaderLabels(["Source", "Target", "Size (MB)"])
        self.edges_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.edges_table.itemChanged.connect(self._preview_from_tables)
        layout.addWidget(self.edges_table)

        buttons = QHBoxLayout()
        add_button = QPushButton("添加边")
        delete_button = QPushButton("删除选中边")
        add_button.clicked.connect(self._add_edge_row)
        delete_button.clicked.connect(lambda: self._delete_selected_rows(self.edges_table))
        buttons.addWidget(add_button)
        buttons.addWidget(delete_button)
        layout.addLayout(buttons)
        return group

    def _build_environment_group(self) -> QGroupBox:
        group = QGroupBox("环境与策略 (Environment)")
        layout = QGridLayout(group)

        self.bandwidth_spin = QDoubleSpinBox()
        self.bandwidth_spin.setRange(0.001, 1_000_000.0)
        self.bandwidth_spin.setDecimals(3)
        self.bandwidth_spin.setSuffix(" MB/s")
        self.bandwidth_spin.setValue(50.0)

        self.latency_spin = QDoubleSpinBox()
        self.latency_spin.setRange(0.0, 1_000_000.0)
        self.latency_spin.setDecimals(3)
        self.latency_spin.setSuffix(" ms")
        self.latency_spin.setValue(5.0)

        self.latency_limit_spin = QDoubleSpinBox()
        self.latency_limit_spin.setRange(0.0, 1_000_000.0)
        self.latency_limit_spin.setDecimals(3)
        self.latency_limit_spin.setSuffix(" ms")
        self.latency_limit_spin.setSpecialValueText("No limit")
        self.latency_limit_spin.setValue(0.0)

        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(70)
        self.weight_label = QLabel("0.70")
        self.weight_slider.valueChanged.connect(
            lambda value: self.weight_label.setText(f"{value / 100.0:.2f}")
        )
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(
            ["Auto", "Enumerate", "Random Search", "Simulated Annealing"]
        )
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 1_000_000)
        self.iterations_spin.setValue(3000)
        self.batch_transfers_check = QCheckBox("Batch successive outgoing transfers")

        layout.addWidget(QLabel("Bandwidth"), 0, 0)
        layout.addWidget(self.bandwidth_spin, 0, 1)
        layout.addWidget(QLabel("Latency"), 1, 0)
        layout.addWidget(self.latency_spin, 1, 1)
        layout.addWidget(QLabel("E2E Limit"), 2, 0)
        layout.addWidget(self.latency_limit_spin, 2, 1)
        layout.addWidget(QLabel("Weight w"), 3, 0)
        layout.addWidget(self.weight_slider, 3, 1)
        layout.addWidget(self.weight_label, 3, 2)
        weight_hint = QLabel("0 = prefer device utilization, 1 = prefer low latency")
        weight_hint.setStyleSheet("color: #6b7280;")
        layout.addWidget(weight_hint, 4, 1, 1, 2)
        layout.addWidget(QLabel("Network"), 5, 0)
        layout.addWidget(self.batch_transfers_check, 5, 1, 1, 2)
        layout.addWidget(QLabel("Solver"), 6, 0)
        layout.addWidget(self.algorithm_combo, 6, 1, 1, 2)
        layout.addWidget(QLabel("Iterations"), 7, 0)
        layout.addWidget(self.iterations_spin, 7, 1, 1, 2)
        return group

    def _build_actions(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        load_button = QPushButton("📂 加载配置 (Load)")
        save_button = QPushButton("💾 保存配置 (Save)")
        solve_button = QPushButton("🚀 求解切分预案 (Solve)")
        load_button.clicked.connect(self._load_config)
        save_button.clicked.connect(self._save_config)
        solve_button.clicked.connect(self._solve)
        layout.addWidget(load_button)
        layout.addWidget(save_button)
        layout.addWidget(solve_button)
        return layout

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        graph_group = QGroupBox("拓扑视图")
        graph_layout = QVBoxLayout(graph_group)
        self.graph_view = QWebEngineView()
        graph_layout.addWidget(self.graph_view)
        layout.addWidget(graph_group, stretch=4)

        metrics_group = QGroupBox("性能看板")
        metrics_layout = QGridLayout(metrics_group)
        self.latency_label = QLabel("端到端时延: -- ms")
        self.util_label = QLabel("端侧算力利用率: -- %")
        self.loss_label = QLabel("最优 Loss: --")
        self.mode_label = QLabel("求解模式: --")
        metrics_layout.addWidget(self.latency_label, 0, 0)
        metrics_layout.addWidget(self.util_label, 0, 1)
        metrics_layout.addWidget(self.loss_label, 1, 0)
        metrics_layout.addWidget(self.mode_label, 1, 1)
        layout.addWidget(metrics_group, stretch=0)

        timeline_group = QGroupBox("时序图 (Solved Schedule)")
        timeline_layout = QVBoxLayout(timeline_group)
        self.timeline_view = QWebEngineView()
        timeline_layout.addWidget(self.timeline_view)
        layout.addWidget(timeline_group, stretch=3)
        self._clear_timeline()
        return panel

    def _seed_example(self) -> None:
        self._set_table_data(
            [
                {
                    "id": "v1",
                    "name": "Sensor Pre",
                    "c_dev": 15.5,
                    "c_host": 2.1,
                    "fixed_dev": True,
                },
                {
                    "id": "v2",
                    "name": "Backbone",
                    "c_dev": 40.0,
                    "c_host": 8.5,
                    "fixed_dev": False,
                },
                {
                    "id": "v3",
                    "name": "Control Head",
                    "c_dev": 22.0,
                    "c_host": 4.0,
                    "fixed_dev": False,
                },
            ],
            [
                {"source": "v1", "target": "v2", "size": 2.0},
                {"source": "v2", "target": "v3", "size": 1.0},
            ],
            Environment(),
        )

    def _set_table_data(
        self,
        nodes: List[Dict[str, object]],
        edges: List[Dict[str, object]],
        environment: Environment,
    ) -> None:
        self._loading_tables = True
        try:
            self.nodes_table.setRowCount(0)
            for node in nodes:
                self._add_node_row(
                    str(node["id"]),
                    str(node.get("name", node["id"])),
                    float(node["c_dev"]),
                    float(node["c_host"]),
                    bool(node.get("fixed_dev", False)),
                )

            self.edges_table.setRowCount(0)
            for edge in edges:
                self._add_edge_row(str(edge["source"]), str(edge["target"]), float(edge["size"]))

            self.bandwidth_spin.setValue(environment.bandwidth)
            self.latency_spin.setValue(environment.latency)
            self.latency_limit_spin.setValue(environment.latency_limit)
            self.weight_slider.setValue(round(environment.weight_latency * 100))
            self.batch_transfers_check.setChecked(environment.batch_transfers)
        finally:
            self._loading_tables = False

    def _add_node_row(
        self,
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        c_dev: float = 10.0,
        c_host: float = 2.0,
        fixed_dev: bool = False,
    ) -> None:
        row = self.nodes_table.rowCount()
        self.nodes_table.insertRow(row)
        next_id = node_id or f"v{row + 1}"
        self.nodes_table.setItem(row, 0, QTableWidgetItem(next_id))
        self.nodes_table.setItem(row, 1, QTableWidgetItem(name or next_id))
        self.nodes_table.setItem(row, 2, QTableWidgetItem(f"{c_dev:g}"))
        self.nodes_table.setItem(row, 3, QTableWidgetItem(f"{c_host:g}"))
        fixed_item = QTableWidgetItem()
        fixed_item.setFlags(fixed_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        fixed_item.setCheckState(Qt.CheckState.Checked if fixed_dev else Qt.CheckState.Unchecked)
        self.nodes_table.setItem(row, 4, fixed_item)
        self._preview_from_tables()

    def _add_edge_row(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        size: float = 1.0,
    ) -> None:
        row = self.edges_table.rowCount()
        self.edges_table.insertRow(row)
        self.edges_table.setItem(row, 0, QTableWidgetItem(source or "v1"))
        self.edges_table.setItem(row, 1, QTableWidgetItem(target or "v2"))
        self.edges_table.setItem(row, 2, QTableWidgetItem(f"{size:g}"))
        self._preview_from_tables()

    def _delete_selected_rows(self, table: QTableWidget) -> None:
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)
        self._preview_from_tables()

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load configuration", "", "JSON (*.json)")
        if not path:
            return
        try:
            state = load_from_json(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return

        nodes, edges = self._records_from_graph_for_gui(state.graph)
        self._set_table_data(nodes, edges, state.environment)
        self.graph = state.graph
        self.solver_result = None
        self._clear_metrics()
        self._clear_timeline()
        self._draw_graph()

    def _save_config(self) -> None:
        try:
            graph = self._graph_from_tables()
        except ValueError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save configuration", "", "JSON (*.json)")
        if not path:
            return
        state = ProjectState(graph=graph, environment=self._environment_from_controls())
        try:
            save_to_json(state, path)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _solve(self) -> None:
        try:
            graph = self._graph_from_tables()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc))
            return

        if not nx.is_directed_acyclic_graph(graph):
            QMessageBox.warning(self, "Invalid topology", "Graph contains a cycle; DAG required.")
            return
        free_nodes = [
            node for node, attrs in graph.nodes(data=True) if not attrs.get("fixed_dev", False)
        ]
        if self.algorithm_combo.currentText() == "Enumerate" and len(free_nodes) > 20:
            QMessageBox.warning(
                self,
                "Solver rejected",
                f"Enumerate would require {2 ** len(free_nodes):,} combinations. "
                "Use Auto, Random Search, or Simulated Annealing for this graph.",
            )
            return

        environment = self._environment_from_controls()
        try:
            result = solve(
                graph,
                bandwidth=environment.bandwidth,
                latency=environment.latency,
                weight_latency=environment.weight_latency,
                algorithm=self.algorithm_combo.currentText(),
                heuristic_iterations=self.iterations_spin.value(),
                latency_limit=environment.latency_limit,
                batch_transfers=environment.batch_transfers,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Solve failed", str(exc))
            return

        for node, value in result.assignment.items():
            graph.nodes[node]["x"] = int(value)
        self.graph = graph
        self.solver_result = result
        self._update_metrics(result)
        self._draw_graph()
        self._draw_timeline()

    def _graph_from_tables(self) -> nx.DiGraph:
        nodes, edges = self._read_tables()
        return graph_from_records(nodes, edges)

    def _preview_from_tables(self, *args: object) -> None:
        if self._loading_tables:
            return
        try:
            graph = self._graph_from_tables()
        except (AttributeError, ValueError):
            return
        if not nx.is_directed_acyclic_graph(graph):
            return
        self.graph = graph
        self.solver_result = None
        self._clear_metrics()
        self._clear_timeline()
        self._draw_graph()

    def _read_tables(self) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        nodes: List[Dict[str, object]] = []
        seen: set[str] = set()
        for row in range(self.nodes_table.rowCount()):
            node_id = self._cell_text(self.nodes_table, row, 0)
            if not node_id:
                raise ValueError(f"Node ID is empty at row {row + 1}.")
            if node_id in seen:
                raise ValueError(f"Duplicate Node ID: {node_id}")
            seen.add(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "name": self._cell_text(self.nodes_table, row, 1) or node_id,
                    "c_dev": self._cell_float(self.nodes_table, row, 2, "C_dev (ms)"),
                    "c_host": self._cell_float(self.nodes_table, row, 3, "C_host (ms)"),
                    "fixed_dev": self.nodes_table.item(row, 4).checkState()
                    == Qt.CheckState.Checked,
                }
            )

        if not nodes:
            raise ValueError("At least one node is required.")

        edges: List[Dict[str, object]] = []
        for row in range(self.edges_table.rowCount()):
            source = self._cell_text(self.edges_table, row, 0)
            target = self._cell_text(self.edges_table, row, 1)
            if source not in seen or target not in seen:
                raise ValueError(
                    f"Edge row {row + 1} references unknown nodes: {source} -> {target}"
                )
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "size": self._cell_float(self.edges_table, row, 2, "Size (MB)"),
                }
            )
        return nodes, edges

    def _environment_from_controls(self) -> Environment:
        return Environment(
            bandwidth=self.bandwidth_spin.value(),
            latency=self.latency_spin.value(),
            weight_latency=self.weight_slider.value() / 100.0,
            latency_limit=self.latency_limit_spin.value(),
            batch_transfers=self.batch_transfers_check.isChecked(),
        )

    def _cell_text(self, table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _cell_float(self, table: QTableWidget, row: int, column: int, label: str) -> float:
        value = self._cell_text(table, row, column)
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric at row {row + 1}.") from exc
        if number < 0:
            raise ValueError(f"{label} must be non-negative at row {row + 1}.")
        return number

    def _records_from_graph_for_gui(
        self, graph: nx.DiGraph
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        nodes = [
            {
                "id": node,
                "name": attrs.get("name", node),
                "c_dev": attrs.get("c_dev", 0.0),
                "c_host": attrs.get("c_host", 0.0),
                "fixed_dev": attrs.get("fixed_dev", False),
            }
            for node, attrs in graph.nodes(data=True)
        ]
        edges = [
            {"source": source, "target": target, "size": attrs.get("size", 0.0)}
            for source, target, attrs in graph.edges(data=True)
        ]
        return nodes, edges

    def _draw_graph(self) -> None:
        if self.graph.number_of_nodes() == 0:
            self.graph_view.setHtml(self._placeholder_html("No graph to display"))
            return

        graph = self.graph.copy()
        raw_pos = self._layered_layout(graph)
        x_values = [xy[0] for xy in raw_pos.values()]
        y_values = [xy[1] for xy in raw_pos.values()]
        min_x = min(x_values)
        max_y = max(y_values)
        x_gap_px = 230.0
        y_gap_px = 135.0
        left_pad = 120.0
        top_pad = 95.0
        right_pad = 180.0
        bottom_pad = 110.0
        pos = {
            node: (
                left_pad + (xy[0] - min_x) / 2.9 * x_gap_px,
                top_pad + (max_y - xy[1]) / 1.65 * y_gap_px,
            )
            for node, xy in raw_pos.items()
        }
        width = int(max(x for x, _ in pos.values()) + right_pad)
        height = int(max(y for _, y in pos.values()) + bottom_pad)
        environment = self._environment_from_controls()
        rows = [
            '<defs><marker id="arrow-gray" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#6b7280"/></marker>',
            '<marker id="arrow-red" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#dc2626"/></marker></defs>',
        ]
        for source, target, attrs in graph.edges(data=True):
            sx, sy = pos[source]
            tx, ty = pos[target]
            size_mb = float(attrs.get("size", 0.0))
            if int(graph.nodes[source].get("x", 0)) == int(graph.nodes[target].get("x", 0)):
                color = "#6b7280"
                dash = ""
                marker = "arrow-gray"
                label = "local / 0 ms"
            else:
                color = "#dc2626"
                dash = ' stroke-dasharray="7 5"'
                marker = "arrow-red"
                transfer_ms = edge_transfer_ms(size_mb, environment.bandwidth, environment.latency)
                label = f"{size_mb:g} MB / {transfer_ms:.1f} ms"
            dx = tx - sx
            dy = ty - sy
            dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
            radius = 24.0
            start_x = sx + dx / dist * radius
            start_y = sy + dy / dist * radius
            end_x = tx - dx / dist * (radius + 5)
            end_y = ty - dy / dist * (radius + 5)
            mid_x = (start_x + end_x) / 2.0
            mid_y = (start_y + end_y) / 2.0 - 10.0
            rows.append(
                f'<line class="edge" x1="{start_x:.1f}" y1="{start_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
                f'stroke="{color}"{dash} marker-end="url(#{marker})"><title>{html.escape(str(source))} → {html.escape(str(target))}: {html.escape(label)}</title></line>'
                f'<text class="edge-label" x="{mid_x:.1f}" y="{mid_y:.1f}">{html.escape(label)}</text>'
            )
        for node, attrs in graph.nodes(data=True):
            x_pos, y_pos = pos[node]
            x_value = int(attrs.get("x", 0))
            fill = "#2563eb" if x_value == 0 else "#16a34a"
            place = "Dev" if x_value == 0 else "Host"
            compute = float(attrs["c_dev"]) if x_value == 0 else float(attrs["c_host"])
            stroke_width = 4 if attrs.get("fixed_dev", False) else 1.5
            rows.append(
                f'<g><title>{html.escape(str(node))} {html.escape(str(attrs.get("name", node)))} - {place}: {compute:g} ms</title>'
                f'<text class="node-name" x="{x_pos:.1f}" y="{y_pos - 43:.1f}">{html.escape(str(attrs.get("name", node)))}</text>'
                f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="24" fill="{fill}" stroke="#111827" stroke-width="{stroke_width}" />'
                f'<text class="node-id" x="{x_pos:.1f}" y="{y_pos + 5:.1f}">{html.escape(str(node))}</text>'
                f'<text class="compute" x="{x_pos:.1f}" y="{y_pos + 48:.1f}">{place}: {compute:g} ms</text></g>'
            )
        self.graph_view.setHtml(
            self._svg_page("Topology", "\n".join(rows), width, height, scale_axis="xy")
        )

    def _layered_layout(self, graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        generations = [list(generation) for generation in nx.topological_generations(graph)]
        x_gap = 2.9
        y_gap = 1.65
        pos: Dict[str, Tuple[float, float]] = {}

        for layer_index, generation in enumerate(generations):
            ordered_nodes = sorted(generation, key=str)
            count = len(ordered_nodes)
            total_height = (count - 1) * y_gap
            for index, node in enumerate(ordered_nodes):
                x_coord = layer_index * x_gap
                y_coord = total_height / 2.0 - index * y_gap
                pos[node] = (x_coord, y_coord)

        return pos

    def _update_metrics(self, result: SolverResult) -> None:
        self.latency_label.setText(f"端到端时延: {result.metrics.latency:.1f} ms")
        self.util_label.setText(
            f"端侧算力利用率: {result.metrics.device_utilization * 100.0:.1f} %"
        )
        self.loss_label.setText(f"最优 Loss: {result.metrics.loss:.3f}")
        self.mode_label.setText(f"求解模式: {result.mode} ({result.iterations})")

    def _clear_metrics(self) -> None:
        self.latency_label.setText("端到端时延: -- ms")
        self.util_label.setText("端侧算力利用率: -- %")
        self.loss_label.setText("最优 Loss: --")
        self.mode_label.setText("求解模式: --")

    def _draw_timeline(self) -> None:
        if self.solver_result is None:
            self._clear_timeline()
            return
        self.timeline_view.setHtml(self._timeline_html())

    def _clear_timeline(self) -> None:
        self.timeline_view.setHtml(self._placeholder_html("Click Solve to render schedule timeline"))

    def _timeline_html(self) -> str:
        assert self.solver_result is not None
        metrics = self.solver_result.metrics
        assignment = self.solver_result.assignment
        environment = self._environment_from_controls()

        total_ms = max(1.0, max(metrics.finish_times.values()) if metrics.finish_times else 1.0)
        px_per_ms = 9.0
        left_pad = 120.0
        right_pad = 80.0
        top_pad = 42.0
        lane_height = 58.0
        lane_y = {1: top_pad, "transfer": top_pad + lane_height, 0: top_pad + lane_height * 2}
        width = int(left_pad + total_ms * px_per_ms + right_pad)
        height = int(top_pad + lane_height * 3 + 50)

        def esc(value: object) -> str:
            return html.escape(str(value), quote=True)

        def x_at(ms: float) -> float:
            return left_pad + ms * px_per_ms

        rows = []
        for label, y_pos in [("Host", lane_y[1]), ("Transfer", lane_y["transfer"]), ("Device", lane_y[0])]:
            rows.append(
                f'<line class="lane" x1="0" y1="{y_pos}" x2="{width}" y2="{y_pos}" />'
                f'<text class="lane-label" x="16" y="{y_pos + 5}">{label}</text>'
            )

        tick_step = self._timeline_tick_step(total_ms)
        tick = 0.0
        while tick <= total_ms + 1e-9:
            x_pos = x_at(tick)
            rows.append(
                f'<line class="tick" data-ms="{tick:.6f}" x1="{x_pos:.1f}" y1="20" x2="{x_pos:.1f}" y2="{height - 26}" />'
                f'<text class="tick-label" data-ms="{tick:.6f}" x="{x_pos:.1f}" y="{height - 8}">{tick:.0f} ms</text>'
            )
            tick += tick_step

        for node in sorted(metrics.start_times, key=lambda n: metrics.start_times[n]):
            start = metrics.start_times[node]
            finish = metrics.finish_times[node]
            duration = max(0.1, finish - start)
            x_value = int(assignment[node])
            y_center = lane_y[x_value]
            x_pos = x_at(start)
            bar_width = max(6.0, duration * px_per_ms)
            color = "#2563eb" if x_value == 0 else "#16a34a"
            node_name = esc(self.graph.nodes[node].get("name", node))
            rows.append(
                f'<g><title>{esc(node)} {node_name}: {start:.1f}-{finish:.1f} ms</title>'
                f'<rect class="op" data-start="{start:.6f}" data-duration="{duration:.6f}" x="{x_pos:.1f}" y="{y_center - 16:.1f}" width="{bar_width:.1f}" '
                f'height="32" rx="4" fill="{color}" />'
                f'<text class="op-id" data-start="{start:.6f}" data-duration="{duration:.6f}" data-anchor="center" x="{x_pos + bar_width / 2:.1f}" y="{y_center + 5:.1f}">{esc(node)}</text>'
                f'<text class="op-meta" data-start="{start:.6f}" data-anchor="start" x="{x_pos:.1f}" y="{y_center - 24:.1f}">{esc(node)} {node_name} '
                f'{start:.1f}-{finish:.1f} ms</text></g>'
            )

        for transfer_index, transfer in enumerate(metrics.transfer_records):
            start = transfer.start
            finish = transfer.finish
            transfer_ms = finish - start
            y_center = lane_y["transfer"] + ((transfer_index % 3) - 1) * 12
            x_pos = x_at(start)
            bar_width = max(6.0, transfer_ms * px_per_ms)
            end_x = x_at(finish)
            if transfer.batched:
                edge_text = ", ".join(f"{source}->{target}" for source, target in transfer.edges)
                title = (
                    f"Batch: {edge_text}; {transfer.size_mb:g} MB, "
                    f"{transfer_ms:.1f} ms"
                )
                label = f"batch {len(transfer.edges)} edges / {transfer.size_mb:g} MB"
            else:
                source, target = transfer.edges[0]
                title = (
                    f"{source} -> {target}: {transfer.size_mb:g} MB, "
                    f"{transfer_ms:.1f} ms"
                )
                label = f"{source}->{target} {transfer.size_mb:g} MB"
            rows.append(
                f'<g><title>{esc(title)}</title>'
                f'<rect class="transfer" data-start="{start:.6f}" data-duration="{transfer_ms:.6f}" x="{x_pos:.1f}" y="{y_center - 10:.1f}" width="{bar_width:.1f}" '
                f'height="20" rx="3" />'
                f'<text class="transfer-label" data-start="{start:.6f}" data-duration="{transfer_ms:.6f}" data-anchor="center" x="{x_pos + bar_width / 2:.1f}" y="{y_center + 4:.1f}">'
                f'{esc(label)}</text>'
            )
            for source, target in transfer.edges:
                source_y = lane_y[int(assignment[source])]
                target_y = lane_y[int(assignment[target])]
                rows.append(
                    f'<line class="dep" data-ms="{start:.6f}" x1="{x_pos:.1f}" y1="{source_y:.1f}" x2="{x_pos:.1f}" y2="{y_center:.1f}" />'
                    f'<line class="dep" data-ms="{finish:.6f}" x1="{end_x:.1f}" y1="{y_center:.1f}" x2="{end_x:.1f}" y2="{target_y:.1f}" />'
                )
            rows.append(
                f'</g>'
            )

        svg = "\n".join(rows)
        return self._svg_page(
            "Timeline",
            svg,
            width,
            height,
            extra_info=(
                f"<span>Total: {total_ms:.1f} ms</span>"
                f"<span>Network: {'batched' if environment.batch_transfers else 'serialized'}</span>"
            ),
            scale_axis="timeline",
            timeline_left_pad=left_pad,
            timeline_px_per_ms=px_per_ms,
            timeline_right_pad=right_pad,
            timeline_total_ms=total_ms,
        )

    def _svg_page(
        self,
        title: str,
        svg_body: str,
        width: int,
        height: int,
        extra_info: str = "",
        scale_axis: str = "x",
        timeline_left_pad: float = 0.0,
        timeline_px_per_ms: float = 1.0,
        timeline_right_pad: float = 0.0,
        timeline_total_ms: float = 0.0,
    ) -> str:
        if scale_axis == "timeline":
            scale_script = f"""
    const leftPad = {timeline_left_pad};
    const basePxPerMs = {timeline_px_per_ms};
    const rightPad = {timeline_right_pad};
    const totalMs = {timeline_total_ms};
    const pxPerMs = basePxPerMs * factor;
    const scaledWidth = Math.max({width}, leftPad + totalMs * pxPerMs + rightPad);
    content.style.transform = 'none';
    content.style.width = scaledWidth + 'px';
    content.style.height = {height} + 'px';
    const svg = document.querySelector('svg');
    svg.setAttribute('width', scaledWidth);
    svg.setAttribute('viewBox', '0 0 ' + scaledWidth + ' {height}');
    svg.style.width = scaledWidth + 'px';
    svg.style.height = '{height}px';
    document.querySelectorAll('.lane').forEach((line) => line.setAttribute('x2', scaledWidth));
    document.querySelectorAll('[data-ms]').forEach((el) => {{
      const x = leftPad + Number(el.dataset.ms) * pxPerMs;
      if (el.tagName === 'line') {{
        el.setAttribute('x1', x);
        el.setAttribute('x2', x);
      }} else {{
        el.setAttribute('x', x);
      }}
    }});
    document.querySelectorAll('[data-start]').forEach((el) => {{
      const start = Number(el.dataset.start);
      const duration = Number(el.dataset.duration || 0);
      const x = leftPad + start * pxPerMs;
      const width = Math.max(6, duration * pxPerMs);
      const anchor = el.dataset.anchor || 'start';
      if (el.tagName === 'rect') {{
        el.setAttribute('x', x);
        el.setAttribute('width', width);
      }} else if (anchor === 'center') {{
        el.setAttribute('x', x + width / 2);
      }} else {{
        el.setAttribute('x', x);
      }}
    }});
"""
        else:
            scale_transform = (
                "'scale(' + factor + ')'" if scale_axis == "xy" else "'scaleX(' + factor + ')'"
            )
            scaled_height = f"({height} * factor)" if scale_axis == "xy" else str(height)
            scale_script = f"""
    content.style.transform = {scale_transform};
    content.style.width = ({width} * factor) + 'px';
    content.style.height = {scaled_height} + 'px';
"""
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{ height: 100%; margin: 0; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  body {{ display: flex; flex-direction: column; background: #ffffff; color: #111827; }}
  .toolbar {{ flex: 0 0 auto; display: flex; align-items: center; gap: 10px; padding: 6px 10px; border-bottom: 1px solid #e5e7eb; background: #f9fafb; font-size: 12px; }}
  .toolbar input {{ width: 180px; }}
  #viewport {{ flex: 1 1 auto; overflow: auto; background: #ffffff; }}
  #content {{ transform-origin: top left; width: {width}px; height: {height}px; }}
  svg {{ display: block; width: {width}px; height: {height}px; }}
  .edge {{ stroke-width: 2; fill: none; }}
  .edge-label {{ fill: #111827; font-size: 11px; text-anchor: middle; paint-order: stroke; stroke: #ffffff; stroke-width: 4px; stroke-linejoin: round; }}
  .node-id {{ fill: #ffffff; font-size: 12px; font-weight: 800; text-anchor: middle; pointer-events: none; }}
  .node-name {{ fill: #0f172a; font-size: 12px; font-weight: 800; text-anchor: middle; paint-order: stroke; stroke: #ffffff; stroke-width: 5px; stroke-linejoin: round; }}
  .compute {{ fill: #374151; font-size: 11px; text-anchor: middle; paint-order: stroke; stroke: #ffffff; stroke-width: 4px; stroke-linejoin: round; }}
  .lane {{ stroke: #d1d5db; stroke-width: 1; }}
  .lane-label {{ font-size: 12px; font-weight: 700; fill: #374151; }}
  .tick {{ stroke: #eef2f7; stroke-width: 1; }}
  .tick-label {{ font-size: 10px; fill: #6b7280; text-anchor: middle; }}
  .op {{ stroke: #111827; stroke-width: 1; }}
  .op-id {{ fill: #ffffff; font-size: 11px; font-weight: 700; text-anchor: middle; pointer-events: none; }}
  .op-meta {{ fill: #111827; font-size: 10px; paint-order: stroke; stroke: #ffffff; stroke-width: 3px; stroke-linejoin: round; }}
  .transfer {{ fill: #f97316; stroke: #9a3412; stroke-width: 1; }}
  .transfer-label {{ fill: #111827; font-size: 10px; font-weight: 700; text-anchor: middle; pointer-events: none; }}
  .dep {{ stroke: #9a3412; stroke-width: 1; opacity: 0.75; }}
</style>
</head>
<body>
  <div class="toolbar">
    <strong>{html.escape(title)}</strong>
    <label for="scale">Scale</label>
    <input id="scale" type="range" min="5" max="500" step="5" value="100">
    <span id="scaleText">100%</span>
    {extra_info}
  </div>
  <div id="viewport">
    <div id="content">
      <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{svg_body}</svg>
    </div>
  </div>
<script>
  const scale = document.getElementById('scale');
  const scaleText = document.getElementById('scaleText');
  const content = document.getElementById('content');
  function applyScale() {{
    const factor = Number(scale.value) / 100;
    scaleText.textContent = scale.value + '%';
{scale_script}
  }}
  scale.addEventListener('input', applyScale);
  applyScale();
</script>
</body>
</html>"""

    def _placeholder_html(self, message: str) -> str:
        return f"""
        <!doctype html>
        <html><head><meta charset="utf-8">
        <style>
          html, body {{ height: 100%; margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
          body {{ display: grid; place-items: center; color: #6b7280; background: #ffffff; }}
        </style></head>
        <body>{html.escape(message)}</body></html>
        """

    def _timeline_tick_step(self, total_ms: float) -> float:
        if total_ms <= 50:
            return 5.0
        if total_ms <= 150:
            return 10.0
        if total_ms <= 500:
            return 50.0
        return 100.0


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
