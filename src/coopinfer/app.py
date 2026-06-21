from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import networkx as nx
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
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
        self._syncing_timeline_pan = False
        self._timeline_total_ms = 0.0
        self._timeline_window_ms = 0.0

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._seed_example()
        self.graph = self._graph_from_tables()
        self._draw_graph()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.timeline_scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._resize_timeline_to_viewport()
        return super().eventFilter(watched, event)

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

        layout.addWidget(QLabel("Bandwidth"), 0, 0)
        layout.addWidget(self.bandwidth_spin, 0, 1)
        layout.addWidget(QLabel("Latency"), 1, 0)
        layout.addWidget(self.latency_spin, 1, 1)
        layout.addWidget(QLabel("Weight w"), 2, 0)
        layout.addWidget(self.weight_slider, 2, 1)
        layout.addWidget(self.weight_label, 2, 2)
        weight_hint = QLabel("0 = prefer device utilization, 1 = prefer low latency")
        weight_hint.setStyleSheet("color: #6b7280;")
        layout.addWidget(weight_hint, 3, 1, 1, 2)
        layout.addWidget(QLabel("Solver"), 4, 0)
        layout.addWidget(self.algorithm_combo, 4, 1, 1, 2)
        layout.addWidget(QLabel("Iterations"), 5, 0)
        layout.addWidget(self.iterations_spin, 5, 1, 1, 2)
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
        self.figure = Figure(figsize=(7, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        graph_layout.addWidget(self.canvas)
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
        timeline_controls = QHBoxLayout()
        timeline_controls.addWidget(QLabel("Scale"))
        self.timeline_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_scale_slider.setRange(100, 500)
        self.timeline_scale_slider.setSingleStep(25)
        self.timeline_scale_slider.setPageStep(50)
        self.timeline_scale_slider.setValue(100)
        self.timeline_scale_label = QLabel("100%")
        self.timeline_scale_slider.valueChanged.connect(self._set_timeline_scale)
        timeline_controls.addWidget(self.timeline_scale_slider)
        timeline_controls.addWidget(self.timeline_scale_label)
        timeline_layout.addLayout(timeline_controls)

        pan_controls = QHBoxLayout()
        pan_controls.addWidget(QLabel("Pan (time)"))
        self.timeline_pan_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_pan_slider.setRange(0, 0)
        self.timeline_pan_slider.setEnabled(False)
        self.timeline_pan_label = QLabel("0")
        self.timeline_pan_slider.valueChanged.connect(self._pan_timeline)
        pan_controls.addWidget(self.timeline_pan_slider)
        pan_controls.addWidget(self.timeline_pan_label)
        timeline_layout.addLayout(pan_controls)

        self.timeline_figure = Figure(figsize=(7, 3.0))
        self.timeline_figure.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.18)
        self.timeline_canvas = FigureCanvasQTAgg(self.timeline_figure)
        self.timeline_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.timeline_ax = self.timeline_figure.add_subplot(111)
        self.timeline_content = QWidget()
        self.timeline_content_layout = QHBoxLayout(self.timeline_content)
        self.timeline_content_layout.setContentsMargins(0, 0, 0, 0)
        self.timeline_content_layout.addWidget(self.timeline_canvas)
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(False)
        self.timeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.timeline_scroll.setWidget(self.timeline_content)
        self.timeline_scroll.viewport().installEventFilter(self)
        timeline_layout.addWidget(self.timeline_scroll)
        layout.addWidget(timeline_group, stretch=3)
        self._set_timeline_scale(self.timeline_scale_slider.value(), redraw=False)
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
            self.weight_slider.setValue(round(environment.weight_latency * 100))
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
        self.ax.clear()
        self.ax.set_axis_off()
        if self.graph.number_of_nodes() == 0:
            self.canvas.draw()
            return

        graph = self.graph.copy()
        pos = self._layered_layout(graph)

        node_colors = [
            "#2563eb" if int(attrs.get("x", 0)) == 0 else "#16a34a"
            for _, attrs in graph.nodes(data=True)
        ]
        line_widths = [
            3.0 if attrs.get("fixed_dev", False) else 1.2 for _, attrs in graph.nodes(data=True)
        ]
        edge_colors = []
        edge_styles = []
        for source, target in graph.edges:
            cross_device = int(graph.nodes[source].get("x", 0)) != int(graph.nodes[target].get("x", 0))
            edge_colors.append("#dc2626" if cross_device else "#6b7280")
            edge_styles.append("dashed" if cross_device else "solid")

        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=self.ax,
            node_color=node_colors,
            edgecolors="#111827",
            linewidths=line_widths,
            node_size=950,
        )
        for edge, color, style in zip(graph.edges, edge_colors, edge_styles):
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[edge],
                ax=self.ax,
                edge_color=color,
                style=style,
                arrows=True,
                arrowsize=22,
                min_source_margin=18,
                min_target_margin=26,
                width=1.8,
                connectionstyle="arc3,rad=0.08",
            )
        id_labels = {node: str(node) for node in graph.nodes}
        nx.draw_networkx_labels(
            graph,
            pos,
            labels=id_labels,
            ax=self.ax,
            font_color="white",
            font_weight="bold",
            font_size=9,
        )

        compute_labels = {}
        for node, attrs in graph.nodes(data=True):
            x_value = int(attrs.get("x", 0))
            compute = float(attrs["c_dev"]) if x_value == 0 else float(attrs["c_host"])
            place = "Dev" if x_value == 0 else "Host"
            compute_labels[node] = f"{place}: {compute:g} ms"

        name_dy = 0.42
        compute_dy = 0.42

        name_pos = {node: (xy[0], xy[1] + name_dy) for node, xy in pos.items()}
        compute_pos = {node: (xy[0], xy[1] - compute_dy) for node, xy in pos.items()}
        for node, (x_coord, y_coord) in name_pos.items():
            node_name = str(graph.nodes[node].get("name", node)) or str(node)
            self.ax.annotate(
                node_name,
                xy=pos[node],
                xytext=(x_coord, y_coord),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#0f172a",
                zorder=20,
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.34",
                    "fc": "#ffffff",
                    "ec": "#111827",
                    "lw": 1.2,
                    "alpha": 1.0,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#111827",
                    "lw": 0.7,
                    "shrinkA": 10,
                    "shrinkB": 10,
                },
            )
        nx.draw_networkx_labels(
            graph,
            compute_pos,
            labels=compute_labels,
            ax=self.ax,
            font_color="#1f2937",
            font_size=8,
            horizontalalignment="center",
            bbox={"boxstyle": "round,pad=0.22", "fc": "#f8fafc", "ec": "#94a3b8", "alpha": 0.96},
        )

        environment = self._environment_from_controls()
        edge_labels = {}
        for source, target, attrs in graph.edges(data=True):
            size_mb = float(attrs.get("size", 0.0))
            if int(graph.nodes[source].get("x", 0)) == int(graph.nodes[target].get("x", 0)):
                edge_labels[(source, target)] = f"local\n0 ms"
            else:
                transfer_ms = edge_transfer_ms(size_mb, environment.bandwidth, environment.latency)
                edge_labels[(source, target)] = f"{size_mb:g} MB\n{transfer_ms:.1f} ms"
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            ax=self.ax,
            font_size=8,
            font_color="#111827",
            label_pos=0.5,
            rotate=False,
            bbox={"boxstyle": "round,pad=0.2", "fc": "#fefce8", "ec": "#a16207", "alpha": 0.96},
        )
        all_x = [xy[0] for xy in pos.values()] + [xy[0] for xy in name_pos.values()]
        all_y = [xy[1] for xy in pos.values()] + [xy[1] for xy in name_pos.values()]
        x_padding = 0.22 * max(1.0, max(all_x) - min(all_x))
        y_padding = 0.28 * max(1.0, max(all_y) - min(all_y))
        self.ax.set_xlim(min(all_x) - x_padding, max(all_x) + x_padding)
        self.ax.set_ylim(min(all_y) - y_padding, max(all_y) + y_padding)
        self.canvas.draw()

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
        self._apply_timeline_canvas_size()
        self.timeline_ax.clear()
        if self.solver_result is None:
            self._clear_timeline()
            return

        metrics = self.solver_result.metrics
        assignment = self.solver_result.assignment
        lane_by_x = {0: 4.0, 1: 0.0}
        transfer_lane = 2.0
        lane_name_by_y = {4.0: "Device", 2.0: "Transfer", 0.0: "Host"}
        bar_height = 0.52
        transfer_height = 0.28
        label_counts = {0: 0, 1: 0}

        ordered_nodes = sorted(metrics.start_times, key=lambda node: metrics.start_times[node])
        for node in ordered_nodes:
            start = metrics.start_times[node]
            finish = metrics.finish_times[node]
            duration = finish - start
            x_value = int(assignment[node])
            lane = lane_by_x[x_value]
            color = "#2563eb" if x_value == 0 else "#16a34a"
            attrs = self.graph.nodes[node]
            label_name = str(attrs.get("name", node))

            self.timeline_ax.barh(
                lane,
                duration,
                left=start,
                height=bar_height,
                color=color,
                edgecolor="#111827",
                linewidth=1.0,
                alpha=0.92,
                zorder=3,
            )
            self.timeline_ax.text(
                start + duration / 2.0,
                lane,
                str(node),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
                clip_on=True,
                zorder=4,
            )

            index = label_counts[x_value]
            label_counts[x_value] += 1
            label_offset = 0.72 + (index % 2) * 0.48
            label_y = lane + label_offset if x_value == 0 else lane - label_offset
            va = "bottom" if x_value == 0 else "top"
            label_text = f"{node} {label_name}\n{start:.1f}-{finish:.1f} ms ({duration:.1f})"
            self.timeline_ax.text(
                start + duration / 2.0,
                label_y,
                label_text,
                ha="center",
                va=va,
                color="#111827",
                fontsize=7,
                clip_on=True,
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "#ffffff",
                    "ec": "#9ca3af",
                    "alpha": 0.96,
                },
            )
            self.timeline_ax.plot(
                [start + duration / 2.0, start + duration / 2.0],
                [lane + (bar_height / 2.0 if x_value == 0 else -bar_height / 2.0), label_y],
                color="#9ca3af",
                linewidth=0.7,
                zorder=2,
            )

        transfer_index = 0
        environment = self._environment_from_controls()
        for source, target, attrs in self.graph.edges(data=True):
            if source not in metrics.finish_times or target not in metrics.start_times:
                continue
            if int(assignment[source]) == int(assignment[target]):
                continue
            source_y = lane_by_x[int(assignment[source])]
            target_y = lane_by_x[int(assignment[target])]
            start_x = metrics.finish_times[source]
            transfer_ms = edge_transfer_ms(
                float(attrs.get("size", 0.0)), environment.bandwidth, environment.latency
            )
            end_x = start_x + transfer_ms
            if metrics.start_times[target] < start_x:
                continue
            transfer_y = transfer_lane + ((transfer_index % 3) - 1) * 0.34
            transfer_index += 1
            self.timeline_ax.barh(
                transfer_y,
                transfer_ms,
                left=start_x,
                height=transfer_height,
                color="#f97316",
                edgecolor="#9a3412",
                linewidth=1.0,
                alpha=0.95,
                zorder=4,
            )
            self.timeline_ax.text(
                start_x + transfer_ms / 2.0,
                transfer_y,
                f"{source}->{target}  {float(attrs.get('size', 0.0)):g} MB",
                ha="center",
                va="center",
                fontsize=7,
                color="#111827",
                fontweight="bold",
                clip_on=True,
                zorder=5,
            )
            for x_coord, y0, y1 in (
                (start_x, source_y, transfer_y),
                (end_x, transfer_y, target_y),
            ):
                self.timeline_ax.plot(
                    [x_coord, x_coord],
                    [y0, y1],
                    color="#9a3412",
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=2,
                )
            self.timeline_ax.text(
                start_x + transfer_ms / 2.0,
                transfer_y + 0.28,
                f"{transfer_ms:.1f} ms",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#7c2d12",
                clip_on=True,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "fc": "#fff7ed",
                    "ec": "#f97316",
                    "alpha": 0.9,
                },
                zorder=5,
            )

        max_finish = max(metrics.finish_times.values()) if metrics.finish_times else 1.0
        self._timeline_total_ms = max(1.0, max_finish)
        self._update_timeline_pan_range()
        self._apply_timeline_xlim()
        self.timeline_ax.set_ylim(-1.75, 5.75)
        self.timeline_ax.set_yticks([0.0, 2.0, 4.0])
        self.timeline_ax.set_yticklabels(
            [lane_name_by_y[0.0], lane_name_by_y[2.0], lane_name_by_y[4.0]]
        )
        self.timeline_ax.set_xlabel("Time (ms)")
        self.timeline_ax.grid(axis="x", color="#e5e7eb", linestyle="-", linewidth=0.8)
        self.timeline_ax.set_axisbelow(True)
        self.timeline_ax.spines["top"].set_visible(False)
        self.timeline_ax.spines["right"].set_visible(False)
        self._clip_timeline_artists()
        self.timeline_canvas.draw()

    def _clear_timeline(self) -> None:
        self._apply_timeline_canvas_size()
        self.timeline_ax.clear()
        self.timeline_ax.set_axis_off()
        self._timeline_total_ms = 0.0
        self._timeline_window_ms = 0.0
        self.timeline_ax.text(
            0.5,
            0.5,
            "Click Solve to render schedule timeline",
            ha="center",
            va="center",
            transform=self.timeline_ax.transAxes,
            color="#6b7280",
            fontsize=10,
        )
        self.timeline_canvas.draw()
        self._update_timeline_pan_range()

    def _set_timeline_scale(self, value: int, redraw: bool = True) -> None:
        self.timeline_scale_label.setText(f"{value}%")
        self._apply_timeline_canvas_size()
        self._update_timeline_pan_range()
        if redraw:
            if self.solver_result is None:
                self._clear_timeline()
            else:
                self._draw_timeline()

    def _apply_timeline_canvas_size(self) -> None:
        viewport_width = max(1, self.timeline_scroll.viewport().width())
        viewport_height = max(1, self.timeline_scroll.viewport().height())
        width = viewport_width
        height = viewport_height
        self.timeline_canvas.setMinimumSize(width, height)
        self.timeline_canvas.setFixedSize(width, height)
        self.timeline_content.setMinimumSize(width, height)
        self.timeline_content.setFixedSize(width, height)
        dpi = self.timeline_figure.get_dpi()
        self.timeline_figure.set_size_inches(width / dpi, height / dpi, forward=False)

    def _resize_timeline_to_viewport(self) -> None:
        self._apply_timeline_canvas_size()
        self._update_timeline_pan_range()
        if self.solver_result is None:
            self._clear_timeline()
        else:
            self._draw_timeline()

    def _update_timeline_pan_range(self) -> None:
        scale = self.timeline_scale_slider.value() / 100.0
        self._timeline_window_ms = self._timeline_total_ms / max(1.0, scale)
        max_offset = max(0.0, self._timeline_total_ms - self._timeline_window_ms)
        max_pan = int(round(max_offset * 10.0))
        self._syncing_timeline_pan = True
        try:
            self.timeline_pan_slider.setRange(0, max_pan)
            self.timeline_pan_slider.setEnabled(max_pan > 0)
            self.timeline_pan_slider.setValue(min(self.timeline_pan_slider.value(), max_pan))
            self.timeline_pan_label.setText(f"{self.timeline_pan_slider.value() / 10.0:.1f} ms")
        finally:
            self._syncing_timeline_pan = False

    def _pan_timeline(self, value: int) -> None:
        if self._syncing_timeline_pan:
            return
        self.timeline_pan_label.setText(f"{value / 10.0:.1f} ms")
        self._apply_timeline_xlim()
        self._clip_timeline_artists()
        self.timeline_canvas.draw()

    def _apply_timeline_xlim(self) -> None:
        if self._timeline_total_ms <= 0:
            return
        offset = self.timeline_pan_slider.value() / 10.0
        window = self._timeline_window_ms or self._timeline_total_ms
        left = min(max(0.0, offset), max(0.0, self._timeline_total_ms - window))
        right = min(self._timeline_total_ms, left + window)
        if right <= left:
            right = left + 1.0
        self.timeline_ax.set_xlim(left, right)

    def _clip_timeline_artists(self) -> None:
        for artist in self.timeline_ax.get_children():
            if artist is self.timeline_ax.patch:
                continue
            if hasattr(artist, "set_clip_on"):
                artist.set_clip_on(True)
            if hasattr(artist, "set_clip_path"):
                artist.set_clip_path(self.timeline_ax.patch)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
