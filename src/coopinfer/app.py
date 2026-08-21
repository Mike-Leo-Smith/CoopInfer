from __future__ import annotations

import html
import re
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
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

from .evaluator import edge_transfer_ms
from .model import (
    Environment,
    ProjectState,
    graph_from_records,
    load_from_json,
    save_to_json,
    validate_environment,
)
from .solver import SolverResult, solve


_TIMELINE_LAYER_ID = re.compile(
    r"layer_s0*(?P<stack>\d+)_l0*(?P<layer>\d+)"
    r"(?:::step0*(?P<step>\d+))?"
    r"(?:\[f(?P<frame>\d+)\])?$"
)


def _timeline_short_node_label(node_id: object) -> str:
    text = str(node_id)
    match = _TIMELINE_LAYER_ID.search(text)
    if match:
        parts = [f"S{int(match.group('stack'))}", f"L{int(match.group('layer'))}"]
        if match.group("step") is not None:
            parts.append(f"D{int(match.group('step'))}")
        if match.group("frame") is not None:
            parts.append(f"F{int(match.group('frame'))}")
        return "/".join(parts)
    leaf = text.rsplit(".", 1)[-1]
    return leaf if len(leaf) <= 18 else f"{leaf[:8]}…{leaf[-7:]}"


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
        self._draw_config_graph()
        self._clear_solved_graph()

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        data_splitter = QSplitter(Qt.Orientation.Vertical)
        data_splitter.addWidget(self._build_nodes_group())
        data_splitter.addWidget(self._build_edges_group())
        data_splitter.setStretchFactor(0, 3)
        data_splitter.setStretchFactor(1, 2)
        layout.addWidget(data_splitter, stretch=1)
        layout.addWidget(self._build_environment_group(), stretch=0)
        layout.addWidget(self._build_actions_group(), stretch=0)
        return panel

    def _build_nodes_group(self) -> QGroupBox:
        group = QGroupBox("节点配置 (Nodes)")
        group.setMinimumHeight(120)
        layout = QVBoxLayout(group)

        self.nodes_table = QTableWidget(0, 7)
        self.nodes_table.setHorizontalHeaderLabels(
            [
                "Node ID",
                "Name",
                "C_dev (ms)",
                "C_host (ms)",
                "Source Period (ms)",
                "Source Phase (ms)",
                "Placement",
            ]
        )
        node_header = self.nodes_table.horizontalHeader()
        node_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        node_header.setStretchLastSection(False)
        for column, width in enumerate([95, 150, 95, 95, 135, 125, 105]):
            self.nodes_table.setColumnWidth(column, width)
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
        group.setMinimumHeight(100)
        layout = QVBoxLayout(group)

        self.edges_table = QTableWidget(0, 3)
        self.edges_table.setHorizontalHeaderLabels(["Source", "Target", "Size (MB)"])
        edge_header = self.edges_table.horizontalHeader()
        edge_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        edge_header.setStretchLastSection(False)
        for column, width in enumerate([130, 130, 100]):
            self.edges_table.setColumnWidth(column, width)
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
        group.setMinimumHeight(300)
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

        self.max_frame_latency_limit_spin = QDoubleSpinBox()
        self.max_frame_latency_limit_spin.setRange(0.0, 1_000_000.0)
        self.max_frame_latency_limit_spin.setDecimals(3)
        self.max_frame_latency_limit_spin.setSuffix(" ms")
        self.max_frame_latency_limit_spin.setSpecialValueText("No limit")
        self.max_frame_latency_limit_spin.setValue(0.0)

        (
            self.weight_avg_latency_slider,
            self.weight_avg_latency_label,
        ) = self._objective_weight_slider(0.7)
        (
            self.weight_max_latency_slider,
            self.weight_max_latency_label,
        ) = self._objective_weight_slider(0.3)
        (
            self.weight_device_utilization_slider,
            self.weight_device_utilization_label,
        ) = self._objective_weight_slider(0.3)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(
            ["Auto", "Enumerate", "Random Search", "Simulated Annealing"]
        )
        self.algorithm_combo.currentTextChanged.connect(self._update_sa_controls_enabled)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 1_000_000)
        self.iterations_spin.setValue(3000)
        self.solver_threads_spin = QSpinBox()
        self.solver_threads_spin.setRange(0, 256)
        self.solver_threads_spin.setSpecialValueText("Auto")
        self.solver_threads_spin.setValue(0)
        self.anneal_initial_temp_spin = QDoubleSpinBox()
        self.anneal_initial_temp_spin.setRange(0.000001, 1_000_000.0)
        self.anneal_initial_temp_spin.setDecimals(6)
        self.anneal_initial_temp_spin.setValue(1.0)
        self.anneal_final_temp_spin = QDoubleSpinBox()
        self.anneal_final_temp_spin.setRange(0.000001, 1_000_000.0)
        self.anneal_final_temp_spin.setDecimals(6)
        self.anneal_final_temp_spin.setValue(0.01)
        self.batch_transfers_check = QCheckBox("Batch successive outgoing transfers")
        self.pipeline_unroll_spin = QSpinBox()
        self.pipeline_unroll_spin.setRange(1, 64)
        self.pipeline_unroll_spin.setValue(1)

        layout.addWidget(QLabel("Bandwidth"), 0, 0)
        layout.addWidget(self.bandwidth_spin, 0, 1)
        layout.addWidget(QLabel("Latency"), 1, 0)
        layout.addWidget(self.latency_spin, 1, 1)
        layout.addWidget(QLabel("Avg E2E Limit"), 2, 0)
        layout.addWidget(self.latency_limit_spin, 2, 1)
        layout.addWidget(QLabel("Max Frame Limit"), 3, 0)
        layout.addWidget(self.max_frame_latency_limit_spin, 3, 1)
        layout.addWidget(QLabel("w Avg E2E"), 4, 0)
        layout.addWidget(self.weight_avg_latency_slider, 4, 1)
        layout.addWidget(self.weight_avg_latency_label, 4, 2)
        layout.addWidget(QLabel("w Max E2E"), 5, 0)
        layout.addWidget(self.weight_max_latency_slider, 5, 1)
        layout.addWidget(self.weight_max_latency_label, 5, 2)
        layout.addWidget(QLabel("w Device Util"), 6, 0)
        layout.addWidget(self.weight_device_utilization_slider, 6, 1)
        layout.addWidget(self.weight_device_utilization_label, 6, 2)
        weight_hint = QLabel("Loss = w_avg * L_avg + w_max * L_max + w_util * L_util")
        weight_hint.setStyleSheet("color: #6b7280;")
        layout.addWidget(weight_hint, 7, 1, 1, 2)
        layout.addWidget(QLabel("Network"), 8, 0)
        layout.addWidget(self.batch_transfers_check, 8, 1, 1, 2)
        layout.addWidget(QLabel("Pipeline Unroll"), 9, 0)
        layout.addWidget(self.pipeline_unroll_spin, 9, 1, 1, 2)
        layout.addWidget(QLabel("Solver"), 10, 0)
        layout.addWidget(self.algorithm_combo, 10, 1, 1, 2)
        layout.addWidget(QLabel("Iterations"), 11, 0)
        layout.addWidget(self.iterations_spin, 11, 1, 1, 2)
        layout.addWidget(QLabel("Threads"), 12, 0)
        layout.addWidget(self.solver_threads_spin, 12, 1, 1, 2)
        self.anneal_initial_temp_label = QLabel("SA Initial Temp")
        self.anneal_final_temp_label = QLabel("SA Final Temp")
        layout.addWidget(self.anneal_initial_temp_label, 13, 0)
        layout.addWidget(self.anneal_initial_temp_spin, 13, 1, 1, 2)
        layout.addWidget(self.anneal_final_temp_label, 14, 0)
        layout.addWidget(self.anneal_final_temp_spin, 14, 1, 1, 2)
        self._update_sa_controls_enabled(self.algorithm_combo.currentText())
        return group

    def _update_sa_controls_enabled(self, algorithm: str) -> None:
        enabled = algorithm == "Simulated Annealing"
        for widget in (
            self.anneal_initial_temp_label,
            self.anneal_initial_temp_spin,
            self.anneal_final_temp_label,
            self.anneal_final_temp_spin,
        ):
            widget.setEnabled(enabled)

    def _objective_weight_slider(self, default: float) -> Tuple[QSlider, QLabel]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setTickInterval(25)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        label = QLabel()
        label.setMinimumWidth(44)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider.valueChanged.connect(lambda value: label.setText(f"{value / 100.0:.2f}"))
        slider.setValue(round(default * 100))
        label.setText(f"{slider.value() / 100.0:.2f}")
        return slider, label

    def _build_actions_group(self) -> QGroupBox:
        group = QGroupBox("操作 (Actions)")
        group.setMinimumHeight(72)
        layout = QHBoxLayout(group)
        load_button = QPushButton("📂 加载配置 (Load)")
        save_button = QPushButton("💾 保存配置 (Save)")
        solve_button = QPushButton("🚀 求解切分预案 (Solve)")
        load_button.clicked.connect(self._load_config)
        save_button.clicked.connect(self._save_config)
        solve_button.clicked.connect(self._solve)
        layout.addWidget(load_button)
        layout.addWidget(save_button)
        layout.addWidget(solve_button)
        return group

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        view_controls = QHBoxLayout()
        view_controls.addWidget(QLabel("看板布局"))
        self.view_layout_combo = QComboBox()
        self.view_layout_combo.addItems(["标签页", "三面板"])
        view_controls.addWidget(self.view_layout_combo)
        view_controls.addStretch(1)
        layout.addLayout(view_controls)

        self.config_graph_view = QWebEngineView()
        self.solved_graph_view = QWebEngineView()
        self.timeline_view = QWebEngineView()

        self.view_tabs = QTabWidget()
        self.view_tabs.setDocumentMode(True)
        self.view_tabs.setMovable(True)
        self.view_tabs.addTab(self.config_graph_view, "配置拓扑 (Config DAG)")
        self.view_tabs.addTab(self.solved_graph_view, "求解展开拓扑 (Solved Pipeline)")
        self.view_tabs.addTab(self.timeline_view, "时序图 (Solved Schedule)")

        self.view_splitter = QSplitter(Qt.Orientation.Vertical)
        self.view_panel_layouts: List[QVBoxLayout] = []
        for title in (
            "配置拓扑 (Config DAG)",
            "求解展开拓扑 (Solved Pipeline)",
            "时序图 (Solved Schedule)",
        ):
            group = QGroupBox(title)
            group.setMinimumHeight(150)
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(2, 2, 2, 2)
            self.view_panel_layouts.append(group_layout)
            self.view_splitter.addWidget(group)
        self.view_splitter.setStretchFactor(0, 1)
        self.view_splitter.setStretchFactor(1, 1)
        self.view_splitter.setStretchFactor(2, 2)
        self.view_splitter.hide()
        self.view_layout_combo.currentTextChanged.connect(self._set_view_layout)

        metrics_group = QGroupBox("性能看板")
        metrics_group.setMinimumHeight(108)
        metrics_group.setMaximumHeight(168)
        metrics_layout = QGridLayout(metrics_group)
        self.latency_label = QLabel("端到端时延: -- ms")
        self.latency_label.setWordWrap(True)
        self.util_label = QLabel("利用率: 端侧 -- %, 边侧 -- %, 网络 -- %")
        self.util_label.setWordWrap(True)
        self.loss_label = QLabel("最优 Loss: --")
        self.loss_terms_label = QLabel("Loss terms: --")
        self.loss_terms_label.setWordWrap(True)
        self.mode_label = QLabel("求解模式: --")
        self.mode_label.setWordWrap(True)
        metrics_layout.addWidget(self.latency_label, 0, 0)
        metrics_layout.addWidget(self.util_label, 0, 1)
        metrics_layout.addWidget(self.loss_label, 1, 0)
        metrics_layout.addWidget(self.mode_label, 1, 1)
        metrics_layout.addWidget(self.loss_terms_label, 2, 0, 1, 2)

        layout.addWidget(self.view_tabs, stretch=1)
        layout.addWidget(self.view_splitter, stretch=1)
        layout.addWidget(metrics_group, stretch=0)
        self._clear_timeline()
        return panel

    def _set_view_layout(self, mode: str) -> None:
        views = (self.config_graph_view, self.solved_graph_view, self.timeline_view)
        titles = (
            "配置拓扑 (Config DAG)",
            "求解展开拓扑 (Solved Pipeline)",
            "时序图 (Solved Schedule)",
        )
        if mode == "三面板":
            while self.view_tabs.count():
                self.view_tabs.removeTab(0)
            for panel_layout, view in zip(self.view_panel_layouts, views):
                panel_layout.addWidget(view)
                # QTabWidget keeps non-current pages explicitly hidden after
                # removeTab(). Reparenting alone does not clear that state.
                view.show()
            self.view_tabs.hide()
            self.view_splitter.show()
            return

        for panel_layout, view in zip(self.view_panel_layouts, views):
            panel_layout.removeWidget(view)
        for view, title in zip(views, titles):
            self.view_tabs.addTab(view, title)
        self.view_tabs.setCurrentIndex(0)
        self.view_splitter.hide()
        self.view_tabs.show()

    def _seed_example(self) -> None:
        self._set_table_data(
            [
                {
                    "id": "v1",
                    "name": "Sensor Pre",
                    "c_dev": 15.5,
                    "c_host": 2.1,
                    "source_period_ms": 33.333,
                    "source_phase_ms": 0.0,
                    "fixed_dev": True,
                },
                {
                    "id": "v2",
                    "name": "Backbone",
                    "c_dev": 40.0,
                    "c_host": 8.5,
                    "source_period_ms": 0.0,
                    "source_phase_ms": 0.0,
                    "fixed_dev": False,
                },
                {
                    "id": "v3",
                    "name": "Control Head",
                    "c_dev": 22.0,
                    "c_host": 4.0,
                    "source_period_ms": 0.0,
                    "source_phase_ms": 0.0,
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
                    float(node.get("source_period_ms", 0.0)),
                    float(node.get("source_phase_ms", 0.0)),
                    str(
                        node.get(
                            "placement",
                            "device" if node.get("fixed_dev", False) else "free",
                        )
                    ),
                )

            self.edges_table.setRowCount(0)
            for edge in edges:
                self._add_edge_row(str(edge["source"]), str(edge["target"]), float(edge["size"]))

            self.bandwidth_spin.setValue(environment.bandwidth)
            self.latency_spin.setValue(environment.latency)
            self.latency_limit_spin.setValue(environment.latency_limit)
            self.max_frame_latency_limit_spin.setValue(environment.max_frame_latency_limit)
            self._set_objective_slider_value(
                self.weight_avg_latency_slider,
                environment.weight_avg_latency,
            )
            self._set_objective_slider_value(
                self.weight_max_latency_slider,
                environment.weight_max_latency,
            )
            self._set_objective_slider_value(
                self.weight_device_utilization_slider,
                environment.weight_device_utilization,
            )
            self.batch_transfers_check.setChecked(environment.batch_transfers)
            self.pipeline_unroll_spin.setValue(environment.pipeline_unroll)
            self.solver_threads_spin.setValue(environment.solver_threads)
            self.anneal_initial_temp_spin.setValue(environment.anneal_initial_temp)
            self.anneal_final_temp_spin.setValue(environment.anneal_final_temp)
            self._update_sa_controls_enabled(self.algorithm_combo.currentText())
        finally:
            self._loading_tables = False

    def _add_node_row(
        self,
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        c_dev: float = 10.0,
        c_host: float = 2.0,
        source_period_ms: float = 0.0,
        source_phase_ms: float = 0.0,
        placement: str = "free",
    ) -> None:
        row = self.nodes_table.rowCount()
        self.nodes_table.insertRow(row)
        next_id = node_id or f"v{row + 1}"
        self.nodes_table.setItem(row, 0, QTableWidgetItem(next_id))
        self.nodes_table.setItem(row, 1, QTableWidgetItem(name or next_id))
        self.nodes_table.setItem(row, 2, QTableWidgetItem(f"{c_dev:g}"))
        self.nodes_table.setItem(row, 3, QTableWidgetItem(f"{c_host:g}"))
        self.nodes_table.setItem(row, 4, QTableWidgetItem(f"{source_period_ms:g}"))
        self.nodes_table.setItem(row, 5, QTableWidgetItem(f"{source_phase_ms:g}"))
        placement_combo = QComboBox()
        placement_combo.addItems(["Free", "Device", "Host"])
        placement_labels = {"free": "Free", "device": "Device", "host": "Host"}
        placement_combo.setCurrentText(
            placement_labels.get(placement.strip().lower(), "Free")
        )
        placement_combo.currentTextChanged.connect(self._preview_from_tables)
        self.nodes_table.setCellWidget(row, 6, placement_combo)
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
        self._clear_solved_graph()
        self._draw_config_graph()

    def _save_config(self) -> None:
        try:
            graph = self._graph_from_tables()
        except ValueError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._preserve_known_assignments(graph)

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
            node
            for node, attrs in graph.nodes(data=True)
            if str(
                attrs.get(
                    "placement",
                    "device" if attrs.get("fixed_dev", False) else "free",
                )
            ).strip().lower() == "free"
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
                weight_avg_latency=environment.weight_avg_latency,
                weight_max_latency=environment.weight_max_latency,
                weight_device_utilization=environment.weight_device_utilization,
                algorithm=self.algorithm_combo.currentText(),
                heuristic_iterations=self.iterations_spin.value(),
                latency_limit=environment.latency_limit,
                batch_transfers=environment.batch_transfers,
                pipeline_unroll=environment.pipeline_unroll,
                max_frame_latency_limit=environment.max_frame_latency_limit,
                solver_threads=environment.solver_threads,
                anneal_initial_temp=environment.anneal_initial_temp,
                anneal_final_temp=environment.anneal_final_temp,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Solve failed", str(exc))
            return

        for node, value in result.assignment.items():
            graph.nodes[node]["x"] = int(value)
        self.graph = graph
        self.solver_result = result
        self._update_metrics(result)
        self._draw_config_graph()
        self._draw_solved_graph()
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
        self._clear_solved_graph()
        self._draw_config_graph()

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
                    "source_period_ms": self._cell_float(
                        self.nodes_table, row, 4, "Source Period (ms)"
                    ),
                    "source_phase_ms": self._cell_float(
                        self.nodes_table, row, 5, "Source Phase (ms)"
                    ),
                    "placement": (
                        self.nodes_table.cellWidget(row, 6).currentText().strip().lower()
                        if isinstance(self.nodes_table.cellWidget(row, 6), QComboBox)
                        else "free"
                    ),
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
        return validate_environment(
            Environment(
                bandwidth=self.bandwidth_spin.value(),
                latency=self.latency_spin.value(),
                weight_avg_latency=self._objective_slider_value(
                    self.weight_avg_latency_slider
                ),
                weight_max_latency=self._objective_slider_value(
                    self.weight_max_latency_slider
                ),
                weight_device_utilization=self._objective_slider_value(
                    self.weight_device_utilization_slider
                ),
                latency_limit=self.latency_limit_spin.value(),
                batch_transfers=self.batch_transfers_check.isChecked(),
                pipeline_unroll=self.pipeline_unroll_spin.value(),
                max_frame_latency_limit=self.max_frame_latency_limit_spin.value(),
                solver_threads=self.solver_threads_spin.value(),
                anneal_initial_temp=self.anneal_initial_temp_spin.value(),
                anneal_final_temp=self.anneal_final_temp_spin.value(),
            )
        )

    def _objective_slider_value(self, slider: QSlider) -> float:
        return slider.value() / 100.0

    def _set_objective_slider_value(self, slider: QSlider, value: float) -> None:
        slider.setValue(round(max(0.0, min(1.0, value)) * 100))

    def _preserve_known_assignments(self, graph: nx.DiGraph) -> None:
        for node, attrs in graph.nodes(data=True):
            placement = str(
                attrs.get(
                    "placement",
                    "device" if attrs.get("fixed_dev", False) else "free",
                )
            ).strip().lower()
            if placement == "device":
                attrs["x"] = 0
            elif placement == "host":
                attrs["x"] = 1
            elif self.solver_result is not None and node in self.solver_result.assignment:
                attrs["x"] = int(self.solver_result.assignment[node])
            elif node in self.graph and "x" in self.graph.nodes[node]:
                attrs["x"] = int(self.graph.nodes[node]["x"])

    def _op_id(self, node: str, frame: int, unroll: int) -> str:
        return node if unroll == 1 else f"{node}[f{frame}]"

    def _base_node_id(self, op_id: str) -> str:
        return op_id.split("[f", 1)[0] if "[f" in op_id else op_id

    def _frame_index(self, op_id: str) -> int:
        if "[f" not in op_id:
            return 0
        suffix = op_id.rsplit("[f", 1)[1].rstrip("]")
        try:
            return int(suffix)
        except ValueError:
            return 0

    def _source_release_time(self, node: str, frame: int) -> float:
        if self.graph.in_degree(node) != 0:
            return 0.0
        attrs = self.graph.nodes[node]
        period_ms = max(0.0, float(attrs.get("source_period_ms", 0.0)))
        phase_ms = max(0.0, float(attrs.get("source_phase_ms", 0.0)))
        return phase_ms + frame * period_ms

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
                "source_period_ms": attrs.get("source_period_ms", 0.0),
                "source_phase_ms": attrs.get("source_phase_ms", 0.0),
                "placement": attrs.get(
                    "placement",
                    "device" if attrs.get("fixed_dev", False) else "free",
                ),
            }
            for node, attrs in graph.nodes(data=True)
        ]
        edges = [
            {"source": source, "target": target, "size": attrs.get("size", 0.0)}
            for source, target, attrs in graph.edges(data=True)
        ]
        return nodes, edges

    def _draw_config_graph(self) -> None:
        if self.graph.number_of_nodes() == 0:
            self.config_graph_view.setHtml(self._placeholder_html("No graph to display"))
            return
        self.config_graph_view.setHtml(
            self._graph_svg_html(self.graph.copy(), solved=False, title="Config DAG")
        )

    def _clear_solved_graph(self) -> None:
        self.solved_graph_view.setHtml(self._placeholder_html("Click Solve to render unrolled pipeline"))

    def _draw_solved_graph(self) -> None:
        if self.solver_result is None:
            self._clear_solved_graph()
            return
        self.solved_graph_view.setHtml(
            self._graph_svg_html(
                self._unrolled_solution_graph(),
                solved=True,
                title="Solved Pipeline",
                extra_info="<span>Fill = frame</span><span>Blue outline = Device</span><span>Green outline = Host</span>",
            )
        )

    def _graph_svg_html(
        self,
        graph: nx.DiGraph,
        solved: bool,
        title: str,
        extra_info: str = "",
    ) -> str:
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
        frame_markers = "".join(
            f'<marker id="arrow-frame-{index}" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
            f'<path d="M0,0 L10,4 L0,8 z" fill="{self._frame_color(index)}"/></marker>'
            for index in range(8)
        )
        rows = [
            '<defs><marker id="arrow-gray" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#6b7280"/></marker>',
            '<marker id="arrow-red" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#dc2626"/></marker>',
            f'<marker id="arrow-purple" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#7c3aed"/></marker>{frame_markers}</defs>',
        ]
        for source, target, attrs in graph.edges(data=True):
            sx, sy = pos[source]
            tx, ty = pos[target]
            size_mb = float(attrs.get("size", 0.0))
            frame_index = int(graph.nodes[source].get("frame", 0))
            frame_color = self._frame_color(frame_index)
            frame_marker = f"arrow-frame-{frame_index % 8}"
            if attrs.get("kind") == "fifo":
                color = frame_color
                dash = ' stroke-dasharray="3 5"'
                marker = frame_marker
                label = "FIFO"
            elif attrs.get("kind") == "source_period":
                color = frame_color
                dash = ' stroke-dasharray="6 4"'
                marker = ""
                label = f"period {float(attrs.get('period', 0.0)):g} ms"
            elif not solved:
                color = "#6b7280"
                dash = ""
                marker = "arrow-gray"
                label = f"{size_mb:g} MB"
            elif int(graph.nodes[source].get("x", 0)) == int(graph.nodes[target].get("x", 0)):
                color = frame_color
                dash = ""
                marker = frame_marker
                label = "local / 0 ms"
            else:
                color = frame_color
                dash = ' stroke-dasharray="7 5"'
                marker = frame_marker
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
            marker_attr = f' marker-end="url(#{marker})"' if marker else ""
            rows.append(
                f'<line class="edge" x1="{start_x:.1f}" y1="{start_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
                f'stroke="{color}"{dash}{marker_attr}><title>{html.escape(str(source))} → {html.escape(str(target))}: {html.escape(label)}</title></line>'
                f'<text class="edge-label" x="{mid_x:.1f}" y="{mid_y:.1f}">{html.escape(label)}</text>'
            )
        for node, attrs in graph.nodes(data=True):
            x_pos, y_pos = pos[node]
            x_value = int(attrs.get("x", 0))
            is_source = bool(attrs.get("is_source", False))
            fill = "#f8fafc"
            stroke = "#111827"
            if solved:
                fill = self._frame_color(int(attrs.get("frame", 0)))
                stroke = "#2563eb" if x_value == 0 else "#16a34a"
            if not solved:
                fill = "#f8fafc"
            place = "Dev" if x_value == 0 else "Host"
            compute = float(attrs["c_dev"]) if x_value == 0 else float(attrs["c_host"])
            placement = str(
                attrs.get(
                    "placement",
                    "device" if attrs.get("fixed_dev", False) else "free",
                )
            ).strip().lower()
            stroke_width = 4 if placement != "free" else 1.5
            if solved:
                stroke_width = 4
            dash = ' stroke-dasharray="6 4"' if solved and is_source else ""
            display_id = str(attrs.get("display_id", node))
            node_name = str(attrs.get("name", node))
            source_period = float(attrs.get("source_period_ms", 0.0))
            source_phase = float(attrs.get("source_phase_ms", 0.0))
            source_text = ""
            if source_period > 0 or source_phase > 0:
                source_text = f" / src period:{source_period:g} ms phase:{source_phase:g} ms"
            if solved and is_source:
                meta = f"source event: {float(attrs.get('release_ms', 0.0)):g} ms{source_text}"
            elif solved:
                meta = f"{place}: {compute:g} ms{source_text}"
            else:
                meta = f"D:{float(attrs['c_dev']):g} / H:{float(attrs['c_host']):g} ms{source_text}"
            id_fill = "#111827"
            rows.append(
                f'<g><title>{html.escape(str(node))} {html.escape(node_name)} - {html.escape(meta)}</title>'
                f'<text class="node-name" x="{x_pos:.1f}" y="{y_pos - 43:.1f}">{html.escape(node_name)}</text>'
                f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="24" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash} />'
                f'<text class="node-id" style="fill:{id_fill}" x="{x_pos:.1f}" y="{y_pos + 5:.1f}">{html.escape(display_id)}</text>'
                f'<text class="compute" x="{x_pos:.1f}" y="{y_pos + 48:.1f}">{html.escape(meta)}</text></g>'
            )
        return self._svg_page(
            title,
            "\n".join(rows),
            width,
            height,
            extra_info=extra_info,
            scale_axis="xy",
        )

    def _frame_color(self, frame: int) -> str:
        palette = [
            "#93c5fd",
            "#fbbf24",
            "#86efac",
            "#f9a8d4",
            "#c4b5fd",
            "#5eead4",
            "#fca5a5",
            "#7dd3fc",
        ]
        return palette[frame % len(palette)]

    def _unrolled_solution_graph(self) -> nx.DiGraph:
        assert self.solver_result is not None
        result = self.solver_result
        unroll = max(1, result.metrics.pipeline_unroll)
        base_layers = {
            node: layer
            for layer, generation in enumerate(nx.topological_generations(self.graph))
            for node in generation
        }
        graph = nx.DiGraph()
        for frame in range(unroll):
            for node, attrs in self.graph.nodes(data=True):
                op_id = self._op_id(str(node), frame, unroll)
                graph.add_node(
                    op_id,
                    name=f'{attrs.get("name", node)} f{frame}' if unroll > 1 else attrs.get("name", node),
                    display_id=f"{node}/f{frame}" if unroll > 1 else str(node),
                    base_node=str(node),
                    frame=frame,
                    layer=base_layers[node] + frame * (len(set(base_layers.values())) + 1),
                    c_dev=attrs["c_dev"],
                    c_host=attrs["c_host"],
                    source_period_ms=attrs.get("source_period_ms", 0.0),
                    source_phase_ms=attrs.get("source_phase_ms", 0.0),
                    release_ms=self._source_release_time(str(node), frame),
                    is_source=self.graph.in_degree(node) == 0,
                    fixed_dev=attrs.get("fixed_dev", False),
                    x=result.assignment[node],
                )
            for source, target, attrs in self.graph.edges(data=True):
                graph.add_edge(
                    self._op_id(str(source), frame, unroll),
                    self._op_id(str(target), frame, unroll),
                    size=float(attrs.get("size", 0.0)),
                    kind="data",
                )
        if unroll > 1:
            for frame in range(1, unroll):
                for node in self.graph.nodes:
                    if self.graph.in_degree(node) == 0:
                        graph.add_edge(
                            self._op_id(str(node), frame - 1, unroll),
                            self._op_id(str(node), frame, unroll),
                            size=0.0,
                            kind="source_period",
                            period=float(self.graph.nodes[node].get("source_period_ms", 0.0)),
                        )
                        continue
                    graph.add_edge(
                        self._op_id(str(node), frame - 1, unroll),
                        self._op_id(str(node), frame, unroll),
                        size=0.0,
                        kind="fifo",
                    )
        return graph

    def _layered_layout(self, graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        if all("layer" in attrs for _, attrs in graph.nodes(data=True)):
            by_layer: Dict[int, List[str]] = {}
            for node, attrs in graph.nodes(data=True):
                by_layer.setdefault(int(attrs["layer"]), []).append(node)
            generations = [by_layer[layer] for layer in sorted(by_layer)]
        else:
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
        if result.metrics.pipeline_unroll > 1:
            self.latency_label.setText(
                f"平均时延: {result.metrics.latency:.1f} ms/frame; "
                f"启动间隔: {result.metrics.initiation_interval:.1f} ms; "
                f"最大E2E: {result.metrics.max_frame_latency:.1f} ms "
                f"(unroll {result.metrics.pipeline_unroll})"
            )
        else:
            self.latency_label.setText(f"端到端时延: {result.metrics.latency:.1f} ms")
        self.util_label.setText(
            f"利用率: 端侧 {result.metrics.device_utilization * 100.0:.1f} %, "
            f"边侧 {result.metrics.host_utilization * 100.0:.1f} %, "
            f"网络 {result.metrics.network_utilization * 100.0:.1f} %"
        )
        self.loss_label.setText(f"最优 Loss: {result.metrics.loss:.3f}")
        self.loss_terms_label.setText(
            "Loss terms: "
            f"L_avg={result.metrics.avg_latency_loss:.3f}, "
            f"L_max={result.metrics.max_frame_latency_loss:.3f}, "
            f"L_ii={result.metrics.initiation_interval_loss:.3f}, "
            f"L_util={result.metrics.device_utilization_loss:.3f}"
        )
        self.mode_label.setText(f"求解模式: {result.mode} ({result.iterations})")

    def _clear_metrics(self) -> None:
        self.latency_label.setText("端到端时延: -- ms")
        self.util_label.setText("利用率: 端侧 -- %, 边侧 -- %, 网络 -- %")
        self.loss_label.setText("最优 Loss: --")
        self.loss_terms_label.setText("Loss terms: --")
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

        source_nodes = [str(node) for node in self.graph.nodes if self.graph.in_degree(node) == 0]
        max_release_ms = max(
            (
                self._source_release_time(node, frame)
                for node in source_nodes
                for frame in range(max(1, metrics.pipeline_unroll))
            ),
            default=0.0,
        )
        total_ms = max(
            1.0,
            max(metrics.finish_times.values()) if metrics.finish_times else 1.0,
            max_release_ms,
        )
        px_per_ms = 9.0
        left_pad = 120.0
        right_pad = 80.0
        top_pad = 42.0
        lane_height = 58.0
        input_lane_height = max(lane_height, 28.0 * max(1, len(source_nodes)))
        input_top = top_pad
        lane_y = {
            "input": input_top + input_lane_height / 2.0,
            1: input_top + input_lane_height + lane_height,
            "transfer": input_top + input_lane_height + lane_height * 2,
            0: input_top + input_lane_height + lane_height * 3,
        }
        source_lane_y = {
            node: input_top + 22.0 + index * 28.0
            for index, node in enumerate(sorted(source_nodes))
        }
        width = int(left_pad + total_ms * px_per_ms + right_pad)
        height = int(input_top + input_lane_height + lane_height * 3 + 50)

        def esc(value: object) -> str:
            return html.escape(str(value), quote=True)

        def x_at(ms: float) -> float:
            return left_pad + ms * px_per_ms

        rows = []
        for label, y_pos in [
            ("Input", lane_y["input"]),
            ("Host", lane_y[1]),
            ("Transfer", lane_y["transfer"]),
            ("Device", lane_y[0]),
        ]:
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

        for node in source_nodes:
            period = float(self.graph.nodes[node].get("source_period_ms", 0.0))
            y_center = source_lane_y[node]
            node_name = esc(self.graph.nodes[node].get("name", node))
            rows.append(
                f'<text class="source-lane-label" x="{left_pad - 10:.1f}" y="{y_center + 4:.1f}">{node_name}</text>'
            )
            releases = [
                self._source_release_time(node, frame)
                for frame in range(max(1, metrics.pipeline_unroll))
            ]
            if period > 0:
                for frame in range(1, len(releases)):
                    start = releases[frame - 1]
                    finish = releases[frame]
                    x_pos = x_at(start)
                    span = max(6.0, (finish - start) * px_per_ms)
                    fill = self._frame_color(frame - 1)
                    rows.append(
                        f'<rect class="source-span" data-start="{start:.6f}" data-duration="{finish - start:.6f}" '
                        f'x="{x_pos:.1f}" y="{y_center - 8:.1f}" width="{span:.1f}" height="16" rx="3" '
                        f'style="fill:{fill};stroke:{fill}">'
                        f'<title>{node_name}: period span {start:.1f}-{finish:.1f} ms</title></rect>'
                    )
            for frame in range(max(1, metrics.pipeline_unroll)):
                release = releases[frame]
                x_pos = x_at(release)
                label = f"{node} f{frame}" if metrics.pipeline_unroll > 1 else node
                fill = self._frame_color(frame)
                placement_y = lane_y[int(assignment[node])]
                rows.append(
                    f'<g><title>{esc(label)} source release at {release:.1f} ms; period {period:g} ms</title>'
                    f'<line class="dep release-dep" data-ms="{release:.6f}" x1="{x_pos:.1f}" y1="{y_center:.1f}" '
                    f'x2="{x_pos:.1f}" y2="{placement_y:.1f}" style="stroke:{fill}" />'
                    f'<line class="release" data-ms="{release:.6f}" x1="{x_pos:.1f}" y1="{y_center - 13:.1f}" '
                    f'x2="{x_pos:.1f}" y2="{y_center + 13:.1f}" style="stroke:{fill}" />'
                    f'<circle class="release-dot" data-ms="{release:.6f}" cx="{x_pos:.1f}" cy="{y_center:.1f}" r="5" '
                    f'style="fill:{fill};stroke:{fill}" />'
                    f'<text class="release-label" data-ms="{release:.6f}" x="{x_pos:.1f}" y="{y_center - 16:.1f}">f{frame}</text></g>'
                )

        for node in sorted(metrics.start_times, key=lambda n: metrics.start_times[n]):
            base_node = self._base_node_id(node)
            if base_node in source_nodes:
                continue
            start = metrics.start_times[node]
            finish = metrics.finish_times[node]
            duration = max(0.1, finish - start)
            x_value = int(assignment[base_node])
            y_center = lane_y[x_value]
            x_pos = x_at(start)
            bar_width = max(6.0, duration * px_per_ms)
            fill = self._frame_color(self._frame_index(node))
            stroke = "#2563eb" if x_value == 0 else "#16a34a"
            node_name = esc(self.graph.nodes[base_node].get("name", base_node))
            short_label = esc(_timeline_short_node_label(node))
            rows.append(
                f'<g><title>{esc(node)} {node_name}: {start:.1f}-{finish:.1f} ms</title>'
                f'<rect class="op" data-start="{start:.6f}" data-duration="{duration:.6f}" x="{x_pos:.1f}" y="{y_center - 16:.1f}" width="{bar_width:.1f}" '
                f'height="32" rx="4" fill="{fill}" style="stroke:{stroke};stroke-width:2" />'
                f'<text class="op-id op-id-dark timeline-label" data-start="{start:.6f}" data-duration="{duration:.6f}" data-anchor="center" x="{x_pos + bar_width / 2:.1f}" y="{y_center + 5:.1f}">{short_label}</text></g>'
            )

        for transfer_index, transfer in enumerate(metrics.transfer_records):
            start = transfer.start
            finish = transfer.finish
            transfer_ms = finish - start
            frame = self._frame_index(transfer.edges[0][0]) if transfer.edges else 0
            fill = self._frame_color(frame)
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
                label = f"x{len(transfer.edges)} / {transfer.size_mb:g} MB"
            else:
                source, target = transfer.edges[0]
                title = (
                    f"{source} -> {target}: {transfer.size_mb:g} MB, "
                    f"{transfer_ms:.1f} ms"
                )
                label = f"{transfer.size_mb:g} MB"
            rows.append(
                f'<g><title>{esc(title)}</title>'
                f'<rect class="transfer" data-start="{start:.6f}" data-duration="{transfer_ms:.6f}" x="{x_pos:.1f}" y="{y_center - 10:.1f}" width="{bar_width:.1f}" '
                f'height="20" rx="3" style="fill:{fill};stroke:{fill}" />'
                f'<text class="transfer-label timeline-label" data-start="{start:.6f}" data-duration="{transfer_ms:.6f}" data-anchor="center" x="{x_pos + bar_width / 2:.1f}" y="{y_center + 4:.1f}">'
                f'{esc(label)}</text>'
            )
            for source, target in transfer.edges:
                source_y = lane_y[int(assignment[self._base_node_id(source)])]
                target_y = lane_y[int(assignment[self._base_node_id(target)])]
                rows.append(
                    f'<line class="dep" data-ms="{start:.6f}" x1="{x_pos:.1f}" y1="{source_y:.1f}" '
                    f'x2="{x_pos:.1f}" y2="{y_center:.1f}" style="stroke:{fill}" />'
                    f'<line class="dep" data-ms="{finish:.6f}" x1="{end_x:.1f}" y1="{y_center:.1f}" '
                    f'x2="{end_x:.1f}" y2="{target_y:.1f}" style="stroke:{fill}" />'
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
                f"<span>Unroll: {environment.pipeline_unroll}</span>"
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
            timeline_setup = f"""
  const viewport = document.getElementById('viewport');
  const basePxPerMs = {timeline_px_per_ms};
  const leftPad = {timeline_left_pad};
  const rightPad = {timeline_right_pad};
  const totalMs = {timeline_total_ms};
  const pxPerMsInput = document.getElementById('pxPerMs');
  const zoomOut = document.getElementById('zoomOut');
  const zoomReset = document.getElementById('zoomReset');
  const zoomIn = document.getElementById('zoomIn');
  const zoomFit = document.getElementById('zoomFit');
"""
            scale_script = f"""
    const pxPerMs = basePxPerMs * factor;
    const scaledWidth = Math.max(viewport.clientWidth, leftPad + totalMs * pxPerMs + rightPad);
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
      }} else if (el.tagName === 'circle') {{
        el.setAttribute('cx', x);
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
    pxPerMsInput.value = pxPerMs.toFixed(2);
    updateTimelineLabels();
"""
            timeline_controls = """
    <button id="zoomOut" type="button" title="缩小">−</button>
    <button id="zoomReset" type="button">100%</button>
    <button id="zoomIn" type="button" title="放大">+</button>
    <button id="zoomFit" type="button">适应窗口</button>
    <label for="pxPerMs">px/ms</label>
    <input id="pxPerMs" type="number" min="0.01" max="10000" step="0.1">"""
            timeline_events = """
  function setScalePercent(percent) {
    const bounded = Math.max(Number(scale.min), Math.min(Number(scale.max), percent));
    scale.value = String(bounded);
    applyScale();
  }
  zoomOut.addEventListener('click', () => setScalePercent(Number(scale.value) / 1.25));
  zoomReset.addEventListener('click', () => setScalePercent(100));
  zoomIn.addEventListener('click', () => setScalePercent(Number(scale.value) * 1.25));
  zoomFit.addEventListener('click', () => {
    const available = Math.max(1, viewport.clientWidth - leftPad - rightPad);
    const fitted = totalMs > 0 ? 100 * available / (totalMs * basePxPerMs) : 100;
    setScalePercent(fitted);
    viewport.scrollLeft = 0;
  });
  pxPerMsInput.addEventListener('change', () => {
    const pxPerMs = Math.max(0.01, Number(pxPerMsInput.value) || basePxPerMs);
    setScalePercent(100 * pxPerMs / basePxPerMs);
  });
  viewport.addEventListener('wheel', (event) => {
    if (!event.ctrlKey) return;
    event.preventDefault();
    const rect = viewport.getBoundingClientRect();
    const cursorX = event.clientX - rect.left;
    const oldPxPerMs = basePxPerMs * Number(scale.value) / 100;
    const cursorTime = Math.max(0, (viewport.scrollLeft + cursorX - leftPad) / oldPxPerMs);
    const multiplier = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    setScalePercent(Number(scale.value) * multiplier);
    const newPxPerMs = basePxPerMs * Number(scale.value) / 100;
    viewport.scrollLeft = Math.max(0, leftPad + cursorTime * newPxPerMs - cursorX);
  }, { passive: false });
"""
            scale_max = 2000
        else:
            timeline_setup = ""
            timeline_controls = ""
            timeline_events = ""
            scale_max = 500
            scale_transform = (
                "'scale(' + factor + ')'" if scale_axis == "xy" else "'scaleX(' + factor + ')'"
            )
            scaled_height = f"({height} * factor)" if scale_axis == "xy" else str(height)
            scale_script = f"""
    content.style.transform = {scale_transform};
    content.style.width = ({width} * factor) + 'px';
    content.style.height = {scaled_height} + 'px';
"""
        label_controls = "" if scale_axis != "timeline" else """
    <label for="labelMode">标签</label>
    <select id="labelMode">
      <option value="auto" selected>精简</option>
      <option value="none">仅悬停</option>
      <option value="all">全部</option>
    </select>"""
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{ height: 100%; margin: 0; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  body {{ display: flex; flex-direction: column; background: #ffffff; color: #111827; }}
  .toolbar {{ flex: 0 0 auto; display: flex; align-items: center; gap: 10px; padding: 6px 10px; border-bottom: 1px solid #e5e7eb; background: #f9fafb; font-size: 12px; }}
  .toolbar input {{ width: 180px; }}
  .toolbar button {{ min-width: 32px; padding: 2px 8px; }}
  .toolbar #pxPerMs {{ width: 72px; }}
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
  .release {{ stroke: #be123c; stroke-width: 1.5; stroke-dasharray: 4 4; }}
  .release-dot {{ stroke-width: 1.5; }}
  .release-label {{ font-size: 10px; fill: #be123c; font-weight: 700; text-anchor: middle; paint-order: stroke; stroke: #ffffff; stroke-width: 3px; stroke-linejoin: round; }}
  .source-span {{ stroke: #be123c; stroke-width: 1.5; stroke-dasharray: 6 4; opacity: 0.55; }}
  .source-lane-label {{ fill: #7f1d1d; font-size: 10px; font-weight: 700; text-anchor: end; }}
  .op {{ stroke: #111827; stroke-width: 1; }}
  .op-id {{ fill: #ffffff; font-size: 11px; font-weight: 700; text-anchor: middle; pointer-events: none; }}
  .op-id-dark {{ fill: #111827; stroke: none; }}
  .transfer {{ fill: #f97316; stroke: #9a3412; stroke-width: 1; }}
  .transfer-label {{ fill: #111827; font-size: 10px; font-weight: 700; text-anchor: middle; pointer-events: none; }}
  .dep {{ stroke: #9a3412; stroke-width: 1; opacity: 0.75; }}
  .release-dep {{ stroke: #be123c; opacity: 0.65; }}
</style>
</head>
<body>
  <div class="toolbar">
    <strong>{html.escape(title)}</strong>
    <label for="scale">Scale</label>
    <input id="scale" type="range" min="5" max="{scale_max}" step="5" value="100">
    <span id="scaleText">100%</span>
    {timeline_controls}
    {label_controls}
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
  const labelMode = document.getElementById('labelMode');
{timeline_setup}
  function updateTimelineLabels() {{
    if (!labelMode) return;
    document.querySelectorAll('.timeline-label').forEach((label) => {{
      const rect = label.parentElement && label.parentElement.querySelector('rect');
      if (!rect) return;
      const barWidth = Number(rect.getAttribute('width') || 0);
      const textWidth = Math.max(30, (label.textContent || '').trim().length * 6.5 + 12);
      const visible = labelMode.value === 'all'
        || (labelMode.value === 'auto' && barWidth >= textWidth);
      label.style.display = visible ? '' : 'none';
    }});
  }}
  function applyScale() {{
    const factor = Number(scale.value) / 100;
    scaleText.textContent = scale.value + '%';
{scale_script}
  }}
  scale.addEventListener('input', applyScale);
  if (labelMode) labelMode.addEventListener('change', updateTimelineLabels);
{timeline_events}
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
