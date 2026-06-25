#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct Edge {
    int source;
    int target;
    double size;
};

struct TransferRecord {
    std::vector<std::pair<int, int>> edges;
    double start;
    double finish;
    double size;
    bool batched;
};

struct GraphData {
    std::vector<std::string> ids;
    std::vector<double> c_dev;
    std::vector<double> c_host;
    std::vector<bool> fixed_dev;
    std::vector<double> source_period_ms;
    std::vector<double> source_phase_ms;
    std::vector<int> x_initial;
    std::vector<Edge> edges;
    std::vector<std::vector<Edge>> outgoing;
    std::vector<std::vector<Edge>> incoming;
    std::vector<int> topo;
};

struct Metrics {
    double latency = 0.0;
    double max_frame_latency = 0.0;
    double utilization = 0.0;
    double avg_latency_loss = 0.0;
    double max_frame_latency_loss = 0.0;
    double device_utilization_loss = 0.0;
    double loss = 0.0;
    std::vector<double> starts;
    std::vector<double> finishes;
    std::vector<TransferRecord> transfers;
};

struct ObjectiveWeights {
    double avg_latency = 0.7;
    double max_latency = 0.3;
    double device_utilization = 0.3;
};

struct ObjectiveScales {
    double avg_latency = 1.0;
    double max_latency = 1.0;
};

double transfer_ms(double size_mb, double bandwidth_mb_s, double latency_ms) {
    if (!std::isfinite(size_mb)) {
        throw std::runtime_error("Transfer size must be finite.");
    }
    if (size_mb < 0.0) {
        throw std::runtime_error("Transfer size must be non-negative.");
    }
    if (!std::isfinite(bandwidth_mb_s)) {
        throw std::runtime_error("Bandwidth must be finite.");
    }
    if (bandwidth_mb_s <= 0.0) {
        throw std::runtime_error("Bandwidth must be greater than zero.");
    }
    if (!std::isfinite(latency_ms)) {
        throw std::runtime_error("Latency must be finite.");
    }
    if (latency_ms < 0.0) {
        throw std::runtime_error("Latency must be non-negative.");
    }
    return latency_ms + (size_mb / bandwidth_mb_s * 1000.0);
}

double non_negative(double value, const std::string& label) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(label + " must be finite.");
    }
    if (value < 0.0) {
        throw std::runtime_error(label + " must be non-negative.");
    }
    return value;
}

double normalized_time_score(double value, double scale) {
    double safe_scale = std::max(std::abs(scale), 1.0);
    return std::max(0.0, value) / safe_scale;
}

double non_negative_param(const py::dict& params, const char* key, double fallback) {
    if (!params.contains(key)) {
        return fallback;
    }
    return non_negative(params[key].cast<double>(), std::string("Environment ") + key);
}

ObjectiveWeights parse_objective_weights(const py::dict& params) {
    bool has_explicit_weights = params.contains("weight_avg_latency") ||
                                params.contains("weight_max_latency") ||
                                params.contains("weight_device_utilization");
    if (has_explicit_weights) {
        return ObjectiveWeights{
            non_negative_param(params, "weight_avg_latency", 0.7),
            non_negative_param(params, "weight_max_latency", 0.3),
            non_negative_param(params, "weight_device_utilization", 0.3),
        };
    }

    if (params.contains("weight_latency")) {
        double legacy = params["weight_latency"].cast<double>();
        if (!std::isfinite(legacy)) {
            throw std::runtime_error("Environment weight_latency must be finite.");
        }
        legacy = std::min(1.0, std::max(0.0, legacy));
        return ObjectiveWeights{legacy, 0.0, 1.0 - legacy};
    }

    return ObjectiveWeights{};
}

std::string op_id(const GraphData& graph, int op, int unroll) {
    int n = static_cast<int>(graph.ids.size());
    int frame = op / n;
    int node = op % n;
    if (unroll <= 1) {
        return graph.ids[node];
    }
    return graph.ids[node] + "[f" + std::to_string(frame) + "]";
}

double source_release_time(const GraphData& graph, int node, int frame) {
    if (!graph.incoming[node].empty()) {
        return 0.0;
    }
    double period = std::max(0.0, graph.source_period_ms[node]);
    double phase = std::max(0.0, graph.source_phase_ms[node]);
    return phase + static_cast<double>(frame) * period;
}

int frame_index(int op, int node_count) {
    return op / node_count;
}

std::vector<int> topo_sort_base(int node_count, const std::vector<Edge>& edges) {
    std::vector<int> indegree(node_count, 0);
    std::vector<std::vector<int>> outgoing(node_count);
    for (const auto& edge : edges) {
        outgoing[edge.source].push_back(edge.target);
        indegree[edge.target] += 1;
    }
    std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
    for (int node = 0; node < node_count; ++node) {
        if (indegree[node] == 0) {
            ready.push(node);
        }
    }
    std::vector<int> order;
    while (!ready.empty()) {
        int node = ready.top();
        ready.pop();
        order.push_back(node);
        for (int target : outgoing[node]) {
            indegree[target] -= 1;
            if (indegree[target] == 0) {
                ready.push(target);
            }
        }
    }
    if (static_cast<int>(order.size()) != node_count) {
        throw std::runtime_error("Graph contains a cycle; DAG required.");
    }
    return order;
}

GraphData parse_graph(const py::dict& data) {
    GraphData graph;
    graph.ids = data["ids"].cast<std::vector<std::string>>();
    graph.c_dev = data["c_dev"].cast<std::vector<double>>();
    graph.c_host = data["c_host"].cast<std::vector<double>>();
    graph.fixed_dev = data["fixed_dev"].cast<std::vector<bool>>();
    graph.source_period_ms = data["source_period_ms"].cast<std::vector<double>>();
    graph.source_phase_ms = data["source_phase_ms"].cast<std::vector<double>>();
    graph.x_initial = data["x_initial"].cast<std::vector<int>>();
    auto sources = data["edge_sources"].cast<std::vector<int>>();
    auto targets = data["edge_targets"].cast<std::vector<int>>();
    auto sizes = data["edge_sizes"].cast<std::vector<double>>();

    int n = static_cast<int>(graph.ids.size());
    auto require_size = [&](size_t size, const std::string& label) {
        if (size != static_cast<size_t>(n)) {
            throw std::runtime_error(label + " length does not match ids length.");
        }
    };
    require_size(graph.c_dev.size(), "c_dev");
    require_size(graph.c_host.size(), "c_host");
    require_size(graph.fixed_dev.size(), "fixed_dev");
    require_size(graph.source_period_ms.size(), "source_period_ms");
    require_size(graph.source_phase_ms.size(), "source_phase_ms");
    require_size(graph.x_initial.size(), "x_initial");
    graph.outgoing.assign(n, {});
    graph.incoming.assign(n, {});
    for (int node = 0; node < n; ++node) {
        std::string label = "Node " + graph.ids[node];
        graph.c_dev[node] = non_negative(graph.c_dev[node], label + " c_dev");
        graph.c_host[node] = non_negative(graph.c_host[node], label + " c_host");
        graph.source_period_ms[node] = non_negative(
            graph.source_period_ms[node], label + " source_period_ms"
        );
        graph.source_phase_ms[node] = non_negative(
            graph.source_phase_ms[node], label + " source_phase_ms"
        );
        if (graph.x_initial[node] != 0 && graph.x_initial[node] != 1) {
            throw std::runtime_error(label + " x must be 0 or 1.");
        }
    }
    if (sources.size() != targets.size() || sources.size() != sizes.size()) {
        throw std::runtime_error("Edge source, target, and size arrays must have equal length.");
    }
    for (size_t i = 0; i < sources.size(); ++i) {
        if (sources[i] < 0 || sources[i] >= n || targets[i] < 0 || targets[i] >= n) {
            throw std::runtime_error("Edge references an unknown node.");
        }
        Edge edge{sources[i], targets[i], non_negative(sizes[i], "Edge size")};
        graph.edges.push_back(edge);
        graph.outgoing[edge.source].push_back(edge);
        graph.incoming[edge.target].push_back(edge);
    }
    graph.topo = topo_sort_base(n, graph.edges);
    return graph;
}

enum class ResourceKind {
    None,
    Device,
    Host,
    Network,
};

enum class ScheduleRule {
    CriticalPath,
    DeviceFirst,
    FifoReady,
};

struct ScheduleTask {
    ResourceKind resource = ResourceKind::None;
    int op = -1;
    int frame = 0;
    int node = 0;
    double duration = 0.0;
    double release = 0.0;
    double size = 0.0;
    bool batched = false;
    std::vector<std::pair<int, int>> edges;
    std::vector<int> predecessors;
    std::vector<int> successors;
    int pending = 0;
};

void add_task_dependency(std::vector<ScheduleTask>& tasks, int source, int target) {
    auto& successors = tasks[source].successors;
    if (std::find(successors.begin(), successors.end(), target) != successors.end()) {
        return;
    }
    tasks[source].successors.push_back(target);
    tasks[target].predecessors.push_back(source);
    tasks[target].pending += 1;
}

std::vector<ScheduleTask> build_schedule_tasks(
    const GraphData& graph,
    const std::vector<int>& assignment,
    double bandwidth,
    double latency,
    bool batch_transfers,
    bool defer_blocked_transfers,
    bool gate_next_frame_work,
    int unroll
) {
    int n = static_cast<int>(graph.ids.size());
    int op_count = n * unroll;
    std::vector<ScheduleTask> tasks;
    tasks.reserve(op_count + static_cast<int>(graph.edges.size()) * unroll);

    for (int op = 0; op < op_count; ++op) {
        int frame = op / n;
        int node = op % n;
        bool is_source = graph.incoming[node].empty();
        ScheduleTask task;
        task.op = op;
        task.frame = frame;
        task.node = node;
        task.release = source_release_time(graph, node, frame);
        if (is_source) {
            task.resource = ResourceKind::None;
            task.duration = 0.0;
        } else if (assignment[node] == 0) {
            task.resource = ResourceKind::Device;
            task.duration = graph.c_dev[node];
        } else {
            task.resource = ResourceKind::Host;
            task.duration = graph.c_host[node];
        }
        tasks.push_back(std::move(task));
    }

    for (int frame = 0; frame < unroll; ++frame) {
        for (const auto& edge : graph.edges) {
            int source_op = frame * n + edge.source;
            int target_op = frame * n + edge.target;
            if (assignment[edge.source] == assignment[edge.target]) {
                add_task_dependency(tasks, source_op, target_op);
            }
        }
    }

    for (int frame = 1; frame < unroll; ++frame) {
        for (int node : graph.topo) {
            if (graph.incoming[node].empty()) {
                continue;
            }
            int source_op = (frame - 1) * n + node;
            int target_op = frame * n + node;
            add_task_dependency(tasks, source_op, target_op);
        }
    }

    for (int frame = 0; frame < unroll; ++frame) {
        for (int node = 0; node < n; ++node) {
            int source_op = frame * n + node;
            std::vector<std::pair<int, int>> outgoing_cross;
            double total_size = 0.0;
            for (const auto& edge : graph.outgoing[node]) {
                if (assignment[edge.target] == assignment[node]) {
                    continue;
                }
                int target_op = frame * n + edge.target;
                outgoing_cross.push_back({source_op, target_op});
                total_size += edge.size;
            }
            if (outgoing_cross.empty()) {
                continue;
            }

            auto add_transfer = [&](std::vector<std::pair<int, int>> edges, double size, bool batched) {
                ScheduleTask transfer;
                transfer.resource = ResourceKind::Network;
                transfer.op = source_op;
                transfer.frame = frame;
                transfer.node = node;
                transfer.duration = transfer_ms(size, bandwidth, latency);
                transfer.size = size;
                transfer.batched = batched;
                transfer.edges = std::move(edges);
                int transfer_id = static_cast<int>(tasks.size());
                tasks.push_back(std::move(transfer));
                add_task_dependency(tasks, source_op, transfer_id);
                if (defer_blocked_transfers && tasks[transfer_id].edges.size() == 1) {
                    int target = tasks[transfer_id].edges.front().second;
                    auto target_predecessors = tasks[target].predecessors;
                    for (int predecessor : target_predecessors) {
                        if (predecessor == source_op ||
                            tasks[predecessor].resource == ResourceKind::Network) {
                            continue;
                        }
                        add_task_dependency(tasks, predecessor, transfer_id);
                    }
                }
                for (auto [_, target] : tasks[transfer_id].edges) {
                    add_task_dependency(tasks, transfer_id, target);
                }
            };

            if (batch_transfers && outgoing_cross.size() > 1) {
                add_transfer(std::move(outgoing_cross), total_size, true);
            } else {
                for (const auto& edge : graph.outgoing[node]) {
                    if (assignment[edge.target] == assignment[node]) {
                        continue;
                    }
                    int target_op = frame * n + edge.target;
                    add_transfer({{source_op, target_op}}, edge.size, false);
                }
            }
        }
    }

    if (gate_next_frame_work && unroll > 1) {
        std::vector<int> output_nodes;
        for (int node = 0; node < n; ++node) {
            if (graph.outgoing[node].empty() && !graph.incoming[node].empty()) {
                output_nodes.push_back(node);
            }
        }
        if (output_nodes.empty()) {
            for (int node = 0; node < n; ++node) {
                if (!graph.incoming[node].empty()) {
                    output_nodes.push_back(node);
                }
            }
        }

        for (int frame = 1; frame < unroll; ++frame) {
            std::vector<int> previous_outputs;
            previous_outputs.reserve(output_nodes.size());
            for (int output_node : output_nodes) {
                previous_outputs.push_back((frame - 1) * n + output_node);
            }
            for (int task_id = 0; task_id < static_cast<int>(tasks.size()); ++task_id) {
                const auto& task = tasks[task_id];
                if (task.frame != frame || task.resource == ResourceKind::None) {
                    continue;
                }
                for (int previous_output : previous_outputs) {
                    if (previous_output != task_id) {
                        add_task_dependency(tasks, previous_output, task_id);
                    }
                }
            }
        }
    }

    return tasks;
}

std::vector<double> task_ranks(const std::vector<ScheduleTask>& tasks) {
    std::vector<double> ranks(tasks.size(), 0.0);
    std::vector<int> state(tasks.size(), 0);
    std::function<double(int)> visit = [&](int task_id) {
        if (state[task_id] == 2) {
            return ranks[task_id];
        }
        if (state[task_id] == 1) {
            throw std::runtime_error("Expanded pipeline graph contains a cycle.");
        }
        state[task_id] = 1;
        double downstream = 0.0;
        for (int successor : tasks[task_id].successors) {
            downstream = std::max(downstream, visit(successor));
        }
        ranks[task_id] = tasks[task_id].duration + downstream;
        state[task_id] = 2;
        return ranks[task_id];
    };
    for (int task_id = 0; task_id < static_cast<int>(tasks.size()); ++task_id) {
        visit(task_id);
    }
    return ranks;
}

double resource_ready_at(
    ResourceKind resource,
    double dev_ready,
    double host_ready,
    double network_ready
) {
    if (resource == ResourceKind::Device) {
        return dev_ready;
    }
    if (resource == ResourceKind::Host) {
        return host_ready;
    }
    if (resource == ResourceKind::Network) {
        return network_ready;
    }
    return 0.0;
}

void set_resource_ready(
    ResourceKind resource,
    double finish,
    double& dev_ready,
    double& host_ready,
    double& network_ready
) {
    if (resource == ResourceKind::Device) {
        dev_ready = finish;
    } else if (resource == ResourceKind::Host) {
        host_ready = finish;
    } else if (resource == ResourceKind::Network) {
        network_ready = finish;
    }
}

double task_priority(
    const ScheduleTask& task,
    const std::vector<double>& ranks,
    int task_id,
    int unroll,
    ScheduleRule rule
) {
    if (rule == ScheduleRule::CriticalPath) {
        double aging = 1'000'000.0 * static_cast<double>(std::max(0, unroll - 1 - task.frame));
        return ranks[task_id] + aging;
    }
    if (rule == ScheduleRule::DeviceFirst) {
        if (task.resource == ResourceKind::None) {
            return 4'000'000.0 - static_cast<double>(task.frame);
        }
        if (task.resource == ResourceKind::Device) {
            return 3'000'000.0 + task.duration + ranks[task_id] * 1e-6;
        }
        if (task.resource == ResourceKind::Network) {
            return 2'000'000.0 + ranks[task_id] * 1e-6;
        }
        return 1'000'000.0 + task.duration + ranks[task_id] * 1e-6;
    }
    return -static_cast<double>(task_id);
}

Metrics finalize_metrics(
    const GraphData& graph,
    const std::vector<int>& assignment,
    const ObjectiveWeights& weights,
    const ObjectiveScales& scales,
    int unroll,
    Metrics metrics
) {
    int n = static_cast<int>(graph.ids.size());
    int op_count = n * unroll;
    std::vector<double> frame_work_start(unroll, std::numeric_limits<double>::infinity());
    std::vector<bool> frame_has_work(unroll, false);
    for (int op = 0; op < op_count; ++op) {
        int frame = op / n;
        int node = op % n;
        if (graph.incoming[node].empty()) {
            continue;
        }
        frame_work_start[frame] = std::min(frame_work_start[frame], metrics.starts[op]);
        frame_has_work[frame] = true;
    }
    for (const auto& transfer : metrics.transfers) {
        if (transfer.edges.empty()) {
            continue;
        }
        int frame = frame_index(transfer.edges.front().first, n);
        if (frame >= 0 && frame < unroll) {
            frame_work_start[frame] = std::min(frame_work_start[frame], transfer.start);
            frame_has_work[frame] = true;
        }
    }

    double first_work_start = 0.0;
    bool has_pipeline_work = false;
    for (int frame = 0; frame < unroll; ++frame) {
        if (frame_has_work[frame]) {
            first_work_start = has_pipeline_work
                                   ? std::min(first_work_start, frame_work_start[frame])
                                   : frame_work_start[frame];
            has_pipeline_work = true;
        }
    }

    double pipeline_finish = first_work_start;
    for (int frame = 0; frame < unroll; ++frame) {
        if (!frame_has_work[frame]) {
            continue;
        }
        double frame_finish = frame_work_start[frame];
        bool has_output = false;
        for (int node = 0; node < n; ++node) {
            if (graph.outgoing[node].empty() && !graph.incoming[node].empty()) {
                int op = frame * n + node;
                frame_finish = has_output ? std::max(frame_finish, metrics.finishes[op])
                                          : metrics.finishes[op];
                has_output = true;
            }
        }
        if (!has_output) {
            for (int node = 0; node < n; ++node) {
                if (graph.incoming[node].empty()) {
                    continue;
                }
                int op = frame * n + node;
                frame_finish = std::max(frame_finish, metrics.finishes[op]);
            }
        }
        pipeline_finish = std::max(pipeline_finish, frame_finish);
        metrics.max_frame_latency = std::max(
            metrics.max_frame_latency,
            std::max(0.0, frame_finish - frame_work_start[frame])
        );
    }

    double pipeline_time = has_pipeline_work ? std::max(0.0, pipeline_finish - first_work_start)
                                             : 0.0;
    metrics.latency = pipeline_time / static_cast<double>(unroll);

    double dev_active = 0.0;
    for (int op = 0; op < op_count; ++op) {
        int node = op % n;
        if (graph.incoming[node].empty()) {
            continue;
        }
        if (assignment[node] == 0) {
            dev_active += metrics.finishes[op] - metrics.starts[op];
        }
    }
    metrics.utilization = pipeline_time <= 0.0 ? 0.0 : dev_active / pipeline_time;
    metrics.utilization = std::min(1.0, std::max(0.0, metrics.utilization));

    metrics.avg_latency_loss = normalized_time_score(metrics.latency, scales.avg_latency);
    metrics.max_frame_latency_loss = normalized_time_score(
        metrics.max_frame_latency,
        scales.max_latency
    );
    metrics.device_utilization_loss = 1.0 - metrics.utilization;
    metrics.loss = weights.avg_latency * metrics.avg_latency_loss +
                   weights.max_latency * metrics.max_frame_latency_loss +
                   weights.device_utilization * metrics.device_utilization_loss;
    return metrics;
}

Metrics simulate_schedule(
    const GraphData& graph,
    const std::vector<int>& assignment,
    const ObjectiveWeights& weights,
    const ObjectiveScales& scales,
    const std::vector<ScheduleTask>& tasks,
    const std::vector<double>& ranks,
    int unroll,
    ScheduleRule rule
) {
    int n = static_cast<int>(graph.ids.size());
    int op_count = n * unroll;
    std::vector<int> pending(tasks.size(), 0);
    std::vector<double> ready_time(tasks.size(), 0.0);
    std::vector<int> ready;
    ready.reserve(tasks.size());
    for (int task_id = 0; task_id < static_cast<int>(tasks.size()); ++task_id) {
        pending[task_id] = tasks[task_id].pending;
        ready_time[task_id] = tasks[task_id].release;
        if (pending[task_id] == 0) {
            ready.push_back(task_id);
        }
    }

    Metrics metrics;
    metrics.starts.assign(op_count, 0.0);
    metrics.finishes.assign(op_count, 0.0);
    double dev_ready = 0.0;
    double host_ready = 0.0;
    double network_ready = 0.0;
    constexpr double eps = 1e-9;

    for (int scheduled = 0; scheduled < static_cast<int>(tasks.size()); ++scheduled) {
        if (ready.empty()) {
            throw std::runtime_error("Expanded pipeline graph contains a cycle.");
        }
        int best_ready_index = -1;
        int best_task = -1;
        double best_start = std::numeric_limits<double>::infinity();
        double best_priority = -std::numeric_limits<double>::infinity();
        for (int index = 0; index < static_cast<int>(ready.size()); ++index) {
            int task_id = ready[index];
            const auto& task = tasks[task_id];
            double start = std::max(
                ready_time[task_id],
                resource_ready_at(task.resource, dev_ready, host_ready, network_ready)
            );
            start = std::max(start, task.release);
            double priority = task_priority(task, ranks, task_id, unroll, rule);
            bool better = start + eps < best_start;
            if (!better && std::abs(start - best_start) <= eps) {
                better = priority > best_priority + eps ||
                         (std::abs(priority - best_priority) <= eps && task_id < best_task);
            }
            if (better) {
                best_ready_index = index;
                best_task = task_id;
                best_start = start;
                best_priority = priority;
            }
        }

        ready.erase(ready.begin() + best_ready_index);
        const auto& task = tasks[best_task];
        double finish = best_start + task.duration;
        set_resource_ready(task.resource, finish, dev_ready, host_ready, network_ready);

        if (task.resource == ResourceKind::Network) {
            metrics.transfers.push_back(
                TransferRecord{task.edges, best_start, finish, task.size, task.batched}
            );
        } else if (task.op >= 0 && task.op < op_count) {
            metrics.starts[task.op] = best_start;
            metrics.finishes[task.op] = finish;
        }

        for (int successor : task.successors) {
            ready_time[successor] = std::max(ready_time[successor], finish);
            pending[successor] -= 1;
            if (pending[successor] == 0) {
                ready.push_back(successor);
            }
        }
    }

    return finalize_metrics(graph, assignment, weights, scales, unroll, std::move(metrics));
}

bool is_better_metrics(const Metrics& candidate, const Metrics& best) {
    constexpr double eps = 1e-9;
    if (candidate.loss < best.loss - eps) {
        return true;
    }
    if (std::abs(candidate.loss - best.loss) > eps) {
        return false;
    }
    if (candidate.max_frame_latency < best.max_frame_latency - eps) {
        return true;
    }
    if (candidate.max_frame_latency > best.max_frame_latency + eps) {
        return false;
    }
    if (candidate.latency < best.latency - eps) {
        return true;
    }
    if (candidate.latency > best.latency + eps) {
        return false;
    }
    return candidate.utilization > best.utilization + eps;
}

Metrics schedule(
    const GraphData& graph,
    const std::vector<int>& assignment,
    double bandwidth,
    double latency,
    const ObjectiveWeights& weights,
    const ObjectiveScales& scales,
    bool batch_transfers,
    int pipeline_unroll
) {
    int unroll = std::max(1, pipeline_unroll);
    const ScheduleRule rules[] = {
        ScheduleRule::CriticalPath,
        ScheduleRule::DeviceFirst,
        ScheduleRule::FifoReady,
    };
    bool has_best = false;
    Metrics best;
    const bool binary_variants[] = {false, true};
    for (bool defer_blocked_transfers : binary_variants) {
        for (bool gate_next_frame_work : binary_variants) {
            auto tasks = build_schedule_tasks(
                graph,
                assignment,
                bandwidth,
                latency,
                batch_transfers,
                defer_blocked_transfers,
                gate_next_frame_work,
                unroll
            );
            auto ranks = task_ranks(tasks);
            for (ScheduleRule rule : rules) {
                Metrics candidate = simulate_schedule(
                    graph,
                    assignment,
                    weights,
                    scales,
                    tasks,
                    ranks,
                    unroll,
                    rule
                );
                if (!has_best || is_better_metrics(candidate, best)) {
                    best = std::move(candidate);
                    has_best = true;
                }
            }
        }
    }
    return best;
}

ObjectiveScales baseline_scales(
    const GraphData& graph,
    double bandwidth,
    double latency,
    bool batch_transfers,
    int pipeline_unroll
) {
    int n = static_cast<int>(graph.ids.size());
    std::vector<int> all_device(n, 0);
    std::vector<int> mostly_host(n, 1);
    for (int i = 0; i < n; ++i) {
        if (graph.fixed_dev[i]) {
            mostly_host[i] = 0;
        }
    }
    ObjectiveWeights weights{1.0, 1.0, 0.0};
    ObjectiveScales neutral_scales{1.0, 1.0};
    auto dev = schedule(
        graph,
        all_device,
        bandwidth,
        latency,
        weights,
        neutral_scales,
        batch_transfers,
        pipeline_unroll
    );
    auto host = schedule(
        graph,
        mostly_host,
        bandwidth,
        latency,
        weights,
        neutral_scales,
        batch_transfers,
        pipeline_unroll
    );
    return ObjectiveScales{
        std::max(dev.latency, host.latency),
        std::max(dev.max_frame_latency, host.max_frame_latency),
    };
}

bool exceeds_limit(
    const Metrics& metrics,
    double latency_limit,
    double max_frame_latency_limit
) {
    return (latency_limit > 0.0 && metrics.latency > latency_limit) ||
           (max_frame_latency_limit > 0.0 &&
            metrics.max_frame_latency > max_frame_latency_limit);
}

std::vector<int> mostly_host_seed(const GraphData& graph, const std::vector<int>& base) {
    std::vector<int> assignment = base;
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (!graph.fixed_dev[i]) {
            assignment[i] = 1;
        }
    }
    return assignment;
}

py::dict metrics_to_python(const GraphData& graph, const Metrics& metrics, int unroll) {
    py::dict out;
    py::dict starts;
    py::dict finishes;
    int op_count = static_cast<int>(metrics.starts.size());
    for (int op = 0; op < op_count; ++op) {
        std::string id = op_id(graph, op, unroll);
        starts[py::str(id)] = metrics.starts[op];
        finishes[py::str(id)] = metrics.finishes[op];
    }
    py::list transfers;
    for (const auto& transfer : metrics.transfers) {
        py::list edges;
        for (auto [source, target] : transfer.edges) {
            edges.append(py::make_tuple(op_id(graph, source, unroll), op_id(graph, target, unroll)));
        }
        py::dict item;
        item["edges"] = edges;
        item["start"] = transfer.start;
        item["finish"] = transfer.finish;
        item["size_mb"] = transfer.size;
        item["batched"] = transfer.batched;
        transfers.append(item);
    }
    out["latency"] = metrics.latency;
    out["max_frame_latency"] = metrics.max_frame_latency;
    out["device_utilization"] = metrics.utilization;
    out["avg_latency_loss"] = metrics.avg_latency_loss;
    out["max_frame_latency_loss"] = metrics.max_frame_latency_loss;
    out["device_utilization_loss"] = metrics.device_utilization_loss;
    out["loss"] = metrics.loss;
    out["start_times"] = starts;
    out["finish_times"] = finishes;
    out["transfer_records"] = transfers;
    out["pipeline_unroll"] = unroll;
    return out;
}

py::dict solve_core(const py::dict& data, const py::dict& params) {
    GraphData graph = parse_graph(data);
    int n = static_cast<int>(graph.ids.size());
    double bandwidth = non_negative(params["bandwidth"].cast<double>(), "Environment bandwidth");
    if (bandwidth <= 0.0) {
        throw std::runtime_error("Environment bandwidth must be greater than zero.");
    }
    double latency = non_negative(params["latency"].cast<double>(), "Environment latency");
    ObjectiveWeights weights = parse_objective_weights(params);
    std::string algorithm = params["algorithm"].cast<std::string>();
    int heuristic_iterations = params["heuristic_iterations"].cast<int>();
    int seed = params["seed"].cast<int>();
    double latency_limit = non_negative(
        params["latency_limit"].cast<double>(), "Environment latency_limit"
    );
    double max_frame_latency_limit = non_negative(
        params["max_frame_latency_limit"].cast<double>(),
        "Environment max_frame_latency_limit"
    );
    bool batch_transfers = params["batch_transfers"].cast<bool>();
    int pipeline_unroll = std::max(1, params["pipeline_unroll"].cast<int>());

    std::transform(algorithm.begin(), algorithm.end(), algorithm.begin(), [](unsigned char c) {
        if (c == ' ' || c == '-') {
            return '_';
        }
        return static_cast<char>(std::tolower(c));
    });

    std::vector<int> free_nodes;
    std::vector<int> base_assignment(n, 1);
    for (int i = 0; i < n; ++i) {
        if (graph.fixed_dev[i]) {
            base_assignment[i] = 0;
        } else {
            base_assignment[i] = graph.x_initial[i];
            free_nodes.push_back(i);
        }
    }

    ObjectiveScales scales = baseline_scales(
        graph, bandwidth, latency, batch_transfers, pipeline_unroll
    );

    std::string mode;
    int iterations = 0;
    std::vector<int> best_assignment;
    Metrics best_metrics;
    bool has_best = false;

    auto eval_assignment = [&](const std::vector<int>& assignment) {
        return schedule(
            graph, assignment, bandwidth, latency, weights, scales,
            batch_transfers, pipeline_unroll
        );
    };

    bool use_brute = false;
    if ((algorithm == "auto" || algorithm.empty()) && free_nodes.size() <= 12) {
        use_brute = true;
        mode = "Enumerate";
    } else if (algorithm == "enumerate" || algorithm == "brute" || algorithm == "brute_force") {
        use_brute = true;
        mode = "Enumerate";
    }

    if (use_brute) {
        if (free_nodes.size() >= 63) {
            throw std::runtime_error("Enumerate has too many free nodes.");
        }
        std::uint64_t total = static_cast<std::uint64_t>(1) << free_nodes.size();
        for (std::uint64_t mask = 0; mask < total; ++mask) {
            iterations += 1;
            std::vector<int> assignment = base_assignment;
            for (size_t bit = 0; bit < free_nodes.size(); ++bit) {
                assignment[free_nodes[bit]] = static_cast<int>((mask >> bit) & 1U);
            }
            Metrics metrics = eval_assignment(assignment);
            if (exceeds_limit(metrics, latency_limit, max_frame_latency_limit)) {
                continue;
            }
            if (!has_best || metrics.loss < best_metrics.loss) {
                best_assignment = std::move(assignment);
                best_metrics = std::move(metrics);
                has_best = true;
            }
        }
    } else if (algorithm == "auto" || algorithm.empty() || algorithm == "random" ||
               algorithm == "random_search" || algorithm == "random_n") {
        mode = (algorithm == "auto" || algorithm.empty()) ? "Auto Random" : "Random Search";
        std::mt19937 rng(seed);
        best_assignment = mostly_host_seed(graph, base_assignment);
        best_metrics = eval_assignment(best_assignment);
        has_best = !exceeds_limit(best_metrics, latency_limit, max_frame_latency_limit);
        if (!has_best) {
            best_assignment.clear();
        }
        if (free_nodes.empty()) {
            iterations = 1;
        } else {
            int total_iterations = std::max(0, heuristic_iterations);
            iterations = total_iterations;
            for (int step = 0; step < total_iterations; ++step) {
                std::vector<int> assignment = has_best ? best_assignment : mostly_host_seed(graph, base_assignment);
                int max_flips = std::max(1, std::min(3, static_cast<int>(free_nodes.size())));
                std::uniform_int_distribution<int> flip_dist(1, max_flips);
                int flips = flip_dist(rng);
                std::vector<int> pool = free_nodes;
                std::shuffle(pool.begin(), pool.end(), rng);
                for (int i = 0; i < flips; ++i) {
                    int node = pool[i];
                    assignment[node] = 1 - assignment[node];
                }
                Metrics metrics = eval_assignment(assignment);
                if (exceeds_limit(metrics, latency_limit, max_frame_latency_limit)) {
                    continue;
                }
                if (!has_best || metrics.loss < best_metrics.loss) {
                    best_assignment = std::move(assignment);
                    best_metrics = std::move(metrics);
                    has_best = true;
                }
            }
        }
    } else if (algorithm == "simulated_annealing" || algorithm == "annealing" ||
               algorithm == "sim_anneal" || algorithm == "sim_aneal") {
        mode = "Simulated Annealing";
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::vector<int> current = mostly_host_seed(graph, base_assignment);
        Metrics current_metrics = eval_assignment(current);
        if (!exceeds_limit(current_metrics, latency_limit, max_frame_latency_limit)) {
            best_assignment = current;
            best_metrics = current_metrics;
            has_best = true;
        }
        if (free_nodes.empty()) {
            iterations = 1;
        } else {
            int total_iterations = std::max(0, heuristic_iterations);
            iterations = total_iterations;
            double initial_temp = 1.0;
            double final_temp = 0.01;
            std::uniform_int_distribution<int> node_dist(0, static_cast<int>(free_nodes.size()) - 1);
            for (int step = 0; step < total_iterations; ++step) {
                double progress = static_cast<double>(step) / std::max(1, total_iterations - 1);
                double temperature = initial_temp * std::pow(final_temp / initial_temp, progress);
                std::vector<int> candidate = current;
                int node = free_nodes[node_dist(rng)];
                candidate[node] = 1 - candidate[node];
                Metrics metrics = eval_assignment(candidate);
                if (exceeds_limit(metrics, latency_limit, max_frame_latency_limit)) {
                    continue;
                }
                double delta = metrics.loss - current_metrics.loss;
                bool accept = exceeds_limit(
                                  current_metrics,
                                  latency_limit,
                                  max_frame_latency_limit
                              ) ||
                              delta <= 0.0 ||
                              unit(rng) < std::exp(-delta / std::max(temperature, 1e-9));
                if (accept) {
                    current = candidate;
                    current_metrics = metrics;
                }
                if (!has_best || metrics.loss < best_metrics.loss) {
                    best_assignment = std::move(candidate);
                    best_metrics = std::move(metrics);
                    has_best = true;
                }
            }
        }
    } else {
        throw std::runtime_error("Unknown solver algorithm: " + algorithm);
    }

    if (!has_best) {
        throw std::runtime_error(
            "No feasible assignment satisfies latency limits: E2E " +
            std::to_string(latency_limit) +
            " ms, max-frame " +
            std::to_string(max_frame_latency_limit) +
            " ms."
        );
    }

    py::dict result;
    py::dict assignment_dict;
    for (int i = 0; i < n; ++i) {
        assignment_dict[py::str(graph.ids[i])] = best_assignment[i];
    }
    result["assignment"] = assignment_dict;
    result["metrics"] = metrics_to_python(graph, best_metrics, pipeline_unroll);
    result["mode"] = mode;
    result["iterations"] = iterations;
    return result;
}

py::dict evaluate_core(const py::dict& data, const py::dict& params) {
    GraphData graph = parse_graph(data);
    int n = static_cast<int>(graph.ids.size());
    auto assignment = params["assignment"].cast<std::vector<int>>();
    if (assignment.size() != static_cast<size_t>(n)) {
        throw std::runtime_error("Assignment length does not match graph nodes.");
    }
    for (int node = 0; node < n; ++node) {
        if (assignment[node] != 0 && assignment[node] != 1) {
            throw std::runtime_error("Assignment values must be 0 or 1.");
        }
    }

    double bandwidth = non_negative(params["bandwidth"].cast<double>(), "Environment bandwidth");
    if (bandwidth <= 0.0) {
        throw std::runtime_error("Environment bandwidth must be greater than zero.");
    }
    double latency = non_negative(params["latency"].cast<double>(), "Environment latency");
    ObjectiveWeights weights = parse_objective_weights(params);
    bool batch_transfers = params["batch_transfers"].cast<bool>();
    int pipeline_unroll = std::max(1, params["pipeline_unroll"].cast<int>());

    ObjectiveScales scales;
    if (params.contains("avg_latency_scale") && params.contains("max_latency_scale")) {
        scales.avg_latency = params["avg_latency_scale"].cast<double>();
        scales.max_latency = params["max_latency_scale"].cast<double>();
        if (!std::isfinite(scales.avg_latency) || !std::isfinite(scales.max_latency)) {
            throw std::runtime_error("Latency scales must be finite.");
        }
    } else {
        scales = baseline_scales(graph, bandwidth, latency, batch_transfers, pipeline_unroll);
    }

    Metrics metrics = schedule(
        graph,
        assignment,
        bandwidth,
        latency,
        weights,
        scales,
        batch_transfers,
        pipeline_unroll
    );
    return metrics_to_python(graph, metrics, pipeline_unroll);
}

py::tuple baseline_scales_core(const py::dict& data, const py::dict& params) {
    GraphData graph = parse_graph(data);
    double bandwidth = non_negative(params["bandwidth"].cast<double>(), "Environment bandwidth");
    if (bandwidth <= 0.0) {
        throw std::runtime_error("Environment bandwidth must be greater than zero.");
    }
    double latency = non_negative(params["latency"].cast<double>(), "Environment latency");
    bool batch_transfers = params["batch_transfers"].cast<bool>();
    int pipeline_unroll = std::max(1, params["pipeline_unroll"].cast<int>());
    auto scales = baseline_scales(graph, bandwidth, latency, batch_transfers, pipeline_unroll);
    return py::make_tuple(scales.avg_latency, scales.max_latency);
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "C++ solver core for CoopInfer";
    module.def("edge_transfer_ms", &transfer_ms, "Compute serialized transfer duration in ms");
    module.def("evaluate_core", &evaluate_core, "Evaluate a fixed CoopInfer assignment with the C++ core");
    module.def("baseline_scales_core", &baseline_scales_core, "Compute objective latency normalization scales");
    module.def("solve_core", &solve_core, "Solve a CoopInfer graph with the C++ core");
}
