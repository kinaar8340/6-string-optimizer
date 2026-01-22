# app/demo.py
# Gradio real-time demo: watch the six hierarchical layers ("strings") vibrate
# Visualizes:
#   • 3D trajectory of u → [1,1,1] basin (Plotly scatter)
#   • Loss curve over time
#   • Six "string gauges" — tension = recent stagnation severity (or 0 after reset/burst)
#   • Burst events flash red on each string when layer triggers

import gradio as gr
import torch
import geoopt
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from optimizer import GeooptBurstOptimizer, HierarchicalGeooptBurstOptimizer
from optimizer.models import SphereRosenbrockModel
from optimizer.utils import stereographic_projection
from optimizer.losses import rosenbrock_3d

torch.manual_seed(42)


# Collector for real-time metrics
class OptimizerMonitor:
    def __init__(self, layer_names):
        self.layer_names = layer_names
        self.history = {
            "step": [],
            "loss": [],
            "u_x": [], "u_y": [], "u_z": [],
        }
        self.layer_tension = {name: 0.0 for name in layer_names}  # Current "tension" (severity proxy)
        self.layer_events = {name: [] for name in layer_names}  # List of (step, event_type)

    def record_step(self, step, loss_val, u):
        self.history["step"].append(step)
        self.history["loss"].append(loss_val)
        self.history["u_x"].append(u[0])
        self.history["u_y"].append(u[1])
        self.history["u_z"].append(u[2])

    def record_event(self, layer_name, event_type, severity=0.0):
        self.layer_events[layer_name].append((len(self.history["step"]), event_type))
        self.layer_tension[layer_name] = severity if "Burst" in event_type else 0.0


monitor = None

# Patch Hierarchical to report events to monitor
_original_step = HierarchicalGeooptBurstOptimizer.step


def monitored_step(self, closure=None):
    loss = _original_step(self, closure=closure)
    if loss is not None and len(self.loss_history) == self.stagnation_window:
        improvement = self.loss_history[0] - loss.item()
        if improvement < self.stagnation_thresh:
            monitor.record_event(self.name, "Burst", severity=abs(improvement))
        elif improvement > 2 * self.stagnation_thresh and any(
                abs(g["burst_factor_max"] - self.originals[id(g)][0]) > 1e-6 for g in self.param_groups
        ):
            monitor.record_event(self.name, "Reset")
    return loss


HierarchicalGeooptBurstOptimizer.step = monitored_step


def run_optimization(steps_per_update=100, total_steps=50000):
    global monitor
    model = SphereRosenbrockModel(num_instances=1)

    def closure():
        model.zero_grad()
        q = model()
        u = stereographic_projection(q)
        loss = rosenbrock_3d(u).mean()
        loss.backward()
        return loss

    string_names = ["HighE", "B", "G", "D", "A", "LowE"]
    monitor = OptimizerMonitor(string_names)

    # Ultra-conservative Eb Master hierarchy
    windows = [400, 580, 780, 1100, 1600, 2400]
    burst_boosts = [1.02, 1.04, 1.06, 1.08, 1.10, 1.15]
    theta_boosts = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06]

    base_opt = GeooptBurstOptimizer(
        model.parameters(),
        lr=0.03,
        burst_factor_max=8.0,
        damping_min=0.98,
        damping=0.995,
        max_theta=torch.pi / 20,
        verbose=False  # Silent for demo speed
    )

    opt = base_opt
    for name, win, bb, tb in zip(string_names, windows, burst_boosts, theta_boosts):
        opt = HierarchicalGeooptBurstOptimizer(
            opt,
            stagnation_window=win,
            stagnation_thresh=5e-5,
            burst_boost=bb,
            theta_boost=tb,
            name=name,
            verbose=False
        )

    step = 0
    best_loss = float("inf")
    while step < total_steps:
        for _ in range(steps_per_update):
            loss = opt.step(closure)
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
            with torch.no_grad():
                u = stereographic_projection(model.q).squeeze(0).cpu().numpy()
            monitor.record_step(step, loss_val, u)
            step += 1
            if step >= total_steps:
                break

        # Yield updated plots
        df = pd.DataFrame(monitor.history)

        # 3D trajectory
        fig_3d = go.Figure(data=[
            go.Scatter3d(x=df["u_x"], y=df["u_y"], z=df["u_z"], mode='lines+markers',
                         marker=dict(size=3, color=df["loss"], colorscale='Viridis', showscale=True))
        ])
        fig_3d.add_trace(go.Scatter3d(x=[1], y=[1], z=[1], mode='markers', marker=dict(size=8, color='red')))
        fig_3d.update_layout(title="Trajectory → Global Minimum [1,1,1]", scene=dict(aspectmode='cube'))

        # Loss curve
        fig_loss = px.line(df, x="step", y="loss", title="Loss Over Time", log_y=True)

        # String tension gauges (horizontal bars)
        tension_data = pd.DataFrame({
            "String": monitor.layer_names,
            "Tension": [monitor.layer_tension[name] for name in monitor.layer_names]
        })
        fig_gauges = px.bar(tension_data, x="Tension", y="String", orientation='h',
                            title="String Tension (Stagnation Severity)",
                            range_x=[0, tension_data["Tension"].max() * 1.2 or 1])

        yield fig_3d, fig_loss, fig_gauges, f"Step: {step} | Loss: {loss_val:.6f} (best: {best_loss:.6f})"


# Gradio interface
with gr.Blocks(title="6-string-optimizer six-string demo") as demo:
    gr.Markdown("# 6-string-optimizer — Watch the six strings vibrate in real time")
    gr.Markdown(
        "Eb-tuned hierarchical optimizer on compactified Rosenbrock. Trajectory approaches global minimum [1,1,1]. Tension gauges show layer activity.")

    with gr.Row():
        plot_3d = gr.Plot(label="3D Trajectory in R³")
        plot_loss = gr.Plot(label="Loss Curve")
    with gr.Row():
        plot_gauges = gr.Plot(label="String Tension Gauges")
        status = gr.Textbox(label="Status")

    btn = gr.Button("Start Optimization")
    btn.click(run_optimization, inputs=[], outputs=[plot_3d, plot_loss, plot_gauges, status])

demo.queue().launch()