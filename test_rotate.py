import streamlit as st
import plotly.graph_objects as go
import numpy as np

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=[1,2,3], y=[1,2,3], z=[1,2,3], mode='markers'))
fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)))
st.plotly_chart(fig)

st.components.v1.html("""
<script>
let angle = 0;
function rotate() {
    let plots = window.parent.document.querySelectorAll('.js-plotly-plot');
    if (plots.length > 0) {
        let plot = plots[0];
        angle += 0.01;
        let x = 2 * Math.cos(angle);
        let y = 2 * Math.sin(angle);
        // We use Plotly.relayout to update camera without redrawing data
        if (window.parent.Plotly) {
            window.parent.Plotly.relayout(plot, {'scene.camera.eye': {x: x, y: y, z: 0.5}});
        }
    }
    requestAnimationFrame(rotate);
}
setTimeout(rotate, 1000);
</script>
""")
