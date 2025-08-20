cross_opt = [0.739, 0.694, 0.675, 0.542, 0.711, 0.447]
cross_arch = [0.745, 0.624, 0.537, 0.579, 0.666, 0.523]
categories = ["default", "signature", "logic", "constants", "effects", "category"]

import plotly.graph_objects as go

colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']

# Prepare x and y for each group with a gap between groups
gap = 1  # size of the gap between groups
x_indices = list(range(6)) + [6 + gap + i for i in range(6)]
x_cross_opt = x_indices[:6]
x_cross_arch = x_indices[6:]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=x_cross_opt,
    y=cross_opt,
    name='Cross-Optimization',
    marker_color=colors
))

fig.add_trace(go.Bar(
    x=x_cross_arch,
    y=cross_arch,
    name='Cross-Architecture',
    marker_color=colors
))

# Set custom tick labels for both groups
tickvals = x_indices
ticktext = categories + categories

# Remove the first "default" bar from both groups
cross_opt = cross_opt[1:]
cross_arch = cross_arch[1:]
categories = categories[1:]

# Prepare new x indices for 5 bars per group
gap = 1
x_indices = list(range(5)) + [5 + gap + i for i in range(5)]
x_cross_opt = x_indices[:5]
x_cross_arch = x_indices[5:]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=x_cross_opt,
    y=cross_opt,
    name='Cross-Optimization',
    marker_color=colors[1:6]
))

fig.add_trace(go.Bar(
    x=x_cross_arch,
    y=cross_arch,
    name='Cross-Architecture',
    marker_color=colors[1:6]
))

# Set custom tick labels for both groups
tickvals = x_indices
ticktext = categories + categories

# Draw horizontal dotted lines for "default" values (use original first values)
default_cross_opt = 0.739
default_cross_arch = 0.745

# Update legend to show categories by color
for i, cat in enumerate(categories):
    fig.add_trace(go.Bar(
        x=[None],  # Dummy invisible bar for legend
        y=[None],
        marker_color=colors[i+1],
        name=cat,
        showlegend=True,
        visible='legendonly'
    ))
center = (x_cross_opt[-1] + x_cross_arch[0]) / 2

fig.add_shape(
    type="line",
    x0=x_cross_opt[0] - 0.5,
    x1=center,
    y0=default_cross_opt,
    y1=default_cross_opt,
    line=dict(color=colors[0], width=2, dash="dot"),
    layer="above"
)
fig.add_shape(
    type="line",
    x0=center,
    x1=x_cross_arch[-1] + 0.5,
    y0=default_cross_arch,
    y1=default_cross_arch,
    line=dict(color=colors[0], width=2, dash="dot"),
    layer="above"
)

fig.update_layout(
    barmode='group',
    xaxis_title='Category',
    yaxis_title='Score',
    title='Cross-Optimization vs Cross-Architecture by Category',
    legend_title='Group',
    xaxis=dict(
        tickvals=tickvals,
        ticktext=ticktext,
        range=[-1.5, max(x_indices) + 1.5],  # Add space on left and right
    ),
    yaxis=dict(
        range=[0, 1]  # Set y-axis to go up to 1
    )
)

fig.write_image("prompt-ablation.png", scale=3, width=1200, height=800)