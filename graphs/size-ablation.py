import plotly.graph_objs as go

model_size = [0.5, 1.5, 3, 7]
mrr = [0.187, 0.183, 0.31, 0.471]
h_line = 0.739
fig = None
legend_config = dict(
    x=0.02,
    y=0.98,
    xanchor="left",
    yanchor="top",
    bgcolor="rgba(255,255,255,0.7)",
    font=dict(size=20),
    bordercolor="black",
    borderwidth=1,
)
output_path = __file__.replace(".py", ".pdf")
fig = go.Figure(
    data=[
        go.Scatter(
            x=[0, 8],
            y=[h_line, h_line],
            mode="lines",
            line=dict(color="red", width=2, dash="dot"),
            name="Gemini 2.5 Flash",
        ),
        go.Scatter(
            x=model_size,
            y=mrr,
            mode="lines+markers",
            marker=dict(size=8),
            line=dict(width=2, color="blue"),
            name="Qwen2.5 Coder",
        ),
    ]
)
# Add numerical values on top of each point, except for 1.5B parameters which sould be under the point
for i in range(len(model_size)):
    if model_size[i] == 3:
        fig.add_annotation(
            x=model_size[i],
            y=mrr[i],
            text=f"{mrr[i]:.3f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-20,
            font=dict(size=16, color="black"),
            arrowcolor="black",
        )
        continue
    fig.add_annotation(
        x=model_size[i],
        y=mrr[i],
        text=f"{mrr[i]:.3f}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=20,
        font=dict(size=16, color="black"),
        arrowcolor="black",
    )

# Add datapoint for the Gemini 2.5 Flash line on the on the y-axis line
fig.add_annotation(
    x=8,
    y=h_line,
    text=f"{h_line:.3f}",
    showarrow=True,
    arrowhead=1,
    ax=40,
    ay=0,
    font=dict(size=16, color="black"),
    arrowcolor="black",
)

fig.update_layout(
    title=None,
    legend=legend_config,
    margin=dict(l=80, r=100, t=50, b=80),
)
# Add a full contour (box) around the plot area
fig.update_layout(
    xaxis=dict(showline=True, linewidth=2, linecolor="black", mirror=True),
    yaxis=dict(showline=True, linewidth=2, linecolor="black", mirror=True),
)
fig.update_layout(
    xaxis=dict(
        title=dict(text="Model Size (Billion parameters)", font=dict(size=32)),
        tickfont=dict(size=20),
    ),
    yaxis=dict(title=dict(text="Mean Reciprocal Rank", font=dict(size=32)), tickfont=dict(size=20)),
)
fig.update_xaxes(range=[0, 8])
fig.update_yaxes(range=[0, 1])


# Add a linear regression of the Qwen2.5 Coder 7B datapoints
import numpy as np

z = np.polyfit(model_size, mrr, 1)
p = np.poly1d(z)
fig.add_trace(
    go.Scatter(
        x=[0, 8],
        y=[p(0), p(8)],
        mode="lines",
        line=dict(color="blue", width=2, dash="dash"),
        name="Linear Regression",
        showlegend=False,
    )
)

baselines = {
    "CLAP": 0.244,
    "SAFE": 0.189,
    "PalmTree": 0.020,
    "Asm2Vec": 0.494,
    "Order M.": 0.006,
}
# Add grey dotted horizonal lines for each baselines, with labels on the right side of the plot, with some spacing between them
for name, value in baselines.items():
    fig.add_trace(
        go.Scatter(
            x=[0, 8],
            y=[value, value],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.9)", width=1, dash="dot"),
            name=name,
            showlegend=False,
        )
    )
    if name == "PalmTree":
        continue

    fig.add_annotation(
        x=8,
        y=value,
        text=name,
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=18, color="grey"),
        bgcolor="rgba(255,255,255,0.7)",
    )

# Put the palmtree label on top of its (inside of the chart area)
fig.add_annotation(
    x=7.1,
    y=baselines["PalmTree"] + 0.03,
    text="PalmTree",
    showarrow=False,
    xanchor="left",
    yanchor="middle",
    font=dict(size=18, color="grey"),
    bgcolor="rgba(255,255,255,0.7)",
)

# Add an entry in the legend where grey doted line is explained as "Baselines"
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="rgba(0,0,0,0.9)", width=1, dash="dot"),
        name="Baselines",
    )
)

# White template
fig.update_layout(template="plotly_white")

# Save as pdf instead
fig.write_image(output_path, width=1000, height=640, scale=2)
