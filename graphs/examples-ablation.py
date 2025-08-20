import plotly.graph_objects as go

gemini = [0.469, 0.715, 0.711, 0.729, 0.737, 0.707]
qwen = [0.398, 0.502, 0.484, 0.568, 0.441]
blank = [0.472, 0.573, 0.568]
example_count = [0, 1, 2, 3, 4, 5]


fig = go.Figure()

fig.update_yaxes(range=[0.2, 1])

fig.add_trace(go.Scatter(
    x=example_count,
    y=gemini,
    mode='lines+markers',
    name='Gemini 2.5 Flash',
    marker=dict(size=10, symbol='circle'),
    line=dict(width=3, color='red')
))

fig.add_trace(go.Scatter(
    x=example_count[:len(qwen)],
    y=qwen,
    mode='lines+markers',
    name='Qwen2.5 Coder 7B',
    marker=dict(size=10, symbol='square'),
    line=dict(width=3, color='blue')
))

fig.add_hline(
    y=0.751,
    line=dict(color='red', width=2, dash='dot'),
)

fig.add_trace(go.Scatter(
    x=example_count[1:1+len(blank)],
    y=blank,
    mode='lines+markers',
    name='Qwen2.5 Coder 7B Blank',
    marker=dict(size=10, symbol='diamond'),
    line=dict(width=3)
))

fig.update_layout(
    xaxis_title='Number of Examples',
    yaxis_title='MRR',
    font=dict(size=32),
    legend=dict(font=dict(size=24)),
)

# Add a full contour (box) around the plot area
fig.update_layout(
    xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
    yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True)
)

fig.update_layout(
    legend=dict(
        font=dict(size=24),
        x=0.02,
        y=0.98,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor='black',
        borderwidth=1
    )
)
fig.write_image("graphs/examples-ablation.png", scale=3, width=1200, height=800)