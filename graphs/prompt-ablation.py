import plotly.graph_objects as go

cross_opt = [0.739, 0.694, 0.675, 0.542, 0.711, 0.447]
cross_arch = [0.745, 0.624, 0.537, 0.579, 0.666, 0.523]
categories = ["default", "signature", "logic", "constants", "effects", "category"]

fig = go.Figure()

# Define colors for each category (excluding default since it will be a line) - high contrast colors
colors = ['#FF6B35', '#004E89', '#00A896', '#8E44AD', '#E74C3C']

# Add bars for each category (excluding default)
for i, category in enumerate(categories[1:], 1):
    # Cross Optimization bars
    fig.add_trace(go.Bar(
        x=['Cross Optimization'],
        y=[cross_opt[i]],
        name=category,
        legendgroup=category,
        marker_color=colors[i-1],
        offsetgroup=i-1,
        width=0.15,
        showlegend=True
    ))
    
    # Cross Architecture bars
    fig.add_trace(go.Bar(
        x=['Cross Architecture'],
        y=[cross_arch[i]],
        name=category,
        legendgroup=category,
        marker_color=colors[i-1],
        offsetgroup=i-1,
        width=0.15,
        showlegend=False
    ))

# Add horizontal lines for default category as ceiling over each cluster
fig.add_hline(
    y=cross_opt[0], 
    line_dash="dot", 
    line_color="#1f77b4",
    line_width=3,
    x0=-0.0, x1=0.5,  # From left edge of Cross Optimization cluster to center
    xref="x"
)

fig.add_hline(
    y=cross_arch[0], 
    line_dash="dot", 
    line_color="#1f77b4", 
    line_width=3,
    x0=0.5, x1=1.0,  # From center to right edge of Cross Architecture cluster
    xref="x"
)

fig.update_layout(
    barmode='group',
    xaxis_title='',
    yaxis_title='MRR',
    yaxis=dict(range=[0, 1]),
    font=dict(size=32),
)

fig.update_layout(
    legend=dict(
        font=dict(size=32),
        orientation="h", 
        yanchor="bottom", 
        y=1.02, 
        xanchor="center", 
        x=0.5,
    )
)

# Add a full contour (box) around the plot area
fig.update_layout(
    xaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True),
    yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True)
)

fig.write_image("prompt-ablation.png", scale=3, width=1200, height=800)