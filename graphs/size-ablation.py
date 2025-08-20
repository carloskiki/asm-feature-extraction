import plotly.graph_objs as go
import plotly.io as pio

model_size = [0.5, 1.5, 3, 7]
mrr = [0.187, 0.183, 0.31, 0.471]
h_line = 0.739
fig = None
legend_config = dict(x=0.02, y=0.98, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.7)')
output_path = __file__.replace('.py', '.png')
fig = go.Figure(
    data=[
        go.Scatter(
            x=model_size,
            y=mrr,
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2),
            name='MRR'
        ),
        go.Scatter(
            x=[0, 8],
            y=[h_line, h_line],
            mode='lines',
            line=dict(color='red', width=2, dash='dot'),
            name='Gemini 2.5 Flash'
        )
    ]
)
fig.update_layout(
    title=None,
    legend=dict(
        x=0.98,
        y=0.02,
        xanchor='right',
        yanchor='bottom',
        bgcolor='rgba(255,255,255,0.7)'
    )
)
fig.update_xaxes(range=[0, 8])
fig.update_yaxes(range=[0, 1])
fig.update_layout(
    xaxis_title='Model Size (B parameters)',
    yaxis_title='MRR',
    template='plotly_white'
)
fig.write_image(output_path, scale=3, width=1200, height=800)