import os
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import pydicom
import nibabel as nib
import logging

def convert_to_rgb(image):
    """Convert a grayscale image to RGB format"""
    # Normalize to 0-1
    normalized = (image - image.min()) / (image.max() - image.min())
    # Stack the same values for R, G, and B channels
    return np.stack([normalized] * 3, axis=2)

def create_transparent_colorscale(base_color, opacity=0.5):
    """Create a transparent colorscale for overlays"""
    return [
        [0, f'rgba({base_color[0]},{base_color[1]},{base_color[2]},0)'],
        [1, f'rgba({base_color[0]},{base_color[1]},{base_color[2]},{opacity})']
    ]
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Configuration
BASE_PATH = os.environ.get('OUTPUT_DIR', '/data')

def create_figure(loader, slice_idx, selected_zones, opacity=0.3):
    """Create a plotly figure for the given slice with proper overlay"""
    try:
        logger.debug(f"Creating figure for slice {slice_idx} with zones {selected_zones}")
        
        if not loader.dicom_series:
            logger.error("No DICOM series loaded")
            return go.Figure()

        # Get DICOM image
        dicom_data = loader.dicom_series[slice_idx]['image']
        logger.debug(f"DICOM shape: {dicom_data.shape}")
        
        # Normalize DICOM data to 0-255 range
        dicom_normalized = ((dicom_data - dicom_data.min()) / (dicom_data.max() - dicom_data.min()) * 255).astype(np.uint8)
        
        fig = go.Figure()

        # Add DICOM base image as Heatmap
        fig.add_trace(go.Heatmap(
            z=dicom_normalized,
            colorscale='gray',
            showscale=False,
            name='DICOM',
            hoverongaps=False,
            hoverinfo='none'
        ))

        # Define Jet-like colormaps for each zone
        colors = {
            'wg': [  # Red-based Jet
                [0.0, 'rgba(0,0,0,0)'],
                [0.2, 'rgba(128,0,0,{})'.format(opacity)],
                [0.4, 'rgba(255,0,0,{})'.format(opacity)],
                [0.6, 'rgba(255,69,0,{})'.format(opacity)],
                [0.8, 'rgba(255,140,0,{})'.format(opacity)],
                [1.0, 'rgba(255,215,0,{})'.format(opacity)]
            ],
            'tz': [  # Green-based Jet
                [0.0, 'rgba(0,0,0,0)'],
                [0.2, 'rgba(0,100,0,{})'.format(opacity)],
                [0.4, 'rgba(0,128,0,{})'.format(opacity)],
                [0.6, 'rgba(34,139,34,{})'.format(opacity)],
                [0.8, 'rgba(50,205,50,{})'.format(opacity)],
                [1.0, 'rgba(144,238,144,{})'.format(opacity)]
            ],
            'pz': [  # Blue-based Jet
                [0.0, 'rgba(0,0,0,0)'],
                [0.2, 'rgba(0,0,128,{})'.format(opacity)],
                [0.4, 'rgba(0,0,255,{})'.format(opacity)],
                [0.6, 'rgba(30,144,255,{})'.format(opacity)],
                [0.8, 'rgba(135,206,235,{})'.format(opacity)],
                [1.0, 'rgba(176,224,230,{})'.format(opacity)]
            ]
        }

        # Add probability masks
        for zone in selected_zones:
            if zone in loader.probability_masks:
                mask_data = loader.probability_masks[zone][:, :, slice_idx]
                
                # Normalize mask data
                mask_min, mask_max = mask_data.min(), mask_data.max()
                if mask_max > mask_min:
                    mask_normalized = (mask_data - mask_min) / (mask_max - mask_min)
                else:
                    mask_normalized = mask_data

                # Add mask as heatmap with Jet colormap
                fig.add_trace(go.Heatmap(
                    z=mask_normalized,
                    colorscale=colors[zone],
                    showscale=True,
                    opacity=1,  # Using 1 here because opacity is handled in the colorscale
                    name=f'{zone.upper()}',
                    hoverongaps=False,
                    hoverinfo='name+z',
                    colorbar=dict(
                        title=f'{zone.upper()} Probability',
                        titleside='right',
                        thickness=15,
                        len=0.75,
                    )
                ))

        # Update layout with correct properties
        fig.update_layout(
            title=f'Slice {slice_idx + 1}',
            height=800,
            width=800,
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='black',
            plot_bgcolor='black',
            yaxis=dict(
                scaleanchor='x',
                scaleratio=1,
                constrain='domain'
            ),
            xaxis=dict(
                constrain='domain'
            ),
            autosize=False
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=False, 
            showticklabels=False, 
            zeroline=False,
            range=[0, dicom_normalized.shape[1]]
        )
        fig.update_yaxes(
            showgrid=False, 
            showticklabels=False, 
            zeroline=False,
            range=[dicom_normalized.shape[0], 0]
        )

        return fig
        
    except Exception as e:
        logger.error(f"Error creating figure: {str(e)}")
        raise

class ImageLoader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.dicom_series = []
        self.probability_masks = {}
        logger.info(f"Initialized ImageLoader with base path: {base_path}")
        
    def load_data(self):
        """Load all necessary data"""
        try:
            logger.info("Starting data loading...")
            
            # Find anonymized directory
            anonymized_dir = self.base_path / 'anonymized'
            logger.debug(f"Looking for anonymized directory: {anonymized_dir}")
            
            if not anonymized_dir.exists():
                raise FileNotFoundError(f"Anonymized directory not found: {anonymized_dir}")
            
            # Get first patient directory
            patient_dirs = [d for d in anonymized_dir.iterdir() if d.is_dir()]
            if not patient_dirs:
                raise FileNotFoundError("No patient directories found")
            
            patient_dir = patient_dirs[0]
            patient_id = patient_dir.name
            logger.info(f"Processing patient: {patient_id}")
            
            # Load DICOM series
            self.load_dicom_series(patient_dir)
            
            # Load probability masks
            resampled_dir = self.base_path / f"{patient_id}_1" / 'Resampled'
            self.load_probability_masks(resampled_dir)
            
            logger.info("Data loading completed successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_dicom_series(self, patient_dir):
        """Load DICOM series from patient directory"""
        try:
            study_dir = next(patient_dir.iterdir())
            series_dir = next(study_dir.iterdir())
            
            dicom_files = sorted(series_dir.glob('image_*.dcm'))
            logger.info(f"Found {len(dicom_files)} DICOM files")
            
            self.dicom_series = []
            for dcm_path in dicom_files:
                ds = pydicom.dcmread(str(dcm_path))
                self.dicom_series.append({
                    'image': ds.pixel_array,
                    'position': float(ds.ImagePositionPatient[2])
                })
            
            self.dicom_series.sort(key=lambda x: x['position'])
            logger.info("DICOM series loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading DICOM series: {str(e)}")
            raise
    
    def load_probability_masks(self, resampled_dir):
        """Load probability masks from resampled directory"""
        try:
            logger.debug(f"Loading masks from: {resampled_dir}")
            
            for mask_type in ['wg', 'tz', 'pz']:
                mask_path = resampled_dir / f'{mask_type}_probs.nii.gz'
                if mask_path.exists():
                    logger.debug(f"Loading {mask_type} mask: {mask_path}")
                    nifti = nib.load(str(mask_path))
                    mask_data = nifti.get_fdata()
                    
                    # Orient mask
                    mask_data = np.rot90(mask_data, k=-1, axes=(0, 1))
                    mask_data = np.flip(mask_data, axis=1)
                    
                    self.probability_masks[mask_type] = mask_data
                    logger.debug(f"Loaded {mask_type} mask, shape: {mask_data.shape}")
                else:
                    logger.warning(f"Mask not found: {mask_path}")
            
        except Exception as e:
            logger.error(f"Error loading probability masks: {str(e)}")
            raise

# App layout
app.layout = dbc.Container([
    html.H1("Prostate Zone Visualization", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            html.H4("Controls", className="mb-3"),
            dcc.Slider(
                id='slice-slider',
                min=0,
                max=29,
                step=1,
                value=0,
                marks={i: str(i+1) for i in range(0, 30, 5)},
            ),
            html.Div([
                dbc.Checklist(
                    id='zone-checklist',
                    options=[
                        {'label': ' Whole Gland (WG)', 'value': 'wg'},
                        {'label': ' Transition Zone (TZ)', 'value': 'tz'},
                        {'label': ' Peripheral Zone (PZ)', 'value': 'pz'},
                    ],
                    value=['wg', 'tz', 'pz'],
                    inline=True,
                    className="my-3",
                    switch=True
                ),
            ]),
            dbc.Label("Opacity"),
            dcc.Slider(
                id='opacity-slider',
                min=0,
                max=1,
                step=0.1,
                value=0.3,
                marks={i/10: str(i/10) for i in range(0, 11, 2)},
            ),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='slice-viewer',
                config={
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'modeBarButtonsToAdd': ['drawclosedpath', 'eraseshape']
                },
                style={'height': '800px'}
            ),
        ], width=12, className="d-flex justify-content-center"),
    ], className="mt-4"),
    
], fluid=True, style={'backgroundColor': '#f8f9fa'})

@app.callback(
    Output('slice-viewer', 'figure'),
    [Input('slice-slider', 'value'),
     Input('zone-checklist', 'value'),
     Input('opacity-slider', 'value')]
)
def update_figure(slice_idx, selected_zones, opacity):
    """Update the displayed figure based on user input"""
    try:
        logger.info(f"Updating figure: slice={slice_idx}, zones={selected_zones}, opacity={opacity}")
        
        # Initialize loader with current data
        loader = ImageLoader(BASE_PATH)
        loader.load_data()
        
        return create_figure(loader, slice_idx, selected_zones, opacity)
        
    except Exception as e:
        logger.error(f"Error updating figure: {str(e)}")
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)