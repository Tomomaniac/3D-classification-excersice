import streamlit as st
import open3d as o3d
import numpy as np
import pickle
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="3D Object Classifier", page_icon="🎯", layout="wide")

st.title("🎯 3D Object Classifier")
st.markdown("Upload a 3D object (.off file) and the model will classify it!")

# Load the trained model
@st.cache_resource
def load_model():
    with open('model_enhanced.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    return model, classes

try:
    clf, classes = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.info(f"**Classes:** {', '.join(classes)}")
except:
    st.error("❌ Could not load model. Make sure 'model_enhanced.pkl' and 'classes.pkl' are in the same directory.")
    st.stop()

# Feature extraction function
def compute_enhanced_features(pcd):
    """Compute 77-dimensional enhanced features"""
    features = []
    points = np.asarray(pcd.points)
    
    # 1. FPFH features (66 dims)
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=100)
    )
    fpfh_global = np.concatenate([
        np.mean(fpfh.data, axis=1),
        np.std(fpfh.data, axis=1)
    ])
    features.append(fpfh_global)
    
    # 2. Bounding box dimensions (3 dims)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_dims = bbox.get_extent()
    features.append(bbox_dims)
    
    # 3. Volume metrics (2 dims)
    bbox_volume = np.prod(bbox_dims)
    try:
        hull = ConvexHull(points)
        convex_volume = hull.volume
    except:
        convex_volume = bbox_volume
    features.append([bbox_volume, convex_volume])
    
    # 4. Surface area (1 dim)
    try:
        hull = ConvexHull(points)
        surface_area = hull.area
    except:
        surface_area = 0.0
    features.append([surface_area])
    
    # 5. Compactness (1 dim)
    if surface_area > 0:
        compactness = convex_volume / surface_area
    else:
        compactness = 0.0
    features.append([compactness])
    
    # 6. Aspect ratios (3 dims)
    sorted_dims = np.sort(bbox_dims)[::-1]
    if sorted_dims[1] > 0 and sorted_dims[2] > 0:
        ratio_1_2 = sorted_dims[0] / sorted_dims[1]
        ratio_1_3 = sorted_dims[0] / sorted_dims[2]
        ratio_2_3 = sorted_dims[1] / sorted_dims[2]
    else:
        ratio_1_2 = ratio_1_3 = ratio_2_3 = 1.0
    features.append([ratio_1_2, ratio_1_3, ratio_2_3])
    
    # 7. Centroid height (1 dim)
    centroid = points.mean(axis=0)
    features.append([centroid[2]])
    
    return np.concatenate(features)

# File uploader
uploaded_file = st.file_uploader("Upload a 3D object (.off file)", type=['off'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.off", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("✅ File uploaded successfully!")
    
    # Process the object
    with st.spinner("🔄 Processing object..."):
        try:
            # Load mesh
            mesh = o3d.io.read_triangle_mesh("temp.off")
            pcd = mesh.sample_points_uniformly(number_of_points=2048)
            
            # Normalize
            points = np.asarray(pcd.points)
            centroid = points.mean(axis=0)
            points = points - centroid
            max_dist = np.linalg.norm(points, axis=1).max()
            points = points / (max_dist + 1e-12)
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Extract features
            features = compute_enhanced_features(pcd)
            
            # Predict
            prediction = clf.predict([features])[0]
            probabilities = clf.predict_proba([features])[0]
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎯 Prediction")
                st.markdown(f"## **{prediction.upper()}**")
                st.markdown(f"**Confidence:** {probabilities.max()*100:.1f}%")
                
                st.subheader("📊 All Class Probabilities")
                prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
                sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                
                for cls, prob in sorted_probs:
                    st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
            
            with col2:
                st.subheader("🎨 3D Visualization")
                
                # Create plotly 3D scatter
                points = np.asarray(pcd.points)
                fig = go.Figure(data=[go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=points[:, 2],
                        colorscale='Viridis',
                        showscale=True
                    )
                )])
                
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(showbackground=False),
                        yaxis=dict(showbackground=False),
                        zaxis=dict(showbackground=False),
                        aspectmode='data'
                    ),
                    height=500,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show geometric features
            with st.expander("📐 Geometric Features"):
                st.markdown("**Bounding Box Dimensions:**")
                st.write(f"- Width: {features[66]:.4f}")
                st.write(f"- Height: {features[67]:.4f}")
                st.write(f"- Depth: {features[68]:.4f}")
                
                st.markdown("**Volume & Surface:**")
                st.write(f"- Bounding box volume: {features[69]:.4f}")
                st.write(f"- Convex hull volume: {features[70]:.4f}")
                st.write(f"- Surface area: {features[71]:.4f}")
                st.write(f"- Compactness: {features[72]:.4f}")
                
                st.markdown("**Aspect Ratios:**")
                st.write(f"- Ratio 1:2: {features[73]:.4f}")
                st.write(f"- Ratio 1:3: {features[74]:.4f}")
                st.write(f"- Ratio 2:3: {features[75]:.4f}")
                
                st.write(f"**Centroid height:** {features[76]:.4f}")
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

else:
    st.info("👆 Upload a .off file to get started!")
    
    # Show example
    st.markdown("---")
    st.subheader("📚 About this app")
    st.markdown("""
    This app uses a **Random Forest classifier** trained on ModelNet10 dataset with:
    - **77 features** (66 FPFH + 11 geometric)
    - **68% accuracy** on 10 object classes
    - **500 training samples**
    
    **Supported classes:**
    - bathtub, bed, chair, desk, dresser
    - monitor, night_stand, sofa, table, toilet
    """)