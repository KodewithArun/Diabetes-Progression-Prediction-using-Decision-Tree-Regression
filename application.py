import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.stAlert {
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Cache data loading and model training
@st.cache_data
def load_data():
    """Load and prepare the diabetes dataset"""
    dataset = load_diabetes()
    feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    df = pd.DataFrame(dataset.data, columns=feature_names)
    df['target'] = dataset.target
    return df, dataset

@st.cache_resource
def train_model(X_train, y_train):
    """Train the optimized decision tree model"""
    param_grid = {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Progression Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df, dataset = load_data()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Overview", "📊 Data Analysis", "🤖 Model Training", "🔮 Predictions", "📈 Model Insights"]
    )
    
    if page == "🏠 Overview":
        show_overview(df, dataset)
    elif page == "📊 Data Analysis":
        show_data_analysis(df, X_train, y_train)
    elif page == "🤖 Model Training":
        show_model_training(X_train, X_test, y_train, y_test)
    elif page == "🔮 Predictions":
        show_predictions(X_train, y_train, X.columns)
    elif page == "📈 Model Insights":
        show_model_insights(X_train, y_train, X.columns)

def show_overview(df, dataset):
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Target Mean", f"{df['target'].mean():.1f}")
    with col4:
        st.metric("Target Std", f"{df['target'].std():.1f}")
    
    st.subheader("Dataset Description")
    st.info("""
    This dataset contains information about diabetes progression in patients. 
    The target variable represents a quantitative measure of disease progression one year after baseline.
    All features have been standardized and centered.
    """)
    
    # Feature descriptions
    feature_descriptions = {
        'age': 'Age of the patient',
        'sex': 'Gender of the patient',
        'bmi': 'Body Mass Index',
        'bp': 'Average Blood Pressure',
        's1': 'Total Cholesterol',
        's2': 'Low-Density Lipoproteins',
        's3': 'High-Density Lipoproteins',
        's4': 'Total Cholesterol / HDL',
        's5': 'Log of Serum Triglycerides',
        's6': 'Blood Sugar Level'
    }
    
    st.subheader("Feature Descriptions")
    feature_df = pd.DataFrame([
        {"Feature": k, "Description": v} for k, v in feature_descriptions.items()
    ])
    st.dataframe(feature_df, use_container_width=True)
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

def show_data_analysis(df, X_train, y_train):
    st.header("📊 Exploratory Data Analysis")
    
    # Target distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(y_train, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Diabetes Progression Distribution')
        ax.set_xlabel('Progression Score')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Target Statistics")
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Value': [
                y_train.mean(),
                y_train.median(),
                y_train.std(),
                y_train.min(),
                y_train.max()
            ]
        })
        st.dataframe(stats_df.round(2), use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = X_train.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='RdYlBu_r',
        center=0,
        mask=mask,
        square=True,
        fmt='.2f',
        ax=ax
    )
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    selected_features = st.multiselect(
        "Select features to visualize:",
        X_train.columns.tolist(),
        default=['bmi', 'bp', 's1']
    )
    
    if selected_features:
        fig, axes = plt.subplots(1, len(selected_features), figsize=(5*len(selected_features), 4))
        if len(selected_features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(selected_features):
            axes[i].hist(X_train[feature], bins=20, alpha=0.7, color='lightcoral')
            axes[i].set_title(f'{feature.upper()} Distribution')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)

def show_model_training(X_train, X_test, y_train, y_test):
    st.header("🤖 Model Training & Evaluation")
    
    # Train model
    with st.spinner("Training model with hyperparameter optimization..."):
        model = train_model(X_train, y_train)
    
    # Display best parameters
    st.subheader("Optimal Hyperparameters")
    params_df = pd.DataFrame([
        {"Parameter": k, "Value": v} for k, v in model.best_params_.items()
    ])
    st.dataframe(params_df, use_container_width=True)
    
    # Model performance
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("MAE", f"{mae:.2f}")
    with col3:
        st.metric("MSE", f"{mse:.2f}")
    with col4:
        st.metric("RMSE", f"{rmse:.2f}")
    
    # Performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predictions vs Actual")
        fig = px.scatter(
            x=y_test, y=y_pred,
            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
            title='Model Predictions vs Actual Values'
        )
        fig.add_shape(
            type="line",
            x0=y_test.min(), y0=y_test.min(),
            x1=y_test.max(), y1=y_test.max(),
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Residual Analysis")
        residuals = y_test - y_pred
        fig = px.scatter(
            x=y_pred, y=residuals,
            labels={'x': 'Predicted Values', 'y': 'Residuals'},
            title='Residual Plot'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

def show_predictions(X_train, y_train, feature_names):
    st.header("🔮 Make Predictions")
    
    # Train the model for predictions
    model = train_model(X_train, y_train)
    
    st.subheader("Enter Patient Information")
    st.info("All values should be standardized. Use the dataset statistics as reference.")
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", -3.0, 3.0, 0.0, 0.1)
        sex = st.slider("Sex", -3.0, 3.0, 0.0, 0.1)
        bmi = st.slider("BMI", -3.0, 3.0, 0.0, 0.1)
        bp = st.slider("Blood Pressure", -3.0, 3.0, 0.0, 0.1)
        s1 = st.slider("Total Cholesterol", -3.0, 3.0, 0.0, 0.1)
    
    with col2:
        s2 = st.slider("LDL Cholesterol", -3.0, 3.0, 0.0, 0.1)
        s3 = st.slider("HDL Cholesterol", -3.0, 3.0, 0.0, 0.1)
        s4 = st.slider("Total Cholesterol / HDL", -3.0, 3.0, 0.0, 0.1)
        s5 = st.slider("Log Triglycerides", -3.0, 3.0, 0.0, 0.1)
        s6 = st.slider("Blood Sugar", -3.0, 3.0, 0.0, 0.1)
    
    # Make prediction
    if st.button("🔍 Predict Diabetes Progression", type="primary"):
        input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown("### Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="text-align: center; color: #1f77b4;">
                    Predicted Progression Score: {prediction:.1f}
                </h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Interpretation
        if prediction < 100:
            interpretation = "Lower than average diabetes progression"
            color = "green"
        elif prediction < 200:
            interpretation = "Moderate diabetes progression"
            color = "orange"
        else:
            interpretation = "Higher than average diabetes progression"
            color = "red"
        
        st.markdown(f"**Interpretation:** :{color}[{interpretation}]")
        
        # Feature contribution visualization
        st.subheader("Feature Contribution Analysis")
        feature_values = [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': feature_values
        })
        
        fig = px.bar(
            feature_df,
            x='Feature',
            y='Value',
            title='Input Feature Values',
            color='Value',
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_insights(X_train, y_train, feature_names):
    st.header("📈 Model Insights")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.best_estimator_.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Decision Tree Model',
        color='Importance',
        color_continuous_scale='viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Top features table
    st.subheader("Top 5 Most Important Features")
    st.dataframe(importance_df.head().round(4), use_container_width=True)
    
    # Model parameters
    st.subheader("Best Model Parameters")
    params_df = pd.DataFrame([
        {"Parameter": k, "Value": str(v)} for k, v in model.best_params_.items()
    ])
    st.dataframe(params_df, use_container_width=True)
    
    # Cross-validation score
    st.subheader("Model Performance")
    cv_score = -model.best_score_
    st.metric("Cross-Validation MSE", f"{cv_score:.2f}")
    
    # Decision tree visualization
    st.subheader("Decision Tree Structure")
    st.info("Showing simplified tree structure (depth limited for readability)")
    
    # Create a simplified tree for visualization
    simple_tree = DecisionTreeRegressor(
        max_depth=3,
        random_state=42,
        **{k: v for k, v in model.best_params_.items() if k != 'max_depth'}
    )
    simple_tree.fit(X_train, y_train)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    tree.plot_tree(
        simple_tree,
        filled=True,
        feature_names=feature_names,
        rounded=True,
        fontsize=10
    )
    ax.set_title('Decision Tree Visualization (Depth Limited to 3)')
    st.pyplot(fig)

# Sidebar information
def add_sidebar_info():
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    This app uses machine learning to predict diabetes progression based on patient characteristics.
    
    **Model:** Decision Tree Regressor with hyperparameter optimization
    
    **Features:** Age, Sex, BMI, Blood Pressure, and various serum measurements
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write("**Source:** Scikit-learn Diabetes Dataset")
    st.sidebar.write("**Samples:** 442 patients")
    st.sidebar.write("**Features:** 10 standardized physiological variables")

# Run the app
if __name__ == "__main__":
    add_sidebar_info()
    main()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with Streamlit • Diabetes Progression Predictor • Machine Learning Analysis</p>
    </div>
    """, unsafe_allow_html=True)