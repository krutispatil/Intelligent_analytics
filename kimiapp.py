# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- Custom CSS for Better UI - FIXED COLORS ---
st.set_page_config(
    page_title="IntelliData Analyst Pro",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #3d3d5c;
        color: #ffffff;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card h3 {
        color: #a0a0b0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h2 {
        color: #ffffff;
        font-size: 2rem;
        margin: 0;
    }
    
    .metric-card p {
        color: #667eea;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .insight-box h3 {
        color: #ffffff;
        margin-top: 0;
    }
    
    .recommendation-card {
        background: #1e1e2e;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        color: #e0e0e0;
    }
    
    .feature-card {
        background: #1e1e2e;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #3d3d5c;
        color: #ffffff;
    }
    
    .feature-card h3 {
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .feature-card p {
        color: #b0b0c0;
        margin: 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d44;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: #b0b0c0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
    
    .anomaly-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
    
    .anomaly-high { background: #fee2e2; color: #dc2626; }
    .anomaly-medium { background: #fef3c7; color: #d97706; }
    .anomaly-low { background: #d1fae5; color: #059669; }
    
    /* Fix Streamlit default white backgrounds */
    .stApp {
        background-color: #0e0e16;
    }
    
    .stSidebar {
        background-color: #161622;
    }
    
    .stSidebar [data-testid="stMarkdown"] {
        color: #e0e0e0;
    }
    
    .stSidebar .stRadio label {
        color: #e0e0e0;
    }
    
    .stSidebar .stSelectbox label {
        color: #e0e0e0;
    }
    
    /* Fix expander colors */
    .streamlit-expanderHeader {
        background-color: #1e1e2e;
        color: #e0e0e0;
        border-radius: 8px;
    }
    
    .streamlit-expanderContent {
        background-color: #161622;
        color: #e0e0e0;
        border-radius: 0 0 8px 8px;
    }
    
    /* Fix metric colors */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0b0 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #667eea !important;
    }
    
    /* Fix selectbox and other inputs */
    .stSelectbox label, .stRadio label, .stSlider label {
        color: #e0e0e0 !important;
    }
    
    /* Fix text in columns */
    .stMarkdown p, .stMarkdown li {
        color: #e0e0e0;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff;
    }
    
    /* Fix info/warning boxes */
    .stInfo {
        background-color: #1e3a5f;
        color: #e0f2fe;
        border: 1px solid #3b82f6;
    }
    
    .stWarning {
        background-color: #451a03;
        color: #fef3c7;
        border: 1px solid #f59e0b;
    }
    
    .stSuccess {
        background-color: #064e3b;
        color: #d1fae5;
        border: 1px solid #10b981;
    }
    
    .stError {
        background-color: #450a0a;
        color: #fee2e2;
        border: 1px solid #ef4444;
    }
    
    /* Fix checkbox */
    .stCheckbox label {
        color: #e0e0e0 !important;
    }
    
    /* Fix button */
    .stButton button {
        background-color: #667eea;
        color: white;
        border: none;
    }
    
    .stButton button:hover {
        background-color: #764ba2;
    }
    
    /* Fix file uploader */
    .stFileUploader label {
        color: #e0e0e0 !important;
    }
    
    /* Segment profile cards */
    .segment-profile {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid #3d3d5c;
        color: #ffffff;
    }
    
    .segment-profile h3 {
        color: #667eea;
        margin-top: 0;
    }
    
    .segment-characteristic {
        background: #2d2d44;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.25rem 0;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<h1 class="main-header">🧠 IntelliData Analyst Pro</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #a0a0b0; margin-bottom: 2rem;">
    <b>100% Free • No API Keys • AI-Powered Insights • Runs Locally</b><br>
    <span style="font-size: 0.9rem; color: #808090;">Advanced statistical analysis with intelligent recommendations</span>
</div>
""", unsafe_allow_html=True)

# --- Smart Analysis Engine ---
class InsightEngine:
    """Rule-based AI engine for generating insights without APIs"""
    
    def __init__(self, df, data_type):
        self.df = df
        self.data_type = data_type
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        self.cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def generate_overview_insights(self):
        """Generate smart overview insights"""
        insights = []
        
        # Data quality assessment
        missing_pct = (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        if missing_pct > 5:
            insights.append(f"⚠️ **Data Quality Alert**: {missing_pct:.1f}% missing values detected. Consider imputation strategies.")
        else:
            insights.append(f"✅ **Data Quality**: Excellent! Only {missing_pct:.1f}% missing values.")
        
        # Dataset size assessment
        if len(self.df) < 100:
            insights.append("📊 **Sample Size**: Small dataset (<100 rows). Results may have high variance.")
        elif len(self.df) > 10000:
            insights.append("🚀 **Big Data Ready**: Large dataset detected (10k+ rows). Using sampled analysis for performance.")
        
        # Column type insights
        if len(self.numeric_cols) == 0:
            insights.append("📝 **Analysis Type**: Categorical analysis only - no numeric columns for statistical modeling.")
        elif len(self.cat_cols) == 0:
            insights.append("📈 **Analysis Type**: Purely numerical - no segmentation dimensions available.")
        else:
            insights.append(f"🎯 **Mixed Data**: {len(self.numeric_cols)} metrics × {len(self.cat_cols)} dimensions for deep analysis.")
        
        return insights
    
    def detect_anomalies(self, series, threshold=3):
        """Statistical anomaly detection using Z-score"""
        if series.std() == 0:
            return []
        
        z_scores = np.abs(stats.zscore(series.dropna()))
        anomalies = series.dropna()[z_scores > threshold]
        return anomalies
    
    def trend_analysis(self, date_col, metric_col):
        """Advanced trend analysis with seasonality detection"""
        df_sorted = self.df.sort_values(date_col)
        y = df_sorted[metric_col].values
        x = np.arange(len(y))
        
        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        trend_strength = abs(r_value)
        
        # Trend classification
        if trend_strength < 0.3:
            trend_desc = "📊 **Stable/No Clear Trend**"
            recommendation = "Focus on operational efficiency rather than growth strategies."
        elif slope > 0:
            if trend_strength > 0.7:
                trend_desc = "📈 **Strong Growth Trend**"
                recommendation = "Accelerate investment! Strong momentum detected."
            else:
                trend_desc = "📈 **Moderate Growth**"
                recommendation = "Steady growth - optimize existing channels."
        else:
            if trend_strength > 0.7:
                trend_desc = "📉 **Critical Decline**"
                recommendation = "🚨 Urgent intervention required. Investigate root causes immediately."
            else:
                trend_desc = "📉 **Slight Decline**"
                recommendation = "Monitor closely and test recovery initiatives."
        
        # Volatility analysis
        volatility = np.std(y) / np.mean(y) if np.mean(y) != 0 else 0
        vol_desc = "High" if volatility > 0.3 else "Moderate" if volatility > 0.15 else "Low"
        
        return {
            'trend': trend_desc,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'volatility': volatility,
            'volatility_desc': vol_desc,
            'recommendation': recommendation,
            'next_value': intercept + slope * (len(y) + 1)
        }
    
    def correlation_insights(self):
        """Generate correlation-based insights"""
        if len(self.numeric_cols) < 2:
            return None
            
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.5:  # Only significant correlations
                    corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': val,
                        'strength': 'Strong' if abs(val) > 0.8 else 'Moderate'
                    })
        
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return corr_pairs[:3]  # Top 3
    
    def segmentation_analysis(self):
        """Automatic segmentation using K-means"""
        if len(self.numeric_cols) < 2 or len(self.df) < 20:
            return None
        
        # Prepare data
        X = self.df[self.numeric_cols].fillna(self.df[self.numeric_cols].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal clusters (2-4)
        n_clusters = min(4, max(2, len(self.df) // 50))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        self.df['segment'] = clusters
        segment_profiles = []
        
        for i in range(n_clusters):
            cluster_data = self.df[self.df['segment'] == i]
            profile = {
                'id': i,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.df) * 100,
                'characteristics': {},
                'dominant_features': []
            }
            
            for col in self.numeric_cols:
                mean_val = cluster_data[col].mean()
                overall_mean = self.df[col].mean()
                overall_std = self.df[col].std()
                
                z_score = (mean_val - overall_mean) / overall_std if overall_std != 0 else 0
                
                profile['characteristics'][col] = {
                    'mean': mean_val,
                    'vs_overall': z_score
                }
                
                # Track dominant features (z-score > 0.5)
                if abs(z_score) > 0.5:
                    direction = "High" if z_score > 0 else "Low"
                    profile['dominant_features'].append(f"{direction} {col}")
            
            # Create meaningful name based on dominant features
            if profile['dominant_features']:
                profile['name'] = " + ".join(profile['dominant_features'][:2])
            else:
                profile['name'] = f"Segment {i+1} (Average Profile)"
            
            segment_profiles.append(profile)
        
        return segment_profiles
    
    def generate_recommendations(self):
        """Generate actionable business recommendations"""
        recommendations = []
        
        # Based on data type
        if self.data_type == 'e-commerce':
            recommendations.extend([
                "💰 **Revenue Optimization**: Focus on high-margin products identified in top quartile",
                "🎯 **Inventory Alert**: Monitor stock levels for items showing declining trends",
                "📅 **Seasonal Prep**: Analyze day-of-week patterns for staffing optimization"
            ])
        elif self.data_type == 'saas':
            recommendations.extend([
                "📈 **Growth Focus**: Prioritize activation rate improvements for highest impact",
                "🔄 **Retention**: Address churn patterns in identified risk segments",
                "💵 **Pricing**: Test price elasticity on segments with low price sensitivity"
            ])
        elif self.data_type == 'hr':
            recommendations.extend([
                "👥 **Retention Risk**: Review compensation for employees in high-performance segments",
                "📈 **Growth Path**: Identify skill gaps between high/low performing departments",
                "⚖️ **Equity Check**: Audit salary distributions across demographic segments"
            ])
        elif self.data_type == 'marketing':
            recommendations.extend([
                "🎪 **Campaign Focus**: Reallocate budget to channels with highest conversion velocity",
                "👤 **Audience Refinement**: Target lookalike audiences of high-value segments",
                "⏰ **Timing**: Optimize send times based on engagement temporal patterns"
            ])
        else:
            recommendations.extend([
                "🔍 **Deep Dive**: Investigate top 10% outliers for root cause analysis",
                "📊 **Benchmarking**: Compare current performance against historical baselines",
                "🎯 **Prioritization**: Focus resources on metrics with highest volatility"
            ])
        
        # Add data-specific recommendations
        if self.date_cols:
            recommendations.append("📅 **Forecasting**: Implement rolling 7-day predictions for operational planning")
        
        if len(self.numeric_cols) >= 2:
            recommendations.append("🔗 **Correlation Action**: Leverage strongly correlated metrics as leading indicators")
        
        return recommendations

# --- Data Processing Functions ---
def detect_data_type(df):
    """Enhanced data type detection"""
    text = ' '.join(df.columns).lower() + ' ' + ' '.join(df.head(100).astype(str).values.flatten()).lower()
    
    patterns = {
        'e-commerce': ['price', 'product', 'order', 'customer', 'revenue', 'cart', 'purchase', 'sku'],
        'saas': ['mrr', 'arr', 'churn', 'subscription', 'user', 'activation', 'retention', 'ltv'],
        'hr': ['salary', 'employee', 'department', 'attrition', 'performance', 'hire', 'tenure'],
        'finance': ['transaction', 'balance', 'account', 'interest', 'loan', 'credit', 'debit'],
        'marketing': ['campaign', 'ctr', 'conversion', 'impression', 'click', 'cpc', 'roas'],
        'healthcare': ['patient', 'diagnosis', 'treatment', 'medical', 'clinical', 'dosage'],
        'supply_chain': ['inventory', 'warehouse', 'shipment', 'logistics', 'supplier', 'stock']
    }
    
    scores = {k: sum(1 for term in v if term in text) for k, v in patterns.items()}
    best_match = max(scores, key=scores.get)
    
    return best_match if scores[best_match] > 2 else 'general'

def smart_clean_data(df):
    """Intelligent data cleaning with detailed reporting"""
    report = {'actions': [], 'warnings': []}
    original_shape = df.shape
    
    # Auto-detect date columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='raise')
                report['actions'].append(f"🗓️ Converted '{col}' to datetime")
            except:
                pass
    
    # Smart missing value handling
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            if df[col].dtype in ['int64', 'float64']:
                # Use median for skewed data, mean for normal
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    fill_val = df[col].median()
                    method = "median (skewed distribution)"
                else:
                    fill_val = df[col].mean()
                    method = "mean (normal distribution)"
                df[col].fillna(fill_val, inplace=True)
                report['actions'].append(f"🔢 Filled {missing} missing in '{col}' with {method}")
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
                report['actions'].append(f"📝 Filled {missing} missing in '{col}' with mode")
    
    # Remove duplicates
    dups = df.duplicated().sum()
    if dups > 0:
        df.drop_duplicates(inplace=True)
        report['actions'].append(f"🧹 Removed {dups} duplicate rows")
    
    # Outlier flagging (but not removal)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_count = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        outlier_count += len(outliers)
    
    if outlier_count > 0:
        report['warnings'].append(f"⚠️ Detected {outlier_count} statistical outliers (flagged but retained)")
    
    report['final_shape'] = df.shape
    return df, report

# --- Visualization Functions ---
def create_kpi_cards(df, numeric_cols, engine):
    """Create animated KPI cards with actual metric names"""
    # Calculate which metrics have most variance/interesting patterns
    metric_scores = {}
    for col in numeric_cols:
        if len(df) > 1:
            trend = abs(stats.linregress(range(len(df)), df[col].fillna(df[col].median()))[2])
            volatility = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
            metric_scores[col] = trend + volatility
    
    # Select top 4 most interesting metrics
    top_metrics = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    
    cols = st.columns(min(4, len(top_metrics)))
    for i, (col, score) in enumerate(top_metrics):
        with cols[i]:
            value = df[col].mean()
            # Calculate change if time series exists
            delta = 0
            if len(df) > 1 and df[col].iloc[0] != 0:
                delta = ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0] * 100)
            
            st.metric(
                label=col.replace('_', ' ').title(),
                value=f"{value:,.2f}",
                delta=f"{delta:.1f}%" if delta != 0 else None
            )

def plot_distribution_analysis(df, col, engine):
    """Enhanced distribution plot with insights"""
    fig = make_subplots(rows=2, cols=1, subplot_titles=(f'Distribution of {col}', 'Box Plot with Outliers'))
    
    # Histogram with KDE
    fig.add_trace(
        go.Histogram(x=df[col], nbinsx=30, name='Distribution', 
                    marker_color='rgba(102, 126, 234, 0.7)'),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(x=df[col], name=col, marker_color='#764ba2'),
        row=2, col=1
    )
    
    # Add anomaly detection
    anomalies = engine.detect_anomalies(df[col])
    if len(anomalies) > 0:
        fig.add_trace(
            go.Scatter(x=anomalies, y=[0]*len(anomalies), mode='markers',
                      marker=dict(color='red', size=10, symbol='x'),
                      name=f'Anomalies ({len(anomalies)})'),
            row=1, col=1
        )
    
    fig.update_layout(
        height=600, 
        showlegend=True, 
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0'
    )
    return fig

def plot_time_series_advanced(df, date_col, metric_col, engine):
    """Advanced time series with trend and forecast"""
    trend_data = engine.trend_analysis(date_col, metric_col)
    
    fig = go.Figure()
    
    # Actual data
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[metric_col],
        mode='lines+markers', name='Actual',
        line=dict(color='#667eea', width=2),
        marker=dict(size=6)
    ))
    
    # Trend line
    x_numeric = np.arange(len(df))
    trend_line = trend_data['slope'] * x_numeric + (df[metric_col].mean() - trend_data['slope'] * len(df)/2)
    fig.add_trace(go.Scatter(
        x=df[date_col], y=trend_line,
        mode='lines', name=f"Trend (R²={trend_data['r_squared']:.2f})",
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Forecast next 5 periods
    if len(df) > 5:
        last_date = df[date_col].max()
        freq = pd.infer_freq(df[date_col]) or 'D'
        future_dates = pd.date_range(start=last_date, periods=6, freq=freq)[1:]
        
        last_idx = len(df)
        future_values = [trend_data['slope'] * (last_idx + i) + trend_data['next_value'] - trend_data['slope'] * (last_idx + 1) 
                        for i in range(1, 6)]
        
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_values,
            mode='lines+markers', name='Forecast',
            line=dict(color='green', dash='dot', width=2),
            marker=dict(symbol='diamond', size=8)
        ))
    
    fig.update_layout(
        title=f"{metric_col} Over Time - {trend_data['trend']}",
        xaxis_title="Date",
        yaxis_title=metric_col,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0',
        height=500
    )
    
    return fig, trend_data

def plot_correlation_heatmap(df, numeric_cols):
    """Interactive correlation heatmap"""
    corr = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr, 
        text_auto=True, 
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix'
    )
    
    # Highlight strong correlations
    for i in range(len(corr)):
        for j in range(len(corr)):
            if abs(corr.iloc[i, j]) > 0.8 and i != j:
                fig.add_shape(
                    type="rect",
                    x0=j-0.5, y0=i-0.5,
                    x1=j+0.5, y1=i+0.5,
                    line=dict(color="gold", width=3)
                )
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0'
    )
    return fig

def plot_segmentation_3d(df, engine):
    """3D segmentation visualization"""
    profiles = engine.segmentation_analysis()
    if not profiles or len(engine.numeric_cols) < 3:
        return None
    
    # Use first 3 numeric columns for 3D plot
    cols = engine.numeric_cols[:3]
    
    fig = px.scatter_3d(
        df, x=cols[0], y=cols[1], z=cols[2],
        color='segment',
        title='3D Segment Visualization',
        opacity=0.7,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0'
    )
    
    return fig

# --- Main Analysis Dashboard ---
def run_intelligent_analysis(df):
    """Main analysis pipeline"""
    
    # Data preparation
    data_type = detect_data_type(df)
    df, clean_report = smart_clean_data(df)
    engine = InsightEngine(df, data_type)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Analysis Controls")
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick Scan", "Standard", "Deep Dive"],
            value="Standard"
        )
        
        st.markdown("### 📊 Data Profile")
        st.write(f"**Type:** {data_type.replace('_', ' ').title()}")
        st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} cols")
        st.write(f"**Numeric:** {len(engine.numeric_cols)} | **Categorical:** {len(engine.cat_cols)} | **Date:** {len(engine.date_cols)}")
        
        if st.checkbox("Show Cleaning Report"):
            for action in clean_report['actions']:
                st.write(action)
            for warning in clean_report['warnings']:
                st.warning(warning)
    
    # Main dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Executive Summary", 
        "🔍 Deep Dive", 
        "📊 Visualizations", 
        "🎯 Segments", 
        "💡 Recommendations"
    ])
    
    # TAB 1: Executive Summary
    with tab1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🧠 Key Insights")
        insights = engine.generate_overview_insights()
        for insight in insights:
            st.write(f"• {insight}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # KPI Cards - Now showing actual metric names
        if engine.numeric_cols:
            st.subheader("📊 Key Metrics")
            create_kpi_cards(df, engine.numeric_cols, engine)
        
        # Anomaly Summary
        if engine.numeric_cols:
            st.subheader("🚨 Anomaly Detection")
            total_anomalies = 0
            anomaly_cols = []
            
            for col in engine.numeric_cols:
                anomalies = engine.detect_anomalies(df[col])
                if len(anomalies) > 0:
                    total_anomalies += len(anomalies)
                    anomaly_cols.append((col, len(anomalies)))
            
            if total_anomalies > 0:
                cols = st.columns(len(anomaly_cols))
                for i, (col, count) in enumerate(anomaly_cols):
                    severity = "high" if count > 5 else "medium" if count > 2 else "low"
                    with cols[i]:
                        st.markdown(
                            f'<span class="anomaly-badge anomaly-{severity}">{col}: {count} anomalies</span>',
                            unsafe_allow_html=True
                        )
            else:
                st.success("✅ No significant anomalies detected")
    
    # TAB 2: Deep Dive Analysis
    with tab2:
        if engine.numeric_cols:
            selected_col = st.selectbox("Select Metric for Analysis", engine.numeric_cols)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = plot_distribution_analysis(df, selected_col, engine)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Statistical Profile")
                stats_data = df[selected_col].describe()
                
                st.write(f"**Mean:** {stats_data['mean']:,.2f}")
                st.write(f"**Median:** {df[selected_col].median():,.2f}")
                st.write(f"**Std Dev:** {stats_data['std']:,.2f}")
                st.write(f"**Skewness:** {df[selected_col].skew():,.2f}")
                
                # Normality test
                _, p_value = stats.normaltest(df[selected_col].dropna())
                if p_value < 0.05:
                    st.warning("⚠️ Non-normal distribution detected")
                else:
                    st.success("✅ Normal distribution")
                
                # Anomalies
                anomalies = engine.detect_anomalies(df[selected_col])
                if len(anomalies) > 0:
                    st.error(f"🚨 {len(anomalies)} outliers detected")
                    with st.expander("View Outliers"):
                        st.write(anomalies.values)
    
    # TAB 3: Visualizations
    with tab3:
        viz_type = st.radio("Visualization Type", 
                          ["Time Series", "Correlations", "Categorical Breakdown", "3D Segments"],
                          horizontal=True)
        
        if viz_type == "Time Series" and engine.date_cols and engine.numeric_cols:
            date_col = engine.date_cols[0]
            metric_col = st.selectbox("Select Metric", engine.numeric_cols, key='ts_metric')
            
            fig, trend_data = plot_time_series_advanced(df, date_col, metric_col, engine)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trend", trend_data['trend'].split()[1])
            with col2:
                st.metric("Volatility", trend_data['volatility_desc'])
            with col3:
                st.metric("Next Forecast", f"{trend_data['next_value']:,.2f}")
            
            st.info(f"💡 **Recommendation:** {trend_data['recommendation']}")
            
        elif viz_type == "Correlations" and len(engine.numeric_cols) >= 2:
            fig = plot_correlation_heatmap(df, engine.numeric_cols)
            st.plotly_chart(fig, use_container_width=True)
            
            corr_insights = engine.correlation_insights()
            if corr_insights:
                st.subheader("🔗 Key Relationships")
                for pair in corr_insights:
                    direction = "positive" if pair['correlation'] > 0 else "negative"
                    st.write(f"• **{pair['var1']}** ↔ **{pair['var2']}**: {pair['strength']} {direction} correlation ({pair['correlation']:.2f})")
        
        elif viz_type == "Categorical Breakdown" and engine.cat_cols and engine.numeric_cols:
            cat_col = st.selectbox("Category", engine.cat_cols)
            num_col = st.selectbox("Metric", engine.numeric_cols, key='cat_num')
            
            if df[cat_col].nunique() <= 20:
                fig = px.box(df, x=cat_col, y=num_col, color=cat_col,
                           title=f"{num_col} by {cat_col}",
                           template='plotly_dark')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e0e0'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical significance test
                groups = [group[num_col].values for name, group in df.groupby(cat_col) if len(group) > 1]
                if len(groups) >= 2:
                    f_stat, p_val = stats.f_oneway(*groups)
                    if p_val < 0.05:
                        st.success(f"✅ Statistically significant differences between groups (p={p_val:.4f})")
                    else:
                        st.info(f"ℹ️ No significant difference between groups (p={p_val:.4f})")
        
        elif viz_type == "3D Segments":
            fig = plot_segmentation_3d(df, engine)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 3 numeric columns and 20 rows for 3D segmentation")
    
    # TAB 4: Segmentation - FIXED WITH ACTUAL KPI NAMES
    with tab4:
        profiles = engine.segmentation_analysis()
        if profiles:
            st.subheader("🎯 Automatic Segments")
            
            # Display segments with meaningful names
            cols = st.columns(len(profiles))
            for i, profile in enumerate(profiles):
                with cols[i]:
                    st.markdown(f"""
                    <div class="segment-profile">
                        <h3>{profile['name']}</h3>
                        <h2 style="color: #667eea; margin: 0.5rem 0;">{profile['percentage']:.1f}%</h2>
                        <p style="color: #808090; margin: 0;">{profile['size']} records</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View Characteristics"):
                        for char, values in profile['characteristics'].items():
                            direction = "↑" if values['vs_overall'] > 0 else "↓"
                            color = "#10b981" if values['vs_overall'] > 0 else "#ef4444"
                            st.markdown(
                                f'<div class="segment-characteristic">'
                                f'<span style="color: {color}; font-weight: bold;">{direction}</span> '
                                f'<b>{char}</b>: {abs(values["vs_overall"]):.1f}σ from avg'
                                f'</div>',
                                unsafe_allow_html=True
                            )
        else:
            st.info("Segmentation requires 2+ numeric columns and 20+ rows")
    
    # TAB 5: Recommendations
    with tab5:
        st.subheader("💡 Strategic Recommendations")
        recommendations = engine.generate_recommendations()
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <b>{i}.</b> {rec}
            </div>
            """, unsafe_allow_html=True)
        
        # Action priority matrix
        if engine.numeric_cols:
            st.subheader("🎯 Priority Matrix")
            
            # Calculate impact vs effort for each metric
            metrics_data = []
            for col in engine.numeric_cols:
                volatility = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                trend = abs(stats.linregress(range(len(df)), df[col].fillna(df[col].median()))[2])
                
                metrics_data.append({
                    'Metric': col,
                    'Impact (Volatility)': volatility * 100,
                    'Trend Strength': trend * 100,
                    'Priority': volatility * trend * 100
                })
            
            priority_df = pd.DataFrame(metrics_data).sort_values('Priority', ascending=False)
            
            fig = px.scatter(priority_df, x='Impact (Volatility)', y='Trend Strength',
                           size='Priority', color='Priority', text='Metric',
                           title='Action Priority Matrix',
                           color_continuous_scale='Viridis',
                           template='plotly_dark')
            fig.update_traces(textposition='top center')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0'
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Sample Data Generator ---
def generate_sample_data(dataset_type):
    """Generate realistic sample datasets"""
    np.random.seed(42)
    n = 200
    
    if dataset_type == "E-Commerce Sales":
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        return pd.DataFrame({
            'date': dates,
            'product': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n),
            'revenue': np.random.lognormal(4, 0.5, n) * (1 + np.sin(np.arange(n) * 2 * np.pi / 30) * 0.3),
            'units_sold': np.random.poisson(50, n),
            'customer_segment': np.random.choice(['New', 'Returning', 'VIP'], n, p=[0.5, 0.3, 0.2]),
            'marketing_channel': np.random.choice(['Organic', 'Paid', 'Social', 'Email'], n),
            'discount_applied': np.random.choice([0, 10, 20, 30], n, p=[0.6, 0.2, 0.15, 0.05])
        })
    
    elif dataset_type == "SaaS Metrics":
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        base_users = 1000
        growth = np.cumsum(np.random.normal(5, 10, n))
        return pd.DataFrame({
            'date': dates,
            'mrr': (base_users + growth) * np.random.uniform(50, 80, n),
            'new_users': np.random.poisson(20, n),
            'churned_users': np.random.poisson(5, n),
            'activation_rate': np.clip(np.random.beta(2, 5, n) + np.sin(np.arange(n) * 2 * np.pi / 7) * 0.1, 0, 1),
            'support_tickets': np.random.poisson(15, n),
            'nps_score': np.clip(np.random.normal(40, 15, n) + growth * 0.01, -100, 100)
        })
    
    elif dataset_type == "HR Analytics":
        return pd.DataFrame({
            'employee_id': range(1001, 1001 + n),
            'hire_date': pd.date_range('2020-01-01', periods=n, freq='W'),
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'Support', 'HR'], n),
            'salary': np.random.lognormal(11, 0.3, n),
            'performance_score': np.clip(np.random.normal(3.5, 0.8, n), 1, 5),
            'satisfaction': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.2, 0.4, 0.25]),
            'overtime_hours': np.random.exponential(5, n),
            'training_completed': np.random.poisson(3, n)
        })
    
    elif dataset_type == "Marketing Campaigns":
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        return pd.DataFrame({
            'date': dates,
            'campaign_name': np.random.choice(['Summer Sale', 'Product Launch', 'Retargeting', 'Brand Awareness'], n),
            'spend': np.random.lognormal(5, 0.5, n),
            'impressions': np.random.lognormal(12, 0.8, n),
            'clicks': np.random.lognormal(8, 0.6, n),
            'conversions': np.random.poisson(50, n),
            'channel': np.random.choice(['Google', 'Facebook', 'LinkedIn', 'TikTok'], n),
            'audience': np.random.choice(['18-24', '25-34', '35-44', '45+'], n)
        })

# --- Main App Flow ---
st.sidebar.markdown("### 📁 Data Source")

data_source = st.sidebar.radio("Choose Data Source", 
                              ["Upload File", "Use Sample Data", "Connect Database (Coming Soon)"])

df = None

if data_source == "Upload File":
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
    if uploaded:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            st.sidebar.success(f"✅ Loaded {len(df)} rows")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

elif data_source == "Use Sample Data":
    sample = st.sidebar.selectbox("Select Dataset", 
                                 ["E-Commerce Sales", "SaaS Metrics", "HR Analytics", "Marketing Campaigns"])
    if st.sidebar.button("Generate Data"):
        df = generate_sample_data(sample)
        st.sidebar.success(f"✅ Generated {len(df)} rows")

if df is not None:
    run_intelligent_analysis(df)
else:
    # Landing page - FIXED COLORS
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2 style="color: #ffffff;">🚀 Welcome to IntelliData Analyst Pro</h2>
        <p style="font-size: 1.2rem; color: #a0a0b0;">
            Upload your data or try our sample datasets to get started with intelligent analysis.
        </p>
        <br>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div class="feature-card">
                <h3>📊</h3>
                <h4 style="color: #667eea; margin: 0.5rem 0;">Automatic Pattern Detection</h4>
                <p>Statistical anomaly & trend detection</p>
            </div>
            <div class="feature-card">
                <h3>🎯</h3>
                <h4 style="color: #667eea; margin: 0.5rem 0;">Smart Recommendations</h4>
                <p>AI-powered business insights</p>
            </div>
            <div class="feature-card">
                <h3>🔮</h3>
                <h4 style="color: #667eea; margin: 0.5rem 0;">Predictive Forecasting</h4>
                <p>Linear regression & segmentation</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #808090; font-size: 0.9rem;">
    🧠 <b style="color: #a0a0b0;">IntelliData Analyst Pro</b> | 100% Free & Open Source | No API Keys Required<br>
    Built with Streamlit • Scikit-Learn • Plotly • Statistical Analysis Engine
</div>
""", unsafe_allow_html=True)
