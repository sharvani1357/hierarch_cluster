import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    layout="wide"
)

# =====================================================
# DARK THEME STYLING
# =====================================================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
    background-attachment: fixed;
    color: #f1f5f9;
}

/* Main Card */
.block-container {
    background: rgba(30,41,59,0.8);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
}

/* Titles */
h1, h2, h3, h4 {
    color: #e2e8f0 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a,#1e293b,#0f172a);
    color: #f1f5f9;
}

section[data-testid="stSidebar"] label {
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#6366f1,#3b82f6);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(99,102,241,0.5);
}

/* Hide footer */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.title("🟣 News Topic Discovery Dashboard")
st.write("This system uses **Hierarchical Clustering** to group similar news articles based on textual similarity.")
st.markdown("👉 *Discover hidden themes without defining categories upfront.*")

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def detect_text_column(df):
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return obj_cols[0] if obj_cols else df.columns[0]

def clean_text_series(s):
    return s.fillna("").astype(str).str.replace(r"\s+"," ",regex=True).str.strip()

def compute_top_keywords(X, labels, feature_names, top_n=10):
    rows = []
    for c in np.unique(labels):
        idx = np.where(labels==c)[0]
        if len(idx)==0: continue
        mean_vec = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = mean_vec.argsort()[::-1][:top_n]
        keywords = [feature_names[i] for i in top_idx if mean_vec[i]>0]
        rows.append((int(c),len(idx),", ".join(keywords)))
    return pd.DataFrame(rows,columns=["Cluster","Articles","Top Keywords"])

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("🔧 Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

max_features = st.sidebar.slider("Max TF-IDF Features",100,2000,1000,50)
use_stopwords = st.sidebar.checkbox("Use English Stopwords",True)

ngram_choice = st.sidebar.selectbox("N-gram Range",
["Unigrams","Bigrams","Unigrams + Bigrams"])

if ngram_choice=="Unigrams":
    ngram_range=(1,1)
elif ngram_choice=="Bigrams":
    ngram_range=(2,2)
else:
    ngram_range=(1,2)

linkage_method = st.sidebar.selectbox(
"Linkage Method",
["ward","complete","average","single"]
)

subset_dendro = st.sidebar.slider("Articles for Dendrogram",20,200,100)

btn_dendro = st.sidebar.button("🟦 Generate Dendrogram")

k_clusters = st.sidebar.slider("Number of Clusters",2,12,4)
btn_cluster = st.sidebar.button("🟩 Apply Clustering")

# =====================================================
# DATA LOAD
# =====================================================
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file,encoding="utf-8")
    except:
        df = pd.read_csv(uploaded_file,encoding="latin1")
else:
    st.warning("Please upload a CSV dataset.")
    st.stop()

st.subheader("📌 Dataset Preview")
st.dataframe(df.head(),use_container_width=True)

# =====================================================
# TEXT COLUMN
# =====================================================
text_col = detect_text_column(df)
text_col = st.selectbox("Select Text Column",df.columns,
index=list(df.columns).index(text_col))

texts = clean_text_series(df[text_col])

# =====================================================
# TF-IDF
# =====================================================
vectorizer = TfidfVectorizer(
stop_words="english" if use_stopwords else None,
max_features=max_features,
ngram_range=ngram_range
)

X = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

st.success(f"TF-IDF Created: {X.shape}")

# =====================================================
# DENDROGRAM
# =====================================================
st.markdown("---")
st.subheader("🌳 Dendrogram")

if btn_dendro:
    n = min(subset_dendro,X.shape[0])
    X_sub = X[:n].toarray()

    Z = linkage(X_sub,method=linkage_method)

    fig, ax = plt.subplots(figsize=(12,5))
    dendrogram(Z,ax=ax,no_labels=True)
    ax.set_title(f"Dendrogram ({linkage_method.title()} Linkage)")
    ax.set_xlabel("Articles")
    ax.set_ylabel("Distance")
    st.pyplot(fig,use_container_width=True)
else:
    st.info("Click Generate Dendrogram to view clustering tree.")

# =====================================================
# CLUSTERING
# =====================================================
st.markdown("---")
st.subheader("🧩 Clustering Results")

if btn_cluster:

    model = AgglomerativeClustering(
        n_clusters=k_clusters,
        linkage=linkage_method,
        metric="euclidean"
    )

    labels = model.fit_predict(X.toarray())
    df_out = df.copy()
    df_out["Cluster"]=labels

    # Validation
    st.subheader("📊 Silhouette Score")
    try:
        sil = silhouette_score(X,labels)
        st.metric("Silhouette Score",f"{sil:.4f}")
    except:
        st.warning("Silhouette could not be computed.")

    # PCA Visualization
    st.subheader("📌 2D Cluster Visualization")
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(X.toarray())

    fig2, ax2 = plt.subplots(figsize=(6,5))
    ax2.scatter(X2d[:,0],X2d[:,1],c=labels)
    ax2.set_xlabel("PCA1")
    ax2.set_ylabel("PCA2")
    st.pyplot(fig2)

    # Summary
    st.subheader("📋 Cluster Summary")
    summary = compute_top_keywords(X,labels,feature_names)
    st.dataframe(summary,use_container_width=True)

    # Sample Headlines
    st.subheader("🧾 Sample Headlines per Cluster")
    for c in sorted(df_out["Cluster"].unique()):
        st.write(f"🟣 Cluster {c}")
        samples = df_out[df_out["Cluster"]==c][text_col].head(3)
        for s in samples:
            st.write("•",s[:160])

    # Business Interpretation
    st.subheader("💡 Business Interpretation")
    st.info(
        "Articles grouped in the same cluster share similar vocabulary and themes. "
        "These clusters help in automatic tagging, recommendations, and editorial organization."
    )

    st.download_button(
        "⬇ Download Clustered CSV",
        df_out.to_csv(index=False).encode("utf-8"),
        "news_clusters.csv",
        "text/csv"
    )

else:
    st.info("Set parameters and click Apply Clustering.")
