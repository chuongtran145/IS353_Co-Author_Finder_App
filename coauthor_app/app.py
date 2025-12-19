import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import math
from xgboost import XGBClassifier
from networkx.algorithms.community import louvain_communities, asyn_lpa_communities

# Cấu hình trang
st.set_page_config(page_title="Co-Author Recommendation", layout="wide")

# ==========================================
# 1. HÀM XỬ LÝ DATA & TRAINING (CORE LOGIC)
# ==========================================

@st.cache_resource
def train_pipeline(uploaded_file):
    """
    Hàm này thực hiện toàn bộ quy trình:
    Load Data -> Build Graph -> Feature Eng -> Train XGBoost
    Được cache lại để không chạy lại mỗi lần reload trang.
    """
    with st.spinner('Đang xử lý dữ liệu và huấn luyện mô hình... (Có thể mất vài phút)'):
        # 1.1 Load Data
        edges = []
        # Đọc file upload (bytes -> string)
        content = uploaded_file.getvalue().decode("utf-8")
        for line in content.splitlines():
            if line.startswith("#"): continue
            parts = line.strip().split()
            if len(parts) < 2: continue
            u, v = int(parts[0]), int(parts[1])
            if u == v: continue
            if u > v: u, v = v, u
            edges.append((u, v))
        
        edges = list(set(edges)) # Remove duplicates
        
        # 1.2 Build Graph
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # Core filtering (k=1 như trong notebook)
        G_core = nx.k_core(G, k=1)
        
        # Split Train/Test (Ở đây ta dùng toàn bộ G_core làm G_train để demo cho đầy đủ dữ liệu)
        # Trong thực tế production, ta sẽ train trên toàn bộ dữ liệu hiện có
        G_train = G_core 
        
        # 1.3 Community Detection
        # Louvain
        louvain_comms = louvain_communities(G_train, seed=42)
        louvain_map = {}
        for idx, comm in enumerate(louvain_comms):
            for node in comm:
                louvain_map[node] = idx
                
        # Label Propagation
        lpa_comms = asyn_lpa_communities(G_train, seed=42)
        lpa_map = {}
        for idx, comm in enumerate(lpa_comms):
            for node in comm:
                lpa_map[node] = idx
        
        # 1.4 Generate Training Data (Simplified for Demo Speed)
        # Lấy mẫu 1 phần để train cho nhanh (hoặc train full nếu server mạnh)
        # Ở đây mình tái tạo logic feature extraction để train model
        
        # Sinh mẫu Positive
        train_edges = list(G_train.edges())
        # Giới hạn số lượng mẫu train để demo chạy nhanh (ví dụ 10k mẫu pos)
        # Nếu muốn chính xác như notebook gốc thì bỏ đoạn slice [:10000]
        sample_pos = train_edges
        
        X = []
        y = []
        
        # Hàm tính feature cho 1 cặp (u, v)
        def compute_features(u, v, graph):
            # Basic sets
            neu = set(graph.neighbors(u))
            nev = set(graph.neighbors(v))
            common = neu.intersection(nev)
            union_set = neu.union(nev)
            
            # 1. Common Neighbors
            cn = len(common)
            
            # 2. Jaccard
            jaccard = cn / len(union_set) if len(union_set) > 0 else 0
            
            # 3. Adamic-Adar & 4. Resource Allocation
            aa = 0
            ra = 0
            for w in common:
                deg_w = graph.degree(w)
                if deg_w > 1:
                    aa += 1 / math.log(deg_w)
                if deg_w > 0:
                    ra += 1 / deg_w
            
            # 5. Preferential Attachment
            du = graph.degree(u)
            dv = graph.degree(v)
            pa = du * dv
            
            # 6. Community
            same_louvain = 1 if louvain_map.get(u, -1) == louvain_map.get(v, -2) else 0
            same_lpa = 1 if lpa_map.get(u, -1) == lpa_map.get(v, -2) else 0
            
            return [cn, aa, ra, jaccard, pa, du, dv, same_louvain, same_lpa]

        # Tạo dữ liệu Positive
        for u, v in sample_pos:
            feats = compute_features(u, v, G_train)
            X.append(feats)
            y.append(1)
            
        # Tạo dữ liệu Negative (Ratio 1:1 cho nhanh, notebook là 1:5)
        num_neg = len(sample_pos)
        cnt = 0
        nodes_list = list(G_train.nodes())
        while cnt < num_neg:
            u_rnd = np.random.choice(nodes_list)
            v_rnd = np.random.choice(nodes_list)
            if u_rnd != v_rnd and not G_train.has_edge(u_rnd, v_rnd):
                feats = compute_features(u_rnd, v_rnd, G_train)
                X.append(feats)
                y.append(0)
                cnt += 1
                
        # 1.5 Train Model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(np.array(X), np.array(y))
        
        return model, G_train, louvain_map, lpa_map

# ==========================================
# 2. UI CHÍNH
# ==========================================

st.title("🔎 Co-Author Finder System")
st.markdown("""
Hệ thống gợi ý đồng tác giả dựa trên **XGBoost** và **Graph Mining**.
Upload dataset (định dạng edge list `.txt`) để bắt đầu.
""")

# Sidebar: Upload Dataset
with st.sidebar:
    st.header("1. Dataset Selection")
    uploaded_file = st.file_uploader("Chọn file dataset (VD: ca-HepPh.txt)", type=['txt'])
    
    st.info("Format: File text, mỗi dòng là `u v` hoặc `u \t v`. Dòng bắt đầu bằng # sẽ bị bỏ qua.")

if uploaded_file is not None:
    # Trigger pipeline
    try:
        model, G_train, louvain_map, lpa_map = train_pipeline(uploaded_file)
        st.success(f"✅ Đã train xong mô hình! Số lượng Nodes: {G_train.number_of_nodes()}, Edges: {G_train.number_of_edges()}")
        
        # Main Area: Input Author ID
        st.header("2. Link Prediction / Recommendation")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            # Chọn 1 ID có sẵn để demo cho tiện
            example_id = list(G_train.nodes())[0]
            target_u = st.number_input("Nhập ID Tác giả (Author ID):", min_value=0, value=example_id)
            top_k = st.slider("Số lượng gợi ý (Top-k):", 5, 50, 10)
            btn_run = st.button("🚀 Gợi ý Đồng tác giả")

        if btn_run:
            if target_u not in G_train:
                st.error(f"Author ID {target_u} không tồn tại trong đồ thị!")
            else:
                with st.spinner(f"Đang tìm kiếm ứng viên 2-hop cho {target_u}..."):
                    # --- BƯỚC INFERENCE ---
                    
                    # 1. Tìm ứng viên 2-hop (Neighbors of Neighbors)
                    neighbors = set(G_train.neighbors(target_u))
                    candidates = set()
                    for n in neighbors:
                        candidates.update(G_train.neighbors(n))
                    
                    # Loại bỏ chính nó và các neighbor trực tiếp (đã là co-author rồi)
                    candidates.discard(target_u)
                    candidates = list(candidates - neighbors)
                    
                    if not candidates:
                        st.warning("Không tìm thấy ứng viên 2-hop nào (Author này có thể bị cô lập hoặc đã kết nối hết).")
                    else:
                        # 2. Tính feature cho candidates
                        # Copy logic compute_features từ trên xuống để dùng cho inference
                        X_pred = []
                        valid_candidates = []
                        
                        # Cache thông tin node u để tính nhanh
                        neu = neighbors # set
                        du = len(neu)
                        comm_u_louvain = louvain_map.get(target_u, -1)
                        comm_u_lpa = lpa_map.get(target_u, -1)
                        
                        candidate_details = [] # Lưu thông tin giải thích
                        
                        for v in candidates:
                            nev = set(G_train.neighbors(v))
                            common = neu.intersection(nev) # Justification Path chính là tập này
                            union_set = neu.union(nev)
                            
                            cn = len(common)
                            jaccard = cn / len(union_set) if len(union_set) > 0 else 0
                            
                            aa = 0
                            ra = 0
                            for w in common:
                                deg_w = G_train.degree(w)
                                if deg_w > 1: aa += 1 / math.log(deg_w)
                                if deg_w > 0: ra += 1 / deg_w
                            
                            dv = len(nev)
                            pa = du * dv
                            
                            sl = 1 if comm_u_louvain == louvain_map.get(v, -2) else 0
                            slpa = 1 if comm_u_lpa == lpa_map.get(v, -2) else 0
                            
                            feats = [cn, aa, ra, jaccard, pa, du, dv, sl, slpa]
                            X_pred.append(feats)
                            valid_candidates.append(v)
                            
                            # Lưu detail để hiển thị
                            candidate_details.append({
                                "id": v,
                                "common_neighbors": list(common),
                                "same_community": (sl == 1 or slpa == 1)
                            })
                        
                        # 3. Predict & Rank
                        if len(X_pred) > 0:
                            scores = model.predict_proba(np.array(X_pred))[:, 1] # Lấy xác suất lớp 1
                            
                            # Ghép kết quả
                            results = []
                            for i, v in enumerate(valid_candidates):
                                results.append({
                                    "Candidate ID": v,
                                    "Score (Probability)": float(scores[i]),
                                    "Common Neighbors Count": len(candidate_details[i]["common_neighbors"]),
                                    "Common Neighbors List": candidate_details[i]["common_neighbors"],
                                    "Same Community": candidate_details[i]["same_community"]
                                })
                            
                            # Sort desc
                            df_res = pd.DataFrame(results).sort_values(by="Score (Probability)", ascending=False).head(top_k)
                            
                            # --- HIỂN THỊ KẾT QUẢ ---
                            st.subheader(f"Top {top_k} Gợi ý cho Author {target_u}")
                            
                            # Bảng tổng quan
                            st.dataframe(
                                df_res[["Candidate ID", "Score (Probability)", "Common Neighbors Count", "Same Community"]],
                                use_container_width=True
                            )
                            
                            # Justification Paths (Chi tiết)
                            st.markdown("### 🛤️ Justification & Paths")
                            for idx, row in df_res.iterrows():
                                c_id = row["Candidate ID"]
                                score = row["Score (Probability)"]
                                cn_list = row["Common Neighbors List"]
                                
                                with st.expander(f"🏅 Rank {idx+1}: Author **{c_id}** (Score: {score:.4f})"):
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.write("**Why?**")
                                        st.write(f"- Có **{len(cn_list)}** bạn chung.")
                                        st.write(f"- Cùng cộng đồng: **{'Yes' if row['Same Community'] else 'No'}**")
                                    with col_b:
                                        st.write("**Justification Path (via):**")
                                        st.write(f"{target_u} ↔ {cn_list[:10]}... ↔ {c_id}")
                                        if len(cn_list) > 10:
                                            st.caption(f"*Hiển thị 10/{len(cn_list)} nodes trung gian*")
                        else:
                            st.warning("Không tính được feature cho candidates.")

    except Exception as e:
        st.error(f"Có lỗi xảy ra khi xử lý file: {e}")
        st.write("Vui lòng đảm bảo file đúng định dạng edge list.")

else:
    st.info("👈 Vui lòng upload file dataset từ thanh bên trái.")