import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
from automl_core import run_automl, save_model

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    px = None
    go = None
    make_subplots = None

try:
    import shap
except ImportError:
    shap = None

st.set_page_config(page_title="Auto-ML Dashboard", layout="wide", page_icon="🚀")

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .big-font  { font-size:24px !important; font-weight: bold; color: #1f77b4; }
    .metric-card { background-color:#ffffff; border-radius:8px; padding:16px;
                   box-shadow:0 4px 10px rgba(0,0,0,.12); margin-bottom:16px; }
    .warning-text { color: #d62728; font-weight:bold; }
    .section-divider { border-top: 2px solid #e0e0e0; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

st.title("Auto-ML Dashboard")
st.markdown("Advanced Preprocessing, Automated Cross-Validation, & Explainable AI.")

# ── Helper: build evaluation charts from y_test / y_pred ─────────────────────
def render_evaluation_charts(problem_type, y_test, y_pred, y_prob, class_names, prefix=""):
    """Renders metrics cards + comparison charts.
    prefix is used to make widget keys unique across tabs."""

    y_test  = np.array(y_test)
    y_pred  = np.array(y_pred)

    if problem_type == 'classification':
        from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                     confusion_matrix, classification_report)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted')
        auc = np.nan
        if y_prob is not None:
            try:
                y_prob = np.array(y_prob)
                if len(np.unique(y_test)) > 2:
                    auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                else:
                    auc = roc_auc_score(y_test, y_prob[:, 1])
            except Exception:
                pass

        # ── metric cards ────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric(" Accuracy",  f"{acc*100:.2f}%")
        m2.metric(" F1-Score (weighted)", f"{f1:.4f}")
        m3.metric(" ROC-AUC",   f"{auc:.4f}" if not np.isnan(auc) else "N/A")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        # ── Confusion Matrix ─────────────────────────────────────────────────
        with chart_col1:
            st.markdown("**Confusion Matrix**")
            if px and go:
                cm        = confusion_matrix(y_test, y_pred)
                labels    = class_names if class_names else [str(i) for i in sorted(np.unique(y_test))]
                cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised %

                text_vals = [[f"{cm[r][c]}<br>({cm_norm[r][c]*100:.1f}%)"
                              for c in range(len(labels))] for r in range(len(labels))]

                heatmap = go.Figure(go.Heatmap(
                    z=cm_norm, x=labels, y=labels,
                    text=text_vals, texttemplate="%{text}",
                    colorscale="Blues", showscale=True,
                    colorbar=dict(title="Ratio")
                ))
                heatmap.update_layout(
                    title="Predicted vs Actual",
                    xaxis_title="Predicted", yaxis_title="Actual",
                    height=380
                )
                st.plotly_chart(heatmap, use_container_width=True)

        # ── Prediction Distribution (class counts) ───────────────────────────
        with chart_col2:
            st.markdown("**Prediction Distribution**")
            if px:
                labels = class_names if class_names else [str(i) for i in np.unique(y_pred)]
                pred_series  = pd.Series(y_pred).map(
                    {i: (labels[i] if isinstance(i, int) and i < len(labels) else str(i))
                     for i in np.unique(y_pred)}
                )
                dist_df = pred_series.value_counts().reset_index()
                dist_df.columns = ['Class', 'Count']
                fig_dist = px.pie(dist_df, names='Class', values='Count',
                                  title="Predicted Class Distribution",
                                  color_discrete_sequence=px.colors.qualitative.Pastel,
                                  hole=0.35)
                fig_dist.update_layout(height=380)
                st.plotly_chart(fig_dist, use_container_width=True)

        # ── Per-class report ────────────────────────────────────────────────
        with st.expander(" Full Classification Report"):
            report_dict = classification_report(
                y_test, y_pred,
                target_names=class_names,
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report_dict).T.round(4))

        # ── ROC Curve (binary only) ──────────────────────────────────────────
        if y_prob is not None and len(np.unique(y_test)) == 2 and px:
            with st.expander(" ROC Curve"):
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, np.array(y_prob)[:, 1])
                roc_fig = px.line(x=fpr, y=tpr,
                                  labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                                  title=f"ROC Curve (AUC = {auc:.3f})")
                roc_fig.add_shape(type='line', x0=0, x1=1, y0=0, y1=1,
                                  line=dict(dash='dash', color='gray'))
                roc_fig.update_layout(height=380)
                st.plotly_chart(roc_fig, use_container_width=True)

    else:  # regression
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📉 RMSE",  f"{rmse:.4f}")
        m2.metric("📉 MAE",   f"{mae:.4f}")
        m3.metric("📉 MSE",   f"{mse:.4f}")
        m4.metric("📈 R² Score", f"{r2:.4f}")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        if px:
            chart_col1, chart_col2 = st.columns(2)

            # ── Actual vs Predicted scatter ──────────────────────────────────
            with chart_col1:
                st.markdown("**Actual vs Predicted**")
                scatter_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                fig_sc = px.scatter(scatter_df, x='Actual', y='Predicted',
                                    title="Actual vs Predicted Values",
                                    opacity=0.7,
                                    color_discrete_sequence=['#1f77b4'])
                mn = min(y_test.min(), y_pred.min())
                mx = max(y_test.max(), y_pred.max())
                fig_sc.add_shape(type='line', x0=mn, x1=mx, y0=mn, y1=mx,
                                 line=dict(color='red', dash='dash'))
                fig_sc.update_layout(height=380)
                st.plotly_chart(fig_sc, use_container_width=True)

            # ── Residuals distribution ───────────────────────────────────────
            with chart_col2:
                st.markdown("**Residuals Distribution**")
                residuals = y_test - y_pred
                fig_res = px.histogram(residuals, nbins=30,
                                       title="Residuals Distribution",
                                       labels={'value': 'Residual', 'count': 'Count'},
                                       color_discrete_sequence=['#ff7f0e'])
                fig_res.update_layout(height=380)
                st.plotly_chart(fig_res, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([" Training Pipeline", " What-If Predictor", " Batch Prediction"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("1. Upload Training Data")
    uploaded_file = st.file_uploader("Upload your training dataset", type=["csv", "xlsx"], key="train")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head())

            st.header("2. Configuration")
            target_column = st.selectbox("Select Target Column", df.columns)

            st.markdown("---")
            if st.button(" Start Auto-ML Pipeline", type="primary"):
                for key in ['training_results', 'training_df', 'training_target']:
                    st.session_state.pop(key, None)

                progress_text = st.empty()
                progress_bar  = st.progress(0)

                def update_progress(msg):
                    progress_text.text(msg)

                with st.spinner("Executing advanced training pipeline..."):
                    results_dict = run_automl(df, target_column, progress_callback=update_progress)
                    progress_text.text("Pipeline Execution Completed! ")
                    progress_bar.progress(100)

                    st.session_state['training_results'] = results_dict
                    st.session_state['training_df']      = df
                    st.session_state['training_target']  = target_column
                    st.session_state['latest_model']     = results_dict

        except Exception as e:
            st.error(f"Error processing file: {e}")

    # ── Render persisted results ──────────────────────────────────────────────
    if 'training_results' in st.session_state:
        rd              = st.session_state['training_results']
        problem_type    = rd['problem_type']
        results         = rd['results']
        best_model_name = rd['best_model_name']
        qr              = rd['quality_report']

        st.success(f" Pipeline finished! Task Type: **{problem_type.capitalize()}**")

        # ── Data Quality Report ───────────────────────────────────────────────
        st.subheader(" Data Quality Report")
        q1, q2, q3 = st.columns(3)
        with q1:
            st.markdown("**Leakage Check**")
            if qr['leakage_warnings']:
                st.markdown(f"<span class='warning-text'>⚠️ {qr['leakage_warnings']}</span>", unsafe_allow_html=True)
            else:
                st.write(" Passed")
        with q2:
            st.markdown("**Class Imbalance**")
            if qr.get('is_imbalanced'):
                st.markdown("<span class='warning-text'>⚠️ Severe imbalance – class weights applied.</span>", unsafe_allow_html=True)
            else:
                st.write(" Passed / N/A")
        with q3:
            st.markdown("**Outliers (IQR)**")
            if qr['outliers']:
                for col, c in qr['outliers'].items():
                    st.write(f"- {col}: {c}")
            else:
                st.write(" Passed")

        with st.expander("Missing Value Treatment"):
            if qr['missing_values']:
                st.write(qr['missing_values'])
                st.info("Numerical → Median imputation | Categorical → Mode imputation")
            else:
                st.write("No missing values found.")

        st.markdown("---")

        # ── Model Zoo leaderboard ─────────────────────────────────────────────
        st.subheader(" Model Zoo Leaderboard")
        zoo = rd.get('zoo_results', {})
        if zoo:
            zoo_df = pd.DataFrame.from_dict(zoo, orient='index', columns=['CV_Score'])
            if px:
                fig_zoo = px.bar(
                    zoo_df.sort_values('CV_Score', ascending=True).reset_index(),
                    x='CV_Score', y='index', orientation='h',
                    color='CV_Score', color_continuous_scale='Blues',
                    title="Model Comparison (Cross-Val Score)",
                    labels={'index': 'Model', 'CV_Score': 'CV Score'}
                )
                fig_zoo.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_zoo, use_container_width=True)
            else:
                st.dataframe(zoo_df)

        # ── Best Model metrics ────────────────────────────────────────────────
        st.subheader(f" Best Model: {best_model_name}")
        st.markdown(f'<div class="metric-card"><p class="big-font">Winner: {best_model_name}</p></div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame.from_dict(results, orient='index').round(4))

        # ── Feature Importances ───────────────────────────────────────────────
        st.subheader(" Feature Importances")
        fi = rd['feature_importances']
        if fi is not None and px:
            fig_fi = px.bar(fi.tail(15), x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Teal',
                            title=f"Top Features – {best_model_name}")
            fig_fi.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        elif fi is not None:
            st.bar_chart(fi.set_index('Feature').tail(15))
        else:
            st.info("Feature importances unavailable for this model type.")

        # ── Evaluation Metrics & Charts (test-set) ───────────────────────────
        st.markdown("---")
        st.subheader(" Model Evaluation on Hold-out Test Set")
        render_evaluation_charts(
            problem_type  = rd['problem_type'],
            y_test        = rd['y_test'],
            y_pred        = rd['y_pred'],
            y_prob        = rd.get('y_prob'),
            class_names   = rd.get('class_names'),
            prefix        = "train_"
        )

        # ── SHAP ──────────────────────────────────────────────────────────────
        if shap:
            with st.expander(" SHAP Explanation (may take time)"):
                if st.button("Calculate SHAP", key="shap_btn"):
                    with st.spinner("Calculating Shapley values..."):
                        st.warning("SHAP with calibrated pipelines requires custom extraction – omitted for stability.")

        # ── Deployment ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader(" Deployment")
        model_bytes, model_filename = save_model(rd)
        st.download_button("⬇ Download Versioned Model (.pkl)", model_bytes,
                           file_name=model_filename,
                           mime="application/octet-stream",
                           key="dl_model_btn")
        st.info("Click the button above to download the model. Nothing is saved to disk automatically.")



# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – WHAT-IF PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header(" What-If Single Predictor")
    if 'latest_model' in st.session_state:
        res    = st.session_state['latest_model']
        schema = res['schema']
        st.markdown("Enter feature values to get an instant prediction from the trained model:")

        with st.form("whatif_form"):
            user_inputs = {}
            cols = st.columns(3)
            for i, feature in enumerate(schema):
                with cols[i % 3]:
                    user_inputs[feature] = st.text_input(feature, value="")
            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([user_inputs])
            try:
                pipe       = res['best_pipeline']
                prediction = pipe.predict(input_df)[0]
                if res['problem_type'] == 'classification':
                    prob       = pipe.predict_proba(input_df)[0]
                    pred_label = res['label_encoder'].inverse_transform([prediction])[0]
                    st.success(f"**Prediction:** {pred_label}  |  **Confidence:** {max(prob)*100:.2f}%")

                    if px:
                        labels = res.get('class_names') or [str(i) for i in range(len(prob))]
                        prob_fig = px.bar(x=labels, y=prob,
                                          labels={'x': 'Class', 'y': 'Probability'},
                                          title="Class Probabilities",
                                          color=prob,
                                          color_continuous_scale='Blues')
                        prob_fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(prob_fig, use_container_width=True)
                else:
                    st.success(f"**Prediction:** {prediction:.4f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.info("🏋️ Train a model in the **Training Pipeline** tab first to unlock What-If analysis.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header(" Batch Prediction")
    st.markdown("Upload your `.pkl` model and a dataset to generate predictions with full analytics.")

    c1, c2 = st.columns(2)
    with c1:
        model_file = st.file_uploader("1. Upload Saved Model (.pkl)", type=["pkl"], key="model")
    with c2:
        predict_file = st.file_uploader("2. Upload Dataset for Prediction", type=["csv", "xlsx"], key="predict")

    if model_file is not None and predict_file is not None:
        try:
            pred_df = (pd.read_csv(predict_file) if predict_file.name.endswith('.csv')
                       else pd.read_excel(predict_file))

            st.subheader("Dataset Preview")
            st.dataframe(pred_df.head())

            if st.button(" Generate Predictions", type="primary"):
                with st.spinner("Loading model and running predictions..."):
                    loaded_data  = joblib.load(model_file)
                    pipeline     = loaded_data['pipeline']
                    schema       = loaded_data['schema']
                    p_type       = loaded_data['problem_type']
                    label_enc    = loaded_data.get('label_encoder')

                    missing_cols = [c for c in schema if c not in pred_df.columns]
                    if missing_cols:
                        st.error(f" Schema Validation Failed! Missing columns: {missing_cols}")
                    else:
                        raw_preds  = pipeline.predict(pred_df[schema])
                        y_prob_bat = None

                        result_df  = pred_df.copy()

                        if p_type == 'classification':
                            if hasattr(pipeline, "predict_proba"):
                                probs         = pipeline.predict_proba(pred_df[schema])
                                y_prob_bat    = probs
                                result_df['Prediction_Confidence'] = [max(p) for p in probs]

                            if label_enc is not None:
                                result_df['Predictions'] = label_enc.inverse_transform(raw_preds)
                            else:
                                result_df['Predictions'] = raw_preds
                        else:
                            result_df['Predictions'] = raw_preds

                        st.session_state['batch_results']      = result_df
                        st.session_state['batch_raw_preds']    = raw_preds.tolist()
                        st.session_state['batch_y_prob']       = y_prob_bat
                        st.session_state['batch_problem_type'] = p_type
                        st.session_state['batch_class_names']  = (label_enc.classes_.tolist()
                                                                   if label_enc is not None else None)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # ── Render batch results persistently ─────────────────────────────────────
    if 'batch_results' in st.session_state:
        batch_df  = st.session_state['batch_results']
        p_type    = st.session_state.get('batch_problem_type', 'classification')
        raw_preds = np.array(st.session_state.get('batch_raw_preds', []))
        y_prob_b  = st.session_state.get('batch_y_prob')
        cls_names = st.session_state.get('batch_class_names')

        st.success(" Batch predictions complete!")
        st.dataframe(batch_df)

        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Predictions as CSV", csv,
                           file_name='predictions_batch.csv',
                           mime='text/csv', key="dl_batch_btn")

        # ── Analytics — metrics only ──────────────────────────────────────────
        st.markdown("---")
        st.subheader(" Prediction Analytics")

        if p_type == 'classification':
            # Map encoded labels back to class names for display
            if cls_names is not None:
                display_preds = [cls_names[p] if isinstance(p, (int, np.integer)) and p < len(cls_names)
                                 else str(p) for p in raw_preds]
            else:
                display_preds = [str(p) for p in raw_preds]

            pred_series = pd.Series(display_preds)
            dist_df     = pred_series.value_counts().reset_index()
            dist_df.columns = ['Class', 'Count']

            # ── Summary table ─────────────────────────────────────────────────
            st.subheader(" Prediction Summary")
            summary = dist_df.copy()
            summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(2)
            if 'Prediction_Confidence' in batch_df.columns:
                conf_stats = batch_df.groupby('Predictions')['Prediction_Confidence'].agg(['mean','min','max']).round(4)
                conf_stats.columns = ['Avg Confidence', 'Min Confidence', 'Max Confidence']
                conf_stats = conf_stats.reset_index().rename(columns={'Predictions': 'Class'})
                summary = summary.merge(conf_stats, on='Class', how='left')
            st.dataframe(summary.set_index('Class'))

        else:  # regression batch
            preds = batch_df['Predictions']

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(" Mean",   f"{preds.mean():.4f}")
            m2.metric(" Median", f"{preds.median():.4f}")
            m3.metric(" Max",    f"{preds.max():.4f}")
            m4.metric(" Min",    f"{preds.min():.4f}")
