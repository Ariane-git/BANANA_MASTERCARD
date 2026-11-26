"""BananaLab: tableau de bord de classification d'images et de prévision MA."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.image_pipeline import (
    CLASS_NAMES,
    dataset_root,
    load_image_model,
    predict_pil_image,
    project_root,
)
from src.market_forecast import (
    fetch_history,
    forecast_frame,
    load_model as load_market_model,
    make_trading_dates,
)


ROOT = project_root()
DATA_DIR = dataset_root()
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
NAVY = "#17212B"
BLUE = "#3977A8"
BANANA = "#F4C542"
LEAF = "#388E67"
MASTERCARD_RED = "#EB001B"
MASTERCARD_ORANGE = "#F79E1B"
CYAN = "#35A9A1"
PALETTE = [BANANA, LEAF, MASTERCARD_RED, MASTERCARD_ORANGE]

st.set_page_config(
    page_title="BananaLab | Analyse & prévisions",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(show_spinner=False)
def read_json(path_string: str):
    path = Path(path_string)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_csv(path_string: str, date_columns: tuple[str, ...] = ()):
    path = Path(path_string)
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=list(date_columns) or None)


@st.cache_data(show_spinner=False)
def inspect_dataset():
    if not DATA_DIR.is_dir():
        summary = read_json(str(ARTIFACTS_DIR / "eda_summary.json")) or {}
        saved_counts = summary.get("counts", {})
        rows = [
            {"Ensemble": split, "Classe": class_name, "Images": int(saved_counts.get(class_name, {}).get(split, 0))}
            for split in ("train", "valid", "test")
            for class_name in CLASS_NAMES
        ]
        return (
            pd.DataFrame(rows),
            summary.get("dimensions", {}),
            summary.get("modes", {}),
            summary.get("formats", {}),
            int(summary.get("unreadable_count", 0)),
        )
    rows, dimensions, modes, formats, unreadable = [], {}, {}, {}, 0
    for split in ("train", "valid", "test"):
        split_dir = DATA_DIR / split
        if not split_dir.is_dir():
            continue
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            files = [file for file in class_dir.iterdir() if file.is_file()] if class_dir.is_dir() else []
            rows.append({"Ensemble": split, "Classe": class_name, "Images": len(files)})
            for path in files:
                formats[path.suffix.lower()] = formats.get(path.suffix.lower(), 0) + 1
                try:
                    with Image.open(path) as image:
                        size, mode = f"{image.width} × {image.height}", image.mode
                        image.verify()
                    dimensions[size] = dimensions.get(size, 0) + 1
                    modes[mode] = modes.get(mode, 0) + 1
                except Exception:
                    unreadable += 1
    return pd.DataFrame(rows), dimensions, modes, formats, unreadable


@st.cache_resource(show_spinner="Chargement du modèle image…")
def get_image_model():
    return load_image_model(ROOT, "cpu")


@st.cache_resource(show_spinner="Chargement du modèle NeuralProphet…")
def get_market_model():
    return load_market_model(MODELS_DIR)


def load_image_results():
    return read_json(str(ARTIFACTS_DIR / "image_results.json"))


def load_market_metrics():
    return read_json(str(ARTIFACTS_DIR / "mastercard_metrics.json"))


def chart_layout(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="sans-serif", color="#52657a", size=12),
        title_font=dict(family="sans-serif", color=NAVY, size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e7edf3")
    fig.update_yaxes(gridcolor="#edf1f5", zerolinecolor="#dbe3eb")
    return fig


def section_header(kicker: str, title: str, description: str | None = None):
    st.caption(kicker)
    st.title(title)
    if description:
        st.caption(description)


def show_dashboard():
    st.title("Banana & Mastercard")
    st.caption("Vision par ordinateur · Prévision boursière NeuralProphet · Tableau de bord de résultats")
    image_results = load_image_results()
    stock_metrics = load_market_metrics()
    image_meta = read_json(str(MODELS_DIR / "image_metadata.json"))
    stock_model_exists = (MODELS_DIR / "mastercard_neuralprophet.np").is_file()

    st.subheader("État des modèles et résultats clés")
    c1, c2, c3, c4 = st.columns(4)
    if image_results:
        selected = image_results.get("selected_model", "Modèle image")
        test_accuracy = image_results.get("test_once", {}).get("accuracy")
        c1.metric("Modèle image retenu", selected, border=True)
        c2.metric("Accuracy test image", f"{test_accuracy:.1%}" if test_accuracy is not None else "—", border=True)
    else:
        c1.metric("Modèle image", "À vérifier", border=True)
        c2.metric("Accuracy test image", "—", border=True)
    if stock_metrics:
        c3.metric("MAE · Mastercard NeuralProphet", f"{stock_metrics['mae']:.2f} $", delta=f"{stock_metrics['mae'] - stock_metrics['baseline_mae']:+.2f} $ vs naïf", delta_color="inverse", border=True)
        c4.metric("RMSE · Mastercard NeuralProphet", f"{stock_metrics['rmse']:.2f} $", border=True)
    else:
        c3.metric("MAE · Mastercard NeuralProphet", "—", border=True)
        c4.metric("RMSE · Mastercard NeuralProphet", "—", border=True)

    if stock_metrics and stock_metrics["mae"] > stock_metrics.get("baseline_mae", float("inf")):
        st.warning(
            f"Sur la validation, la baseline naïve fait mieux : MAE {stock_metrics['baseline_mae']:.2f} $ "
            f"contre {stock_metrics['mae']:.2f} $ pour NeuralProphet. Le tableau le signale pour éviter de présenter le modèle comme plus performant qu'il ne l'est."
        )

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Effet du transfert d’apprentissage")
        if image_results and image_results.get("comparison"):
            comparison = pd.DataFrame(image_results["comparison"])
            by_model = comparison.set_index("modele")
            metric_names = {
                "accuracy": "Accuracy",
                "precision_macro": "Précision macro",
                "rappel_macro": "Rappel macro",
                "f1_macro": "F1 macro",
            }
            available_metrics = [key for key in metric_names if key in by_model.columns]
            model_names = list(by_model.index)
            if len(model_names) >= 2 and available_metrics:
                first_model, second_model = model_names[0], model_names[1]
                fig = go.Figure()
                for metric in available_metrics:
                    fig.add_trace(go.Scatter(
                        x=[by_model.loc[first_model, metric], by_model.loc[second_model, metric]],
                        y=[metric_names[metric], metric_names[metric]],
                        mode="lines",
                        line=dict(color="#CAD3DC", width=3),
                        showlegend=False,
                        hoverinfo="skip",
                    ))
                for model_name, color, position in ((first_model, BANANA, "bottom right"), (second_model, LEAF, "top left")):
                    values = [float(by_model.loc[model_name, metric]) for metric in available_metrics]
                    fig.add_trace(go.Scatter(
                        x=values,
                        y=[metric_names[metric] for metric in available_metrics],
                        mode="markers+text",
                        name=model_name,
                        marker=dict(color=color, size=14, line=dict(color="white", width=2)),
                        text=[f"{value:.1%}" for value in values],
                        textposition=position,
                        textfont=dict(color=NAVY, size=11),
                    ))
                fig.update_layout(
                    title="Scores validation · CNN vs ResNet18",
                    xaxis_title="Score (axe agrandi : 94–100 %)",
                    xaxis=dict(range=[0.94, 1.0], tickformat=".0%", dtick=0.01),
                    yaxis_title="",
                    height=360,
                )
                st.plotly_chart(chart_layout(fig, 360), width="stretch", alt="Scores validation CNN et ResNet18 sur une échelle agrandie de 94 à 100 pour cent")

                if first_model.lower() == "cnn":
                    cnn_name, transfer_name = first_model, second_model
                elif second_model.lower() == "cnn":
                    cnn_name, transfer_name = second_model, first_model
                else:
                    cnn_name, transfer_name = first_model, second_model
                rows = []
                for metric in available_metrics:
                    cnn_score = float(by_model.loc[cnn_name, metric])
                    transfer_score = float(by_model.loc[transfer_name, metric])
                    rows.append({
                        "Métrique": metric_names[metric],
                        "CNN": f"{cnn_score:.2%}",
                        "ResNet18": f"{transfer_score:.2%}",
                        "Gain ResNet18": f"{(transfer_score - cnn_score) * 100:+.2f} points",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", alt="Scores exacts et gain de ResNet18 sur le CNN en points de pourcentage")
                st.caption("Les deux modèles dépassent 95 %. Les points rapprochés sont attendus ; la table donne les valeurs exactes et leur écart.")
        else:
            st.info("Les résultats du notebook image apparaîtront après dépôt de image_results.json dans artifacts/.")
    with right:
        st.subheader("Mastercard · NeuralProphet")
        validation = read_csv(str(ARTIFACTS_DIR / "mastercard_validation.csv"), ("ds",))
        if not validation.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=validation.ds, y=validation.actual, name="Cours réel", line=dict(color=NAVY, width=2.5)))
            fig.add_trace(go.Scatter(x=validation.ds, y=validation.prediction, name="NeuralProphet", line=dict(color=MASTERCARD_RED, width=2.5)))
            fig.update_layout(title="Prédiction vs réalité · 20 séances", yaxis_title="USD")
            st.plotly_chart(chart_layout(fig), width="stretch", alt="Cours Mastercard réellement observé et prévision NeuralProphet pendant la validation")
        else:
            st.info("Le fichier mastercard_validation.csv est absent du dossier artifacts/.")

    st.subheader("Deux modules, une même démarche")
    a, b, c = st.columns(3)
    a.markdown("#### 01 · Explorer\nEffectifs, équilibre, formats, dimensions et exemples du dataset avant les CNN.")
    b.markdown("#### 02 · Comprendre\nComparer CNN et ResNet18, puis examiner les métriques et erreurs par classe.")
    c.markdown("#### 03 · Interagir\nTester une image ou produire une projection Mastercard sur les prochaines séances.")
    if image_meta:
        st.caption(f"Classes image chargées depuis les artefacts : {', '.join(image_meta.get('classes', []))} · Modèle Mastercard : {'disponible' if stock_model_exists else 'absent'}.")


def show_exploration():
    section_header(
        "Module bananes · avant les modèles",
        "Analyse exploratoire du dataset",
        "Cette page répond à quatre questions : combien d’images avons-nous, les classes sont-elles équilibrées, les splits sont-ils cohérents, et les fichiers sont-ils exploitables ?",
    )
    counts, dimensions, modes, formats, unreadable = inspect_dataset()
    if counts.empty:
        st.error("Les dossiers train, valid et test ou les sous-dossiers des classes sont introuvables.")
        return
    pivot = counts.pivot(index="Classe", columns="Ensemble", values="Images").fillna(0).reindex(CLASS_NAMES)
    for split in ("train", "valid", "test"):
        if split not in pivot.columns:
            pivot[split] = 0
    pivot = pivot[["train", "valid", "test"]]
    total = int(counts.Images.sum())
    split_totals = pivot.sum(axis=0)
    split_percent = split_totals / max(total, 1) * 100
    train_total = int(split_totals["train"])
    train_counts = pivot["train"]
    train_share = train_counts / max(train_total, 1) * 100
    majority_class, minority_class = train_counts.idxmax(), train_counts.idxmin()
    majority_share, minority_share = float(train_share.max()), float(train_share.min())
    max_min = float(train_counts.max() / max(1, train_counts.min()))
    common_size = max(dimensions, key=dimensions.get) if dimensions else "—"
    common_format = max(formats, key=formats.get).replace(".", "").upper() if formats else "—"
    common_mode = max(modes, key=modes.get) if modes else "—"

    st.subheader("1 · Volume et découpage")
    kpis = st.columns(4)
    kpis[0].metric("Images au total", f"{total:,}".replace(",", " "), border=True)
    kpis[1].metric("Train", f"{train_total:,}".replace(",", " "), f"{split_percent['train']:.1f} %", border=True)
    kpis[2].metric("Validation", f"{int(split_totals['valid']):,}".replace(",", " "), f"{split_percent['valid']:.1f} %", border=True)
    kpis[3].metric("Test", f"{int(split_totals['test']):,}".replace(",", " "), f"{split_percent['test']:.1f} %", border=True)
    st.write(
        f"Le dataset contient **{total:,} images**. La majorité ({split_percent['train']:.1f} %) sert à apprendre ; "
        f"la validation ({split_percent['valid']:.1f} %) sert à comparer les modèles ; le test ({split_percent['test']:.1f} %) "
        "est gardé pour l’évaluation finale. Cette séparation évite de choisir le modèle avec les images de test."
    )

    st.subheader("2 · Répartition des quatre classes")
    st.write(
        f"Dans train, **{majority_class}** est la classe la plus fréquente ({int(train_counts.max()):,} images, {majority_share:.1f} %), "
        f"et **{minority_class}** la moins fréquente ({int(train_counts.min()):,}, {minority_share:.1f} %). "
        f"La classe majoritaire apparaît **{max_min:.2f} fois** plus souvent que la minoritaire."
    )
    class_colors = {
        "unripe": "#4E9B63",
        "ripe": "#F4C542",
        "overripe": "#F28E2B",
        "rotten": "#805238",
    }
    left, right = st.columns([1.1, 1])
    with left:
        fig = go.Figure()
        for split, color in (("train", BANANA), ("valid", MASTERCARD_ORANGE), ("test", LEAF)):
            fig.add_trace(go.Bar(x=pivot.index, y=pivot[split], name=split.title(), marker_color=color))
        fig.update_layout(title="Nombre d’images par classe et par ensemble", barmode="group", xaxis_title="Classe", yaxis_title="Images")
        st.plotly_chart(chart_layout(fig, 340), width="stretch", alt="Effectifs comparés des quatre classes dans train, validation et test")
    with right:
        share_frame = pd.DataFrame({"Classe": train_share.index, "Part du train (%)": train_share.values})
        share_fig = go.Figure(go.Bar(
            x=share_frame["Part du train (%)"],
            y=share_frame["Classe"],
            orientation="h",
            marker_color=[class_colors[name] for name in share_frame["Classe"]],
            text=[f"{value:.1f} %" for value in share_frame["Part du train (%)"]],
            textposition="outside",
        ))
        share_fig.update_layout(title="Part de chaque classe dans train", xaxis_title="Part (%)", yaxis_title="", xaxis_range=[0, max(40, majority_share + 5)])
        st.plotly_chart(chart_layout(share_fig, 340), width="stretch", alt="Part en pourcentage de chaque classe dans l’ensemble d’entraînement")

    per_class = pivot.astype(int).copy()
    per_class["Part train"] = train_share.map(lambda value: f"{value:.1f} %")
    st.dataframe(per_class, width="stretch", alt="Effectifs et proportion de chaque classe dans les trois ensembles")
    if max_min > 1.5:
        st.info(
            f"Conséquence pour l’apprentissage : un modèle pourrait favoriser {majority_class}. Le notebook compense cela avec des poids de classes calculés uniquement sur train.",
            title="Déséquilibre à prendre en compte",
        )

    st.subheader("3 · Format, dimensions et couleur")
    quality = st.columns(4)
    quality[0].metric("Format dominant", common_format, border=True)
    quality[1].metric("Taille dominante", common_size, border=True)
    quality[2].metric("Mode couleur", common_mode, border=True)
    quality[3].metric("Fichiers illisibles", unreadable, border=True)
    st.write(
        f"Les fichiers sont majoritairement **{common_format}**, en **{common_mode}**, au format **{common_size} pixels**. "
        "Le canal couleur est conservé : le jaune, le vert et les taches brunes aident à distinguer les stades de maturité."
    )
    if unreadable:
        st.warning(f"{unreadable} image(s) n’ont pas pu être ouvertes. Consulte le notebook pour le détail des chemins.", title="Fichiers à vérifier")
    else:
        st.success("Toutes les images ont passé le contrôle de lecture. Aucun fichier corrompu n’a été détecté.", title="Contrôle qualité réussi")

    st.subheader("4 · Inspection visuelle des classes")
    if DATA_DIR.is_dir():
        st.write("Plusieurs images par classe permettent de voir si le modèle recevra des exemples assez variés et si les étiquettes semblent cohérentes.")
    else:
        st.write("Exemples représentatifs conservés avec l’application. Le dataset complet reste dans le projet de travail et n’est pas requis pour utiliser le modèle publié.")
    sample_columns = st.columns(4)
    for col, class_name in zip(sample_columns, CLASS_NAMES):
        example_dir = DATA_DIR / "train" / class_name if DATA_DIR.is_dir() else ROOT / "assets" / "samples"
        examples = sorted(path for path in example_dir.glob("*") if path.is_file() and (DATA_DIR.is_dir() or path.stem.startswith(class_name + "_")))
        col.markdown(f"**{class_name.title()}** · {int(train_counts[class_name]):,} images".replace(",", " "))
        positions = sorted({0, len(examples) // 2, max(0, len(examples) - 1)})
        for position in positions:
            if examples:
                with Image.open(examples[position]) as image:
                    col.image(image.copy(), width="stretch", alt=f"Exemple visuel de la classe {class_name}")
    with st.expander("Comment ces constats guident les modèles"):
        st.markdown(
            "- **Couleur préservée :** pas de conversion en niveaux de gris.\n"
            "- **Augmentation limitée à train :** validation et test restent intacts.\n"
            "- **Poids de classes :** calculés sur train pour limiter le biais vers la classe dominante.\n"
            "- **Choix du modèle :** CNN et ResNet18 sont comparés sur validation ; le test n’intervient qu’à la fin."
        )


def show_image_module():
    section_header("Vision par ordinateur · étape 02", "Reconnaissance de maturité", "Charge une photo de banane. Le modèle retenu dans Colab et ses classes sont lus depuis le dossier models/.")
    model_path = MODELS_DIR / "image_model.pt"
    metadata_path = MODELS_DIR / "image_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        st.error("Modèle image manquant. Place image_model.pt et image_metadata.json dans models/.")
        return
    upload = st.file_uploader("Importer une image", type=["jpg", "jpeg", "png", "webp"], help="Formats pris en charge : JPG, PNG, WEBP.")
    if upload:
        try:
            image = Image.open(upload).convert("RGB")
            model, metadata, image_transform = get_image_model()
            label, probabilities = predict_pil_image(image, model, metadata, image_transform)
            photo, prediction = st.columns([1, 1.2])
            with photo:
                st.image(image, caption="Aperçu", width="stretch", alt="Image de banane importée pour la classification")
            with prediction:
                st.subheader("Prédiction du modèle")
                st.metric("Classe estimée", label)
                st.metric("Confiance du modèle", f"{probabilities.max():.1%}")
                prob_fig = go.Figure(go.Bar(
                    x=probabilities * 100,
                    y=metadata["classes"],
                    orientation="h",
                    marker_color=[BANANA if name == label else "#b9c6d2" for name in metadata["classes"]],
                    text=[f"{value:.1f}%" for value in probabilities * 100],
                    textposition="outside",
                ))
                prob_fig.update_layout(title="Probabilités par classe", xaxis_title="Probabilité (%)", yaxis_title="", xaxis_range=[0, 110])
                st.plotly_chart(chart_layout(prob_fig, 280), width="stretch", alt="Probabilités estimées pour chacune des classes de maturité")
        except Exception as error:
            st.error(f"Impossible d’analyser cette image : {error}")

    results = load_image_results()
    if not results:
        st.info("Les métriques apparaissent après avoir copié image_results.json dans artifacts/.")
        return
    st.subheader("Évaluation enregistrée depuis Colab")
    summary = results.get("test_once", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Modèle retenu sur validation", results.get("selected_model", "—"))
    m2.metric("Accuracy test", f"{summary.get('accuracy', 0):.1%}")
    m3.metric("Macro F1 validation", f"{results.get('validation', {}).get('resnet18', {}).get('report', {}).get('macro avg', {}).get('f1-score', 0):.1%}" if results.get("selected_architecture") == "resnet18" else "Voir comparaison")
    comparison = pd.DataFrame(results.get("comparison", []))
    if not comparison.empty:
        st.dataframe(comparison.round(4), hide_index=True, width="stretch", alt="Comparaison des métriques de validation entre CNN et ResNet18")
    tabs = st.tabs(["Courbes d’apprentissage", "Matrice de confusion test", "Détails par classe"])
    with tabs[0]:
        histories = results.get("histories", {})
        fig = go.Figure()
        for name, history in histories.items():
            if history:
                frame = pd.DataFrame(history)
                fig.add_trace(go.Scatter(x=frame.epoch, y=frame.valid_loss, mode="lines+markers", name=f"{name} · validation", line=dict(color=PALETTE[len(fig.data) % len(PALETTE)])))
        fig.update_layout(title="Perte de validation par époque", xaxis_title="Époque", yaxis_title="Cross-entropy")
        st.plotly_chart(chart_layout(fig), width="stretch", alt="Perte de validation par époque pour les modèles CNN et ResNet18")
    with tabs[1]:
        matrix = summary.get("confusion_matrix")
        if matrix:
            fig = go.Figure(go.Heatmap(z=matrix, x=CLASS_NAMES, y=CLASS_NAMES, colorscale="Blues", text=matrix, texttemplate="%{text}"))
            fig.update_layout(title="Test final · réel vs prédit", xaxis_title="Classe prédite", yaxis_title="Classe réelle")
            st.plotly_chart(chart_layout(fig), width="stretch", alt="Matrice de confusion du modèle retenu sur le jeu test")
    with tabs[2]:
        report = summary.get("report", {})
        if report:
            st.dataframe(pd.DataFrame(report).T.reindex(CLASS_NAMES)[["precision", "recall", "f1-score", "support"]].round(3), width="stretch", alt="Précision, rappel et F1 par classe sur le test")
    st.caption("La confiance est la probabilité produite par le classifieur ; elle n’est pas une certitude calibrée ni un diagnostic alimentaire.")


def show_market_module():
    section_header("Série temporelle · étape 03", "Mastercard · NeuralProphet", "La validation respecte l’ordre des dates. L’horizon est exprimé en séances NYSE, hors week-ends et jours fériés.")
    metrics = load_market_metrics()
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE · NeuralProphet", f"{metrics['mae']:.2f} USD")
        c2.metric("RMSE · NeuralProphet", f"{metrics['rmse']:.2f} USD")
        c3.metric("MAE · baseline naïve", f"{metrics['baseline_mae']:.2f} USD")
        c4.metric("RMSE · baseline naïve", f"{metrics['baseline_rmse']:.2f} USD")
        if metrics["mae"] > metrics["baseline_mae"]:
            st.warning("La baseline naïve est meilleure sur cette période de validation. La prévision est montrée pour l’analyse du modèle, pas comme une stratégie d’investissement validée.")

    validation = read_csv(str(ARTIFACTS_DIR / "mastercard_validation.csv"), ("ds",))
    if not validation.empty:
        validation_tab, forecast_tab = st.tabs(["Validation du modèle", "Prévision"])
        with validation_tab:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=validation.ds, y=validation.actual, name="Cours réel", line=dict(color=NAVY, width=2.5)))
            fig.add_trace(go.Scatter(x=validation.ds, y=validation.prediction, name="NeuralProphet", line=dict(color=MASTERCARD_RED, width=2.5)))
            if {"borne_basse", "borne_haute"}.issubset(validation.columns):
                fig.add_trace(go.Scatter(x=validation.ds, y=validation.borne_haute, line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=validation.ds, y=validation.borne_basse, fill="tonexty", fillcolor="rgba(247,158,27,.16)", line=dict(width=0), name="Quantiles 10–90 %"))
            fig.update_layout(title="Période de validation · 20 séances", xaxis_title="Date", yaxis_title="USD")
            st.plotly_chart(chart_layout(fig, 390), width="stretch", alt="Cours Mastercard réel et valeurs prévues par NeuralProphet sur validation")
            st.dataframe(validation.rename(columns={"ds": "Date", "actual": "Réel (USD)", "prediction": "Prédit (USD)"}).round(2), hide_index=True, width="stretch", alt="Tableau des valeurs Mastercard réelles et prévues sur validation")
        with forecast_tab:
            horizon = st.slider("Horizon", min_value=5, max_value=40, value=20, help="Nombre de séances de marché, pas de jours calendaires.")
            if st.button("Afficher la prévision", type="primary"):
                try:
                    if horizon == 20 and (ARTIFACTS_DIR / "mastercard_forecast_20_sessions.csv").is_file():
                        forecast = read_csv(str(ARTIFACTS_DIR / "mastercard_forecast_20_sessions.csv"), ("ds",))
                    else:
                        model = get_market_model()
                        history = read_csv(str(ARTIFACTS_DIR / "mastercard_history.csv"), ("ds",))
                        if history.empty:
                            history = fetch_history("8y")
                        dates = make_trading_dates(history.ds.iloc[-1], horizon)
                        forecast = forecast_frame(model, dates)
                    history_chart = read_csv(str(ARTIFACTS_DIR / "mastercard_history.csv"), ("ds",))
                    fig = go.Figure()
                    if not history_chart.empty:
                        recent = history_chart.tail(180)
                        fig.add_trace(go.Scatter(x=recent.ds, y=recent.y, name="Historique", line=dict(color=NAVY, width=2)))
                    fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.prediction, name="Mastercard · NeuralProphet", line=dict(color=MASTERCARD_RED, width=2.5, dash="dash")))
                    if {"borne_basse", "borne_haute"}.issubset(forecast.columns):
                        fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.borne_haute, line=dict(width=0), showlegend=False, hoverinfo="skip"))
                        fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.borne_basse, fill="tonexty", fillcolor="rgba(247,158,27,.16)", line=dict(width=0), name="Quantiles 10–90 %"))
                    fig.update_layout(title=f"Projection · {len(forecast)} séances", xaxis_title="Séance NYSE", yaxis_title="USD")
                    st.plotly_chart(chart_layout(fig, 390), width="stretch", alt=f"Historique du cours Mastercard et prévision NeuralProphet sur {horizon} séances")
                    st.dataframe(forecast.rename(columns={"ds": "Date", "prediction": "Cours estimé (USD)", "borne_basse": "Quantile 10 %", "borne_haute": "Quantile 90 %"}).round(2), hide_index=True, width="stretch", alt="Table des cours estimés et quantiles de prévision")
                    st.download_button("Télécharger les valeurs CSV", forecast.to_csv(index=False).encode("utf-8"), "previsions_mastercard.csv", "text/csv")
                except Exception as error:
                    st.error(f"La prévision n’a pas pu être générée : {error}")
    else:
        st.warning("Fichier de validation absent. Ajoute les artefacts exportés depuis Colab dans le dossier artifacts/.")
    st.caption("Les prévisions de cours sont incertaines et ne constituent pas un conseil financier.")


def main():
    with st.sidebar:
        st.title("🍌 Banana · Mastercard")
        st.caption("Classification d’images · Bourse NeuralProphet")
        st.divider()
        page = st.radio(
            "Navigation",
            ["Vue générale", "Exploration des données", "Analyse des bananes", "Mastercard & NeuralProphet"],
            index=0,
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("PROJET MACHINE LEARNING")
        st.caption("PyTorch · NeuralProphet · Streamlit")

    if page == "Vue générale":
        show_dashboard()
    elif page == "Exploration des données":
        show_exploration()
    elif page == "Analyse des bananes":
        show_image_module()
    else:
        show_market_module()


if __name__ == "__main__":
    main()
