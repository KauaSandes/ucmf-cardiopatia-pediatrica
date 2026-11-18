"""Modelagem (Avaliação A2): classificação NORMAL X ANORMAL na base limpa.

Protocolo
- holdout estratificado 80/20 (random_state=42); o teste é usado uma única vez por modelo;
- ajuste de hiperparâmetros com GridSearchCV + StratifiedKFold(5) somente no treino;
- todo o pré-processamento (imputação, log, padronização, codificação) fica dentro do Pipeline,
  portanto é ajustado apenas nos dados de treino de cada dobra;
- SOPRO é excluído (data leakage apontado na A1); aparece só no experimento de sensibilidade.

Gera results/metricas.json e results/figuras/*.png.  Executar:  python src/modelagem.py
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "results" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
ALVO = "NORMAL X ANORMAL"

NUM_LOG = ["Peso", "FC"]            # assimetria positiva -> log1p
NUM = ["IDADE", "Altura"]
NOMINAIS = ["SEXO", "B2", "PULSOS", "HDA 1", "MOTIVO1", "MOTIVO2"]
PPA = ["PPA"]
EXCLUIDOS = ["SOPRO (leakage)", "ID", "Convenio", "Atendimento", "DN", "HDA2", "PA SISTOLICA", "PA DIASTOLICA",
             "IMC (derivado)"]
GRAU_PPA = {"Normal": 0, "Pré-hipertensão": 1, "HAS estágio 1": 2, "HAS estágio 2": 3}

SCORING = {"roc_auc": "roc_auc", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
           "precision": "precision", "recall": "recall", "f1": "f1"}
C_NORMAL, C_ANORMAL = "#2a78d6", "#eb6834"
CORES = {"Dummy": "#8a8984", "Regressão Logística": "#2a78d6", "Random Forest": "#1baf7a",
         "HistGradientBoosting": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def codificar_ppa(X):
    """Ordinal encoding da PPA (0 a 3) + indicadores de PA não aferida / não calculável (sem estado)."""
    s = pd.Series(np.asarray(X).ravel())
    return np.column_stack([s.map(GRAU_PPA).fillna(0).to_numpy(float),
                            (s == "Não aferida").to_numpy(float),
                            (s == "Não calculável").to_numpy(float)])


def preprocessador(nominais):
    return ColumnTransformer([
        ("num_log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                              ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                              ("esc", StandardScaler())]), NUM_LOG),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("esc", StandardScaler())]), NUM),
        ("ppa", FunctionTransformer(codificar_ppa, feature_names_out=lambda *_: np.array(
            ["PPA_grau", "PPA_nao_aferida", "PPA_nao_calculavel"])), PPA),
        ("nom", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
                          ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                                sparse_output=False))]), nominais),
    ])


MODELOS = {
    "Dummy": (DummyClassifier(strategy="most_frequent"), {}),
    "Regressão Logística": (LogisticRegression(max_iter=3000, random_state=SEED),
                            {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}),
    "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
                      {"clf__max_depth": [None, 10, 20], "clf__min_samples_leaf": [1, 5, 10],
                       "clf__class_weight": [None, "balanced"]}),
    "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=SEED),
                             {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [None, 3, 6],
                              "clf__max_iter": [200, 400]}),
}


def metricas_teste(y, proba):
    pred = (proba >= 0.5).astype(int)
    return {"roc_auc": roc_auc_score(y, proba), "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0)}


def r(d, n=4):
    return {k: round(float(v), n) for k, v in d.items()}


def main():
    df = pd.read_csv(RAIZ / "data" / "UCMF_limpo.csv", encoding="utf-8")
    y_all = (df[ALVO] == "Anormal").astype(int)  # label encoding: Normal = 0, Anormal = 1
    R = {"seed": SEED, "atributos": {"numericos": NUM_LOG + NUM, "ordinal": PPA, "nominais": NOMINAIS,
                                     "excluidos": EXCLUIDOS}}

    # ---------------- EDA para o relatório (base limpa inteira, nenhum ajuste de modelo)
    sopro = pd.crosstab(df["SOPRO"].fillna("Ausente"), df[ALVO])
    R["eda"] = {"n": int(len(df)), "alvo": df[ALVO].value_counts().to_dict(),
                "pct_anormal": round(100 * y_all.mean(), 1),
                "sopro": {k: {"n": int(v.sum()), "pct_anormal": round(100 * v["Anormal"] / v.sum(), 1)}
                          for k, v in sopro.iterrows()},
                "ausentes_pct": {c: round(100 * df[c].isna().mean(), 1) for c in NUM_LOG + NUM + ["SEXO"]}}

    X = df[NUM_LOG + NUM + PPA + NOMINAIS + ["SOPRO"]]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
    R["divisao"] = {"treino": int(len(X_tr)), "teste": int(len(X_te)),
                    "pct_anormal_treino": round(100 * y_tr.mean(), 1), "pct_anormal_teste": round(100 * y_te.mean(), 1)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    feats = NUM_LOG + NUM + PPA + NOMINAIS

    # ---------------- ajuste (treino) e avaliação única (teste)
    R["modelos"], ajustados, probas = {}, {}, {}
    for nome, (clf, grade) in MODELOS.items():
        pipe = Pipeline([("pre", preprocessador(NOMINAIS)), ("clf", clf)])
        t0 = time.time()
        gs = GridSearchCV(pipe, grade or {"clf__strategy": ["most_frequent"]}, scoring=SCORING, refit="roc_auc",
                          cv=cv, n_jobs=-1)
        gs.fit(X_tr[feats], y_tr)
        i = gs.best_index_
        cvm = {m: (gs.cv_results_[f"mean_test_{m}"][i], gs.cv_results_[f"std_test_{m}"][i]) for m in SCORING}
        proba = gs.predict_proba(X_te[feats])[:, 1]
        probas[nome], ajustados[nome] = proba, gs.best_estimator_
        R["modelos"][nome] = {"melhores_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                              "cv_media": r({m: v[0] for m, v in cvm.items()}),
                              "cv_dp": r({m: v[1] for m, v in cvm.items()}),
                              "teste": r(metricas_teste(y_te, proba)),
                              "tempo_ajuste_s": round(time.time() - t0, 1)}
        print(nome, R["modelos"][nome]["cv_media"]["roc_auc"], R["modelos"][nome]["teste"])

    melhor = max((n for n in MODELOS if n != "Dummy"), key=lambda n: R["modelos"][n]["cv_media"]["roc_auc"])
    R["melhor_modelo"] = melhor
    pred = (probas[melhor] >= 0.5).astype(int)
    cm = confusion_matrix(y_te, pred)
    R["matriz_confusao"] = {"modelo": melhor, "linhas_real_colunas_previsto": cm.tolist(),
                            "rotulos": ["Normal", "Anormal"]}

    # limiar de decisão: escolhido nas previsões fora da dobra do TREINO para sensibilidade >= 0,80
    oof = cross_val_predict(clone(ajustados[melhor]), X_tr[feats], y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    fpr, tpr, thr = roc_curve(y_tr, oof)
    limiar = float(thr[np.argmax(tpr >= 0.80)])
    pred_l = (probas[melhor] >= limiar).astype(int)
    R["limiar"] = {"alvo_sensibilidade_treino": 0.80, "limiar": round(limiar, 3),
                   "teste": r(metricas_teste(y_te, probas[melhor]) | {
                       "accuracy": accuracy_score(y_te, pred_l), "balanced_accuracy": balanced_accuracy_score(y_te, pred_l),
                       "precision": precision_score(y_te, pred_l), "recall": recall_score(y_te, pred_l),
                       "f1": f1_score(y_te, pred_l)}),
                   "matriz_confusao": confusion_matrix(y_te, pred_l).tolist()}

    # erros por faixa etária (onde o melhor modelo erra, limiar 0,5)
    te = X_te.assign(real=y_te.values, prev=pred)
    te["faixa"] = pd.cut(te["IDADE"], [-0.01, 1, 3, 7, 11, 20], labels=["<1", "1-3", "3-7", "7-11", "11-19"])
    R["erros_por_faixa"] = {str(k): {"n": int(len(g)), "fn": int(((g.real == 1) & (g.prev == 0)).sum()),
                                     "fp": int(((g.real == 0) & (g.prev == 1)).sum()),
                                     "recall": round(float(recall_score(g.real, g.prev, zero_division=0)), 3)}
                            for k, g in te.groupby("faixa", observed=True)}

    # ---------------- importância dos atributos
    pi = permutation_importance(ajustados[melhor], X_te[feats], y_te, scoring="roc_auc", n_repeats=10,
                                random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    imp_dp = pd.Series(pi.importances_std, index=feats)
    R["importancia_permutacao"] = {"modelo": melhor, "metrica": "queda de ROC-AUC no teste",
                                   "media": r(imp.to_dict()), "dp": r(imp_dp[imp.index].to_dict())}
    lr = ajustados["Regressão Logística"]
    nomes_lr = lr.named_steps["pre"].get_feature_names_out()
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=nomes_lr).sort_values()
    R["coef_regressao_logistica"] = r(pd.concat([coef.head(6), coef.tail(6)]).to_dict(), 3)

    # ---------------- experimentos de sensibilidade (só validação cruzada no treino)
    base_params = {k: v for k, v in ajustados[melhor].get_params().items() if k.startswith("clf__")}
    sens = {}
    for rotulo, nominais in [("modelo final (sem SOPRO)", NOMINAIS),
                             ("com SOPRO (leakage)", NOMINAIS + ["SOPRO"]),
                             ("sem SOPRO e sem MOTIVO1/2", [c for c in NOMINAIS if not c.startswith("MOTIVO")])]:
        pipe = Pipeline([("pre", preprocessador(nominais)), ("clf", clone(MODELOS[melhor][0]))]).set_params(**base_params)
        cols = NUM_LOG + NUM + PPA + nominais
        res = cross_validate(pipe, X_tr[cols], y_tr, cv=cv, scoring=SCORING, n_jobs=-1)
        sens[rotulo] = {"cv_media": r({m: res[f"test_{m}"].mean() for m in SCORING}),
                        "cv_dp": r({m: res[f"test_{m}"].std() for m in SCORING})}
    R["sensibilidade"] = {"modelo": melhor, "resultados": sens}

    (RAIZ / "results" / "metricas.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- figuras
    # Fig. 1 (EDA): alvo e SOPRO x alvo
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.45), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.75})
    ax = axs[0]
    vc = df[ALVO].value_counts()[["Normal", "Anormal"]]
    ax.bar(vc.index, vc.values, color=[C_NORMAL, C_ANORMAL], width=0.6)
    for i, v in enumerate(vc.values):
        ax.text(i, v + 250, f"{v:,}\n({100 * v / vc.sum():.0f}%)".replace(",", "."), ha="center", va="bottom",
                fontsize=6.5)
    ax.set_ylim(0, vc.max() * 1.45)
    ax.set_title("(a) Classes", loc="left")
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=6.5)
    ax = axs[1]
    s = {k: v for k, v in R["eda"]["sopro"].items() if v["n"] >= 30}
    ks = sorted(s, key=lambda k: s[k]["pct_anormal"])
    vals = [s[k]["pct_anormal"] for k in ks]
    ax.barh(ks, vals, color=[C_ANORMAL if v > R["eda"]["pct_anormal"] else "#b8b7b1" for v in vals], height=0.6)
    ax.axvline(R["eda"]["pct_anormal"], color=INK2, lw=0.8, ls="--")
    for j, v in enumerate(vals):
        ax.text(v + 3, j, f"{v:.0f}%", va="center", fontsize=6.5)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 50, 100], ["0", "50", "100%"])
    ax.set_title("(b) % Anormal por SOPRO", loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=6.5)
    fig.savefig(FIG / "fig1_eda.png")
    plt.close(fig)

    # Fig. 2: ROC (teste) + matriz de confusão do melhor modelo
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.55})
    ax = axs[0]
    for nome in MODELOS:
        fpr, tpr, _ = roc_curve(y_te, probas[nome])
        ax.plot(fpr, tpr, color=CORES[nome], lw=1.4 if nome == melhor else 1,
                label=nome.replace("Regressão Logística", "Reg. Logística").replace("HistGradientBoosting", "HistGB")
                + f" ({R['modelos'][nome]['teste']['roc_auc']:.2f})".replace(".", ","))
    ax.set_xlabel("1 − especificidade")
    ax.set_ylabel("Sensibilidade")
    ax.set_title("(a) Curvas ROC (teste)", loc="left")
    ax.legend(fontsize=5.6, loc="lower right", handlelength=1.2)
    ax.tick_params(labelsize=6.5)
    ax = axs[1]
    ax.imshow(cm, cmap="Blues", vmin=0)
    for (a, b), v in np.ndenumerate(cm):
        ax.text(b, a, f"{v:,}".replace(",", "."), ha="center", va="center", fontsize=7.5,
                color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["Normal", "Anormal"])
    ax.set_yticks([0, 1], ["Normal", "Anormal"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.grid(False)
    ax.set_title(f"(b) {melhor.replace('HistGradientBoosting', 'HistGB')}", loc="left")
    ax.tick_params(labelsize=6.5)
    fig.savefig(FIG / "fig2_roc_matriz.png")
    plt.close(fig)

    # Fig. 3: importância por permutação
    fig, ax = plt.subplots(figsize=(3.45, 1.7))
    top = imp.head(10)[::-1]
    ax.barh(top.index, top.values, xerr=imp_dp[top.index].values, color="#2a78d6", height=0.6,
            error_kw={"elinewidth": 0.6, "ecolor": INK2})
    ax.set_xlabel("Queda média de ROC-AUC ao permutar o atributo")
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=6.5)
    fig.savefig(FIG / "fig3_importancia.png")
    plt.close(fig)
    print("melhor:", melhor)
    print(json.dumps(R["sensibilidade"], ensure_ascii=False))


if __name__ == "__main__":
    main()
