"""Gera relatorio/numeros.tex (macros LaTeX) a partir de results/*.json: nenhum número do artigo é digitado à mão.

Executar depois de src/modelagem.py:  python relatorio/gerar_numeros.py
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
R = json.loads((RAIZ / "results" / "metricas.json").read_text(encoding="utf-8"))
L = json.loads((RAIZ / "results" / "limpeza.json").read_text(encoding="utf-8"))


def n(x):  # inteiro com separador de milhar
    return f"{int(x):,}".replace(",", ".")


def d(x, k=3):  # decimal com vírgula
    return f"{x:.{k}f}".replace(".", ",")


M = {}
M["Nbruto"], M["Pbruto"], M["Nlimpo"] = n(L["n_bruto"]), str(L["p_bruto"]), n(L["n_limpo"])
M["Vazios"], M["Dup"], M["AlvoAus"], M["Contra"] = n(L["vazios"]), n(L["duplicatas"]), n(L["alvo_ausente"]), n(L["contraditorios"])
M["Nnormal"], M["Nanormal"] = n(R["eda"]["alvo"]["Normal"]), n(R["eda"]["alvo"]["Anormal"])
M["PctAnormal"] = d(R["eda"]["pct_anormal"], 1)
M["Ntreino"], M["Nteste"] = n(R["divisao"]["treino"]), n(R["divisao"]["teste"])
M["SoproSist"] = d(R["eda"]["sopro"]["Sistólico"]["pct_anormal"], 0)
M["SoproAus"] = d(R["eda"]["sopro"]["Ausente"]["pct_anormal"], 0)
M["AusAltura"], M["AusPeso"] = d(R["eda"]["ausentes_pct"]["Altura"], 0), d(R["eda"]["ausentes_pct"]["Peso"], 0)

melhor = R["melhor_modelo"]
mb = R["modelos"][melhor]
M["Melhor"] = melhor
M["MelhorAUCcv"], M["MelhorAUCcvdp"] = d(mb["cv_media"]["roc_auc"]), d(mb["cv_dp"]["roc_auc"])
for k, nome in [("AUC", "roc_auc"), ("Acc", "accuracy"), ("BAcc", "balanced_accuracy"), ("Prec", "precision"),
                ("Rec", "recall"), ("Fum", "f1")]:
    M[f"Melhor{k}"] = d(mb["teste"][nome])
p = mb["melhores_params"]
M["RFparams"] = (f"min\\_samples\\_leaf={p['min_samples_leaf']}, max\\_depth={p['max_depth']}, "
                 f"class\\_weight={p['class_weight']}")
pl = R["modelos"]["Regressão Logística"]["melhores_params"]
M["LRparams"] = f"C={pl['C']}, class\\_weight={pl['class_weight']}"
ph = R["modelos"]["HistGradientBoosting"]["melhores_params"]
M["HGBparams"] = f"learning\\_rate={ph['learning_rate']}, max\\_depth={ph['max_depth']}, max\\_iter={ph['max_iter']}"
M["LRAUC"] = d(R["modelos"]["Regressão Logística"]["teste"]["roc_auc"])
M["HGBAUC"] = d(R["modelos"]["HistGradientBoosting"]["teste"]["roc_auc"])
M["LRRec"] = d(R["modelos"]["Regressão Logística"]["teste"]["recall"])
M["DummyAcc"] = d(R["modelos"]["Dummy"]["teste"]["accuracy"])

(tn, fp), (fn, tp) = R["matriz_confusao"]["linhas_real_colunas_previsto"]
M["TN"], M["FP"], M["FN"], M["TP"] = n(tn), n(fp), n(fn), n(tp)
M["Nanormalteste"] = n(fn + tp)
lim = R["limiar"]
M["Limiar"] = d(lim["limiar"], 2)
M["LimRec"], M["LimPrec"], M["LimBAcc"] = d(lim["teste"]["recall"]), d(lim["teste"]["precision"]), d(lim["teste"]["balanced_accuracy"])
(ltn, lfp), (lfn, ltp) = lim["matriz_confusao"]
M["LimFN"], M["LimFP"] = n(lfn), n(lfp)

ef = R["erros_por_faixa"]
M["RecMenorUm"], M["RecAdol"] = d(ef["<1"]["recall"], 2), d(ef["11-19"]["recall"], 2)
M["RecUmTres"], M["RecSeteOnze"] = d(ef["1-3"]["recall"], 2), d(ef["7-11"]["recall"], 2)

imp = R["importancia_permutacao"]["media"]
for k, c in [("ImpMotivoDois", "MOTIVO2"), ("ImpMotivoUm", "MOTIVO1"), ("ImpBdois", "B2"), ("ImpFC", "FC"),
             ("ImpHDA", "HDA 1"), ("ImpIdade", "IDADE")]:
    M[k] = d(imp[c])
coef = R["coef_regressao_logistica"]
M["CoefBhiper"] = d(coef["nom__B2_Hiperfonética"], 2)
M["CoefPulsos"] = d(coef["nom__PULSOS_Femorais diminuídos"], 2)
M["CoefMotSopro"] = d(coef["nom__MOTIVO2_6 - Sopro"], 2)
M["CoefBnormal"] = d(coef["nom__B2_Normal"], 2)

s = R["sensibilidade"]["resultados"]
for k, rot in [("Final", "modelo final (sem SOPRO)"), ("Sopro", "com SOPRO (leakage)"), ("SemMot", "sem SOPRO e sem MOTIVO1/2")]:
    M[f"Sens{k}AUC"] = d(s[rot]["cv_media"]["roc_auc"])
    M[f"Sens{k}Rec"] = d(s[rot]["cv_media"]["recall"])
    M[f"Sens{k}BAcc"] = d(s[rot]["cv_media"]["balanced_accuracy"])

# linhas da tabela comparativa
nomes = {"Dummy": "Dummy (majoritária)", "Regressão Logística": "Reg. Logística",
         "Random Forest": "Random Forest", "HistGradientBoosting": "HistGB"}
linhas = []
for nome, m in R["modelos"].items():
    cv, dp, te = m["cv_media"], m["cv_dp"], m["teste"]
    lab = f"\\textbf{{{nomes[nome]}}}" if nome == melhor else nomes[nome]
    linhas.append(f"{lab} & {d(cv['roc_auc'])}$\\pm${d(dp['roc_auc'])} & {d(te['roc_auc'])} & "
                  f"{d(te['balanced_accuracy'])} & {d(te['precision'])} & {d(te['recall'])} & {d(te['f1'])} \\\\")
M["TabModelos"] = "\n".join(linhas)

# tabela de atributos (base limpa)
import pandas as pd
df = pd.read_csv(RAIZ / "data" / "UCMF_limpo.csv", encoding="utf-8")


def aus(c):
    s = df[c]
    return d(100 * (s.isna() | (s == "Não informado")).mean(), 0) + "\\%"


attr = [("IDADE, Peso, Altura", "numérico", f"{aus('IDADE')}, {aus('Peso')}, {aus('Altura')}", "entrada"),
        ("FC", "numérico", aus("FC"), "entrada"),
        ("PPA", "ordinal", d(100 * df["PPA"].isin(["Não aferida", "Não calculável"]).mean(), 0) + "\\%$^a$", "entrada"),
        ("SEXO, B2, PULSOS", "nominal", f"{aus('SEXO')}, {aus('B2')}, {aus('PULSOS')}", "entrada"),
        ("HDA 1, MOTIVO1, MOTIVO2", "nominal", f"{aus('HDA 1')}, {aus('MOTIVO1')}, {aus('MOTIVO2')}", "entrada"),
        ("SOPRO", "nominal", aus("SOPRO"), "excluído (vazamento)"),
        ("PAS, PAD; HDA2", "num.; nominal", f"{aus('PA SISTOLICA')}; {aus('HDA2')}", "excluídos ($>$50\\% aus.)"),
        ("IMC", "numérico", aus("IMC"), "excluído (derivado)"),
        ("ID, Convenio, datas", "--", "--", "excluídos (sem relação)"),
        ("NORMAL X ANORMAL", "binário", "0\\%", "alvo")]
M["TabAtributos"] = "\n".join(f"{a} & {b} & {c} & {e} \\\\" for a, b, c, e in attr)

out = ["% Gerado por relatorio/gerar_numeros.py a partir de results/*.json - nao editar a mao"]
for k, v in M.items():
    out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(RAIZ / "relatorio" / "numeros.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print(len(M), "macros")
