# Predição de cardiopatia em crianças e adolescentes (UCMF)

Trabalho da disciplina **Mineração de Dados** (Prof. Rogerio Salvini, 2026), Grupo 9: Kauã Sandes, Lucas Nogueira, Luccas José e Yago Coqueiro.

O objetivo é classificar pacientes de 0 a 19 anos atendidos no Real Hospital Português (Recife-PE) como **Normal** ou **Anormal**, ou seja, sem ou com patologia cardíaca.

## Como reproduzir

```bash
pip install -r requirements.txt
python src/limpeza.py      # data/UCMF_raw.csv -> data/UCMF_limpo.csv + results/limpeza.json
python src/modelagem.py    # results/metricas.json + results/figuras/*.png
python relatorio/gerar_numeros.py              # números do artigo (relatorio/numeros.tex)
cd relatorio && pdflatex main.tex && pdflatex main.tex   # artigo IEEE
```

Todos os sorteios usam `random_state = 42`. Duas execuções seguidas geram o mesmo `metricas.json`; só os tempos de ajuste mudam. O teste completo leva cerca de 2 minutos numa CPU comum. A versão do Python usada foi a 3.14.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `data/UCMF_raw.csv` | Base original (12.873 registros, 21 atributos), anonimizada e com uso acadêmico aprovado pelos Comitês de Ética do RHP e da Universidade do Porto |
| `data/tabela_pa.json` | Percentis de pressão arterial pediátrica por sexo e idade (NHBPEP, 2004), usados para recalcular a PPA |
| `src/limpeza.py` | Limpeza definida na Avaliação A1: vazios, duplicatas, padronização léxica, IDADE recalculada a partir das datas, valores impossíveis tratados como ausentes, PPA recalculada e exclusão de contradições |
| `src/modelagem.py` | Divisão treino/teste, pipelines, busca de hiperparâmetros, avaliação, análise de sensibilidade e figuras |
| `results/` | Métricas (JSON) e figuras do relatório |
| `relatorio/` | Artigo no formato IEEE (LaTeX e PDF) |

## Protocolo experimental

- **Divisão:** holdout estratificado 80/20, com 9.169 objetos para treino e 2.293 para teste. O conjunto de teste é usado uma única vez para cada modelo.
- **Ajuste de hiperparâmetros:** `GridSearchCV` com `StratifiedKFold(5)` apenas no treino, escolhendo pela ROC-AUC.
- **Pré-processamento:** fica todo dentro do `Pipeline` e é ajustado só nas dobras de treino. Inclui imputação pela mediana com indicador de ausência, log1p, padronização z-score, codificação ordinal da PPA e one-hot dos atributos nominais.
- **Atributo SOPRO:** excluído por vazamento de dados (data leakage). Ele só aparece no experimento de sensibilidade, que mostra o quanto inflaria as métricas.
- **Modelos comparados:** Dummy (classe majoritária), Regressão Logística (baseline), Random Forest e HistGradientBoosting.
