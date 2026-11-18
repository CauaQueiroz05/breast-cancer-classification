# Classificacao - Breast Cancer (UCI)

Trabalho de Data Science utilizando a base Breast Cancer da UCI.

## Arquivos
- `breast-cancer.csv`: base de dados utilizada
- `train.py`: pre-processamento, validacao cruzada, selecao do modelo com maior acuracia e salvamento do pipeline
- `inference.py`: carregamento do modelo treinado para predicao do atributo `Class` de uma nova paciente

## Execucao

```bash
pip install -r requirements.txt
python train.py
python inference.py
