# Preannotate CLI (`preannotate`)

CLI modular para pré-anotação de segmentação em vídeos e sequências de frames, projetada para ser orquestrada por agentes de IA (Codex, Claude Code, etc.) ou humanos em conjunto com a CLI [`cvat`](https://github.com/JeanCarlos899/cvat-cli).

Opera em arquitetura **híbrida**: roda na máquina local do usuário, sincroniza automaticamente os dados via SSH/`rsync` com uma pasta de trabalho isolada por usuário em um servidor GPU remoto, executa os modelos pesados (SAM 2.1, avaliação Leave-One-Out, ou treinamento YOLO-seg) e devolve para a máquina local as máscaras, relatórios, vídeos comparativos e pacotes de propostas prontos para `cvat append` ou `cvat import-missing`.

Referência completa de todos os comandos, entradas, saídas e contratos JSON: **[docs/CLI.md](docs/CLI.md)**.

O fluxo principal usa **SAM 2.1 pré-treinado, sem treinamento de pesos**. De 1 a N frames da mesma task podem servir como referências, consecutivos ou espalhados no vídeo. Todos são usados por padrão, sem teto de 3 ou 12. O desempenho depende de representatividade, resolução e memória disponível. Cada execução propaga uma máscara semântica por classe; objetos da mesma classe são unidos.

O diretório padrão é `/home/getter/<usuário-local>/preannotate`. Configure `--user` ou `PREANNOTATE_WORKSPACE_USER` para escolher a identidade. SSH continua configurável e obrigatório. Venv e checkpoint existentes podem ser reutilizados explicitamente com `PREANNOTATE_REMOTE_VENV` e `PREANNOTATE_SAM2_CHECKPOINT`; a CLI não cria versões incrementais de pesos SAM.

`--ref-dataset` é **experimental**: concatena imagens de outras tasks antes da sequência alvo. Isso não equivale a treino nem garante transferência entre câmeras/cenas. Prefira referências na própria task alvo e avalie visualmente. O fluxo B2 com treino YOLO permanece opcional e separado.

Somente shapes com `source=manual` (ou sem origem em XML antigo) viram referências. Propostas automáticas permanecem protegidas na exportação, mas não são verdade humana. Frames sem shape não são automaticamente exemplos negativos revisados.

---

## Instalação

Requer Python 3.10+ (e `ssh` / `rsync` para execução remota; `ffmpeg` opcional para codificação H.264).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/preannotate --help
.venv/bin/preannotate --version
```

Ou globalmente com `pipx`:

```sh
pipx install .
```

---

## Configuração de SSH e Pasta de Trabalho (`~/.config/preannotate/.env`)

As configurações de servidor e caminho de trabalho ficam fora do repositório, em `~/.config/preannotate/.env` com permissão `0600`:

```sh
mkdir -p ~/.config/preannotate
chmod 700 ~/.config/preannotate
cp perfil.env.example ~/.config/preannotate/.env
chmod 600 ~/.config/preannotate/.env
```

Edite `~/.config/preannotate/.env`:

```ini
# Pode ser um alias configurado no ~/.ssh/config (ex: gpu-server) ou usuario@ip
PREANNOTATE_SSH_TARGET=gpu-server
PREANNOTATE_SSH_PORT=
PREANNOTATE_SSH_IDENTITY=

# Diretório raiz no servidor e nome da pasta do usuário.
# O CLI criará e usará: <REMOTE_ROOT>/<WORKSPACE_USER>/<WORKSPACE_DIR>
# Exemplo: /home/getter/meu_usuario/pasta_do_anotador
PREANNOTATE_REMOTE_ROOT=/home/getter
PREANNOTATE_WORKSPACE_USER=meu_usuario
PREANNOTATE_WORKSPACE_DIR=pasta_do_anotador

# Opcional: caminho para reutilizar venv ou checkpoint já existentes no servidor
PREANNOTATE_REMOTE_VENV=
PREANNOTATE_SAM2_CHECKPOINT=
PREANNOTATE_SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
```

### Verificar e preparar o servidor GPU

```sh
# 1. Verificar conectividade SSH, GPU disponível e caminhos resolvidos
preannotate remote-check

# 2. Ver prévia do que será instalado na pasta remota do usuário
preannotate remote-setup

# 3. Criar pasta remota, instalar ambiente Python/CUDA/SAM 2.1 e baixar checkpoint
preannotate remote-setup --apply
```

---

## Fluxos Recomendados para Agentes de IA

Esta CLI não esconde as decisões em um comando único opaco; cada etapa é uma ferramenta atômica que emite JSON em `stdout` e gera artefatos visuais para o agente e o usuário inspecionarem antes de subir ao CVAT.

### Fluxo A — Propagação Temporal com SAM 2.1 (1 a N frames anotados no vídeo)

Ideal quando o usuário anotou qualquer número de frames representativos (ex.: início, expansão e dissipação) na própria task do CVAT:

```sh
# 1. Baixar dataset (imagens + annotations.xml em CVAT 1.1) usando a CLI cvat
cvat export 2326 dataset.zip --images --format "CVAT for images 1.1"

# 2. Inspecionar frames, resolução e quais índices já possuem anotação manual
preannotate inspect-dataset dataset.zip --label smoke

# 3. (Se houver >= 2 frames anotados) Avaliar estratégias com Leave-One-Out na GPU remota
preannotate evaluate --dataset dataset.zip --out resultados/avaliacao --label smoke

# 4. Executar a propagação escolhida no servidor GPU (ex.: SAM 2.1 imagem inteira ou recorte+fundo)
preannotate propagate \
  --dataset dataset.zip \
  --out resultados/sam2 \
  --label smoke \
  --mode full \
  --fusion none

# 5. Gerar prancha comparativa de níveis de suavização de contorno (subpixel) para revisão visual
preannotate smooth \
  --dataset dataset.zip \
  --masks resultados/sam2/mascaras \
  --out resultados/suavizacao \
  --sigma 0.8 \
  --compare-levels 0,0.5,0.8,1.2,1.6,2.0 \
  --frames 3,20,30

# 6. Gerar vídeo H.264 lado a lado (Original vs Previsão) para inspeção no chat
preannotate render-video \
  --dataset dataset.zip \
  --masks resultados/suavizacao/mascaras \
  --titles "SAM 2.1 (sigma 0.8)" \
  --out resultados/video

# 7. Exportar propostas finais (preservando frames manuais intactos)
preannotate export-proposals \
  --dataset dataset.zip \
  --masks resultados/sam2/mascaras \
  --out resultados/propostas_finais \
  --label smoke \
  --sigma 0.8

# 8. Conferir prévia e enviar apenas nos frames vazios da task no CVAT
cvat append 2326 --json resultados/propostas_finais/proposals.json
cvat append 2326 --json resultados/propostas_finais/proposals.json --apply \
  --backup backups/antes.json --receipt receipts/envio.json
```

---

### Fluxo B — Reutilizando Tasks Anteriores do CVAT (Fusão de Datasets + Condicionamento ou Treino)

Ideal quando já existem outras tasks anotadas no CVAT e você quer aproveitar esse conhecimento para pré-anotar uma nova task:

#### Opção B1 (experimental): Condicionar a memória do SAM 2.1 com tasks anteriores (`--ref-dataset`)

```sh
# 1. Baixar as tasks anteriores já anotadas e a task nova
cvat export 2310 task_antiga_1.zip --images --format "CVAT for images 1.1"
cvat export 2315 task_antiga_2.zip --images --format "CVAT for images 1.1"
cvat export 2326 task_nova.zip --images --format "CVAT for images 1.1"

# 2. Propagar na task nova alimentando o SAM 2.1 com as referências das tasks antigas
preannotate propagate \
  --dataset task_nova.zip \
  --ref-dataset task_antiga_1.zip \
  --ref-dataset task_antiga_2.zip \
  --out resultados/sam2_multitask \
  --label smoke
```

#### Opção B2: Mesclar datasets e treinar um modelo de segmentação no servidor GPU (`merge-datasets` + `train-seg` + `predict-seg`)

```sh
# 1. Mesclar as tasks baixadas em um dataset unificado com split YOLO-seg
preannotate merge-datasets \
  --inputs task_antiga_1.zip task_antiga_2.zip \
  --out dataset_mesclado \
  --label smoke \
  --yolo-seg

# 2. Treinar o modelo de segmentação no servidor GPU remoto
preannotate train-seg \
  --dataset dataset_mesclado \
  --out resultados/treino_seg \
  --label smoke \
  --epochs 25

# 3. Predizer nos frames da task nova (opcionalmente refinando com SAM 2.1 temporal)
preannotate predict-seg \
  --dataset task_nova.zip \
  --weights resultados/treino_seg/best.pt \
  --refine-sam2 \
  --out resultados/predicao_nova_task \
  --label smoke

# 4. Exportar propostas suavizadas e subir no CVAT
preannotate export-proposals \
  --dataset task_nova.zip \
  --masks resultados/predicao_nova_task/mascaras \
  --out resultados/propostas_nova_task \
  --label smoke \
  --sigma 0.8

cvat append 2326 --json resultados/propostas_nova_task/proposals.json --apply \
  --backup backups/antes.json --receipt receipts/envio.json
```

---

## Desenvolvimento e Verificação

```sh
.venv/bin/python -m pip install '.[dev]'
.venv/bin/python -m unittest -v
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m compileall -q preannotate_cli.py preannotate_core.py preannotate_worker.py
.venv/bin/python -m build
```
