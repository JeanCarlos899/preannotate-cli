# Contrato da CLI `preannotate`

Este documento é a referência operacional completa para pessoas e agentes de IA (Codex, Claude Code, etc.) orquestrarem o fluxo de pré-anotação de segmentação em conjunto com a CLI `cvat`. O único ponto de entrada público é:

```sh
preannotate [opções-globais] COMANDO [argumentos]
```

Todas as saídas de comandos em `stdout` são objetos JSON estruturados. Erros são emitidos em `stderr` com código de saída diferente de zero. Nenhum comando sobrescreve diretórios ou arquivos locais de saída existentes (`--out` deve ser sempre um caminho novo).

---

## Opções globais e perfil SSH

| Opção | Tipo / Padrão | Descrição |
| --- | --- | --- |
| `--env ARQUIVO` | `~/.config/preannotate/.env` | Arquivo de perfil `KEY=value` (permissão `0600` exigida no POSIX). |
| `--ssh ALVO` | `PREANNOTATE_SSH_TARGET` | Alias do `~/.ssh/config` (ex.: `gpu-server`) ou `usuario@host` / `ssh://usuario@host:porta`. |
| `--remote-root CAMINHO` | `PREANNOTATE_REMOTE_ROOT` | Diretório base no servidor SSH (ex.: `/home/ubuntu`). |
| `--user USUARIO` | `PREANNOTATE_WORKSPACE_USER` | Nome da pasta do usuário dentro de `--remote-root` (ex.: `meu_usuario`). |
| `--project PROJETO` | `PREANNOTATE_WORKSPACE_DIR` | Nome da subpasta do projeto dentro da pasta do usuário (ex.: `pasta_do_anotador`). |
| `--version` | — | Imprime a versão instalada e encerra. |

### Resolução do diretório de trabalho remoto

O caminho remoto isolado de cada usuário e projeto é sempre resolvido como:

```text
<PREANNOTATE_REMOTE_ROOT>/<PREANNOTATE_WORKSPACE_USER>/<PREANNOTATE_WORKSPACE_DIR>
```

Exemplo: se `PREANNOTATE_REMOTE_ROOT=/home/servidor`, `PREANNOTATE_WORKSPACE_USER=meu_usuario` e `--project pasta_do_anotador`, todos os ambientes, caches e execuções remotas ficam dentro de `/home/servidor/meu_usuario/pasta_do_anotador`.

Variáveis de ambiente (`PREANNOTATE_*`) têm prioridade sobre o arquivo `.env`, e opções de linha de comando (`--ssh`, `--remote-root`, `--user`, `--project`) têm prioridade sobre ambas.

---

## 1. Ambiente remoto e diagnóstico

| Comando | Entradas | Saída JSON (`stdout`) | Efeito remoto |
| --- | --- | --- | --- |
| `remote-check` | nenhuma | `local_tools`, `ssh_host`, `workspace_dir`, `gpus`, `runtime`, `sam2_ready` | consulta SSH sem alterar arquivos |
| `remote-setup [--apply] [--skip-sam2] [--with-ultralytics]` | flags opcionais | prévia (`dry_run: true`) sem `--apply`; status completo do ambiente com `--apply` | cria `<workspace_dir>`, `.venv`, instala dependências CUDA/SAM 2 e baixa checkpoint `sam2.1_hiera_large.pt` |

- Por padrão, `remote-setup` apenas exibe o plano (`dry_run: true`). Passe `--apply` para executar a instalação no servidor remoto.

---

## 2. Inspeção e fusão de datasets CVAT 1.1

| Comando | Entradas | Saída JSON (`stdout`) | Efeito remoto |
| --- | --- | --- | --- |
| `inspect-dataset DATASET [--xml ARQUIVO] [--label NOME]` | pasta ou `.zip` CVAT 1.1; `--label` padrão `smoke` | `total_frames`, `width`, `height`, `labels_present`, `annotated_frames`, `empty_frames`, `frames` | nenhum (100% local) |
| `merge-datasets --inputs ITEM... --out PASTA [--label NOME] [--yolo-seg] [--val-ratio R] [--seed N]` | 1 ou mais pastas/ZIPs CVAT 1.1; `--out` novo | `output_dir`, `annotations_xml`, `total_frames`, `annotated_frames`, `unannotated_frames`, `yolo_seg` | nenhum (100% local) |

- `DATASET` pode ser uma pasta contendo `annotations.xml` + imagens ou diretamente o arquivo `.zip` baixado por `cvat export <TASK_ID> dataset.zip --images --format "CVAT for images 1.1"`.
- A ordem sequencial dos frames sempre segue `int(image.attrib["id"])` do `annotations.xml` (e não os números presentes no nome do arquivo de origem).
- Com `--yolo-seg`, `merge-datasets` também gera `yolo_seg/dataset.yaml`, `images/{train,val}` e `labels/{train,val}` sem duplicar imagens idênticas (verificadas por SHA-256).

---

## 3. Propagação e avaliação de máscaras (Híbrido SSH / Local)

Por padrão, estes comandos enviam os dados via `rsync` para `<workspace_dir>/runs/<run_id>` no servidor SSH, executam na GPU remota e sincronizam os resultados de volta para `--out` na máquina local. Passe `--local` para executar na própria máquina sem SSH.

| Comando | Entradas | Saída JSON (`stdout`) | Artefatos gerados em `--out` |
| --- | --- | --- | --- |
| `baselines --dataset D --out PASTA [--label L] [--local]` | dataset CVAT 1.1, pasta nova | `methods`, `comparison_sheet`, `metrics_json`, `frames` | `fluxo/`, `fluxo_grabcut/`, `aparencia_local/`, `comparacao_baselines.jpg`, `metricas.json` |
| `propagate --dataset D --out PASTA [opções]` | dataset CVAT 1.1, pasta nova | `masks_dir`, `overlays_dir`, `logits_npz`, `predictions_xml`, `prompt_frames`, `elapsed_seconds` | `mascaras/`, `sobreposicoes/`, `logits.npz`, `predicoes_cvat_1.1.xml`, `relatorio.json` |
| `evaluate --dataset D --out PASTA [opções]` | dataset com `>= 2` frames anotados, pasta nova | `annotated_frames`, `crop_band_y`, `summary_by_method`, `records`, `elapsed_seconds` | `avaliacao.json`, `resumo_metricas.json`, `registro_camera.json`, `final_inteiro_sam/`, `final_recorte_sam_fundo/` |

### Opções de `propagate`

- `--label NOME`: classe alvo no XML (padrão: `smoke`).
- `--mode full|crop`: propaga na imagem inteira (`full`, padrão) ou em faixa vertical recortada (`crop`).
- `--crop-y Y0:Y1`: limites verticais em pixels (ex.: `300:620`); se omitido em `--mode crop`, é calculado automaticamente ao redor das máscaras manuais com margem.
- `--fusion none|background|background-zero|temporal`:
  - `none` (padrão): usa apenas os logits do SAM 2.1.
  - `background`: registra a câmera com ORB+RANSAC e combina os logits do SAM 2.1 com evidência de mudança contra o fundo percentil 25 do clipe.
  - `background-zero`: usa a área fora da máscara inicial no frame 0 como fundo conhecido após compensar movimento da câmera e exposição.
  - `temporal`: combina logits do SAM 2.1 com diferença entre frames adjacentes alinhados.
- `--threshold FLOAT`: limiar de logit do SAM 2.1 (padrão: `0.0`).
- `--prompts LISTA`: índices zero-based separados por vírgula (ex.: `0,1,2`) para usar como referências; por padrão usa todos os frames com anotação manual no dataset.
- `--ref-dataset CAMINHO` (repetível): datasets CVAT 1.1 adicionais (de outras tasks já anotadas) cujos frames anotados entram no banco de memória do SAM 2.1 antes de propagar no vídeo alvo.
- `--checkpoint CAMINHO` e `--config NOME`: sobrescrevem o checkpoint e config do SAM 2.1.
- `--device cuda|cpu`: dispositivo de execução (padrão: `cuda`).
- `--local`: executa localmente sem SSH.

---

## 4. Treinamento e predição de modelo de segmentação multi-task

Quando o usuário deseja mesclar tasks anteriores já anotadas no CVAT para treinar um modelo de segmentação dedicado no servidor GPU e depois preanotar uma nova task:

| Comando | Entradas | Saída JSON (`stdout`) | Artefatos gerados em `--out` |
| --- | --- | --- | --- |
| `train-seg --dataset D --out PASTA [--label L] [--model M] [--epochs N] [--imgsz S] [--batch B] [--device DEV] [--local]` | dataset CVAT 1.1 ou pasta gerada por `merge-datasets` | `weights_best`, `dataset_yaml`, `epochs`, `elapsed_seconds` | `best.pt`, `treino_relatorio.json`, `yolo_seg_run/` |
| `predict-seg --dataset D --weights PESOS.pt --out PASTA [--label L] [--conf C] [--imgsz S] [--refine-sam2] [--checkpoint CKPT] [--local]` | dataset alvo e pesos `.pt` | `masks_dir`, `overlays_dir`, `refined_with_sam2`, `frames` | `mascaras/`, `sobreposicoes/`, `predicao_relatorio.json` |

- `--refine-sam2` em `predict-seg` usa as máscaras previstas pelo modelo treinado como prompts iniciais para o SAM 2.1 propagar e estabilizar temporalmente a segmentação ao longo do vídeo.

---

## 5. Refinamento local, visualização e exportação para o `cvat` CLI

Estes comandos rodam localmente sobre as máscaras/logits já trazidos do servidor, sem exigir GPU:

| Comando | Entradas | Saída JSON (`stdout`) | Artefatos gerados em `--out` |
| --- | --- | --- | --- |
| `thresholds --dataset D --logits ARQUIVO.npz --out PASTA [--levels L] [--frames F]` | dataset e `logits.npz`; `--levels` padrão `0,2,4,6` | `sheet`, `thresholds`, `sample_frames` | `limiares.jpg`, `limiares.json`, `limiar_<N>/mascaras/` |
| `smooth --dataset D --masks PASTA_MASCARAS --out PASTA [--sigma S] [--compare-levels L] [--frames F] [--step P]` | dataset e pasta de máscaras PNG; `--sigma` padrão `0.8` | `masks_dir`, `overlays_dir`, `sigma_px`, `comparison_sheets` | `mascaras/`, `sobreposicoes/`, `parametros.json`, `comparacao_frame_<NN>.png` |
| `render-video --dataset D --masks PASTA... --out PASTA [--titles T...] [--fps N] [--crop-y Y0:Y1] [--gif]` | dataset e 1 ou mais pastas de máscaras | `video_mp4`, `video_gif`, `inspect_dir`, `frames`, `panels` | `demonstracao_h264.mp4`, `demonstracao.gif` (opcional), `inspecao/`, `video_summary.json` |
| `export-proposals --dataset D --masks PASTA_MASCARAS --out PASTA [--label L] [--sigma S] [--min-area A] [--step P]` | dataset original, pasta de máscaras e `--sigma` (padrão `0.8`) | `proposals_json`, `annotations_xml`, `predictions_zip`, `shapes_created`, `preserved_frames`, `filled_frames`, `empty_frames` | `proposals.json`, `annotations.xml`, `predictions_cvat_1.1.zip`, `export_summary.json` |

### Garantias de segurança na exportação (`export-proposals`)

1. **Preservação integral de anotações manuais:** qualquer frame que já possua anotação no `annotations.xml` de entrada é listado em `preserved_frames`, **não** recebe polígonos em `proposals.json` e mantém seus polígonos manuais originais intactos em `annotations.xml` / `predictions_cvat_1.1.zip`.
2. **Compatibilidade direta com `cvat` CLI:**
   - Para envio via JSON incremental:
     ```sh
     cvat append <TASK_ID> --json <OUT>/proposals.json
     cvat append <TASK_ID> --json <OUT>/proposals.json --apply --backup backups/antes.json --receipt receipts/envio.json
     ```
   - Para envio via XML/ZIP CVAT 1.1 incremental:
     ```sh
     cvat import-missing <TASK_ID> <OUT>/predictions_cvat_1.1.zip --dry-run
     cvat import-missing <TASK_ID> <OUT>/predictions_cvat_1.1.zip --apply --backup backups/antes.json --receipt receipts/envio.json
     ```
