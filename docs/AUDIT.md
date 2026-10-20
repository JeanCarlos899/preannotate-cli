# Auditoria do fluxo de propagação — 2026-10-20

Fluxo principal: exportar uma task em CVAT for images 1.1 com imagens; inspecionar referências manuais; comparar estratégias; condicionar SAM 2.1 com 1 a N referências da sequência; revisar overlays/vídeo; comparar suavização; exportar polígonos com sigma 0.8; conferir a prévia e acrescentar apenas frames livres no CVAT. O SAM usa pesos pré-treinados e não atualiza checkpoints.

Correções verificadas:

- Leitura do JSON multilinha do worker em execução local e SSH.
- Runtime por execução, resultados locais em pasta parcial antes da conclusão e caminhos locais no JSON final.
- Padrão `/home/getter/<usuário-local>/<projeto>`, com todos os parâmetros SSH/venv/checkpoint configuráveis.
- Referência única no meio da sequência passa a propagar também para trás; nenhuma limitação fixa de 3 referências ou corte silencioso em 12 referências externas.
- Propostas automáticas não são reutilizadas como referência humana. Seleção de prompts inexistentes falha explicitamente.
- Recorte e evidência na validação leave-one-out usam somente as referências do fold; métricas incluem todos os folds. A avaliação continua transdutiva/exploratória, pois usa imagens do clipe inteiro.
- Máscaras sem compressão com perdas, inclusive para nomes JPEG e subdiretórios.
- Treino opcional: paths resolvidos no servidor, modelos locais enviados, validação por task de origem ou cauda temporal, deduplicação de positivos e detecção de rótulos conflitantes. Inferência respeita classe/dispositivo solicitado.
- Propostas JSON incluem a ordem/nome dos frames para checagem pelo CVAT CLI 0.5. XML pode ser usado com `import-missing`.

Verificação real: SSH para RTX 5090, SAM 2.1 Hiera Large existente, sequência de 40 frames e somente frame 2 como prompt selecionado. Os 40 frames receberam logits, inclusive frame 0. Saída sincronizada, 49 polígonos exportados com sigma 0.8 em uma cópia com somente as três anotações manuais; frames 0,1,2 preservados. Gerado vídeo H.264 de 40 frames. Prévia de `cvat append` contra a task original aceitou o contrato e protegeu os frames ocupados. Nenhuma anotação remota foi alterada.

Esse teste verifica execução e integração, não comprova melhora de qualidade com um único prompt. O resultado anterior aprovado com três referências continua sendo a referência visual. Treino YOLO não foi executado nesta auditoria; seus caminhos foram inspecionados e corrigidos, sem tratar esse fluxo opcional como validado na GPU.

Limites: uma classe semântica por execução; regiões da mesma classe são unidas. Não há garantia de transferência SAM entre tasks independentes: `--ref-dataset` é experimental. Buracos em máscaras não são preservados por polígonos simples e são contabilizados no relatório. Proteção JSON por nomes e ordem não prova identidade de imagens com nomes idênticos; selecione a task correta e revise a prévia.
