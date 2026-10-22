# cnx — Motor de vetorização raster→vetor

**Documento de arquitetura e formulação matemática, v0.1 (2026-10-21)**

Este documento parte da especificação recebida (§§1–8) e a detalha em arquitetura, estruturas de dados e matemática. A seção 0 lista os pontos em que eu recomendo divergir da especificação, com a justificativa. As demais seções já seguem essas decisões.

Convenção: **[V]** marca uma fórmula verificada numericamente pelo script [`tests/math/validate_math.py`](../tests/math/validate_math.py) (ver §2.6).

---

## 0. Revisão técnica da especificação

| # | Ponto da especificação | Problema | Decisão |
|---|---|---|---|
| 1 | Uma `QThread` por estágio | Os estágios são **sequencialmente dependentes**: o estágio *k+1* só começa quando o *k* termina. Seis threads não paralelizam nada; só acrescentam sincronização. Além disso, acoplar o núcleo ao Qt impede testes e benchmarks headless e o uso como biblioteca. | Núcleo em C++20 **sem Qt**, com um orquestrador numa *worker thread* (`std::jthread` + `std::stop_token`). O paralelismo é **de dados**, dentro de cada estágio: cada K da paleta, faixas de linhas, avaliação de energia e gradiente, fragmentos de Bézier. Também rodam em paralelo os três consumidores finais (salvar, renderizar e gerar miniaturas). Um adaptador `EngineController : QObject` roda o orquestrador numa `QThread` e converte callbacks em sinais. As faixas de progresso da especificação são mantidas. |
| 2 | Desfoque gaussiano no pré-processamento **e** `GenerativeModel` | O modelo gerador sintetiza pixels com AA de filtro-caixa e os compara com o alvo. Se o alvo for a imagem **desfocada**, o modelo direto não é o processo que gerou a imagem. Bordas retas isoladas ficam no lugar por simetria, mas traços finos (largura comparável a σ) e cantos ficam enviesados: o filtro-caixa não reproduz a rampa alargada e compensa deslocando a geometria. | O alvo do `GenerativeModel` é a `SignalImage` reamostrada **antes** do desfoque. Opcionalmente, o modelo direto passa a ser `K_σ * R(V)`, com o mesmo núcleo e o gradiente pelo adjunto (convolução transposta). O desfoque continua servindo à paleta e à segmentação. |
| 3 | "Escala multiplicativa", exemplo `(13 − v)·0.0001` | O exemplo é **afim**, não multiplicativo. | `p(v) = p_min · (p_max/p_min)^((v − v_min)/(v_max − v_min))`: passos iguais do controle multiplicam o parâmetro pelo mesmo fator (`cnx::multiplicativeScale`). |
| 4 | `WeightsFinder`: 3 a 9 cores por Cholesky | Em RGBA (d = 4), com Σw = 1, o espaço afim tem dimensão 4. Pelo teorema de Carathéodory, qualquer mistura é representável com **no máximo 5 cores**. Para K ≥ 6, `CᵀC` (K×K) tem posto ≤ 4: é **singular** e a decomposição de Cholesky falha ou fica mal posta. Além disso, w ≥ 0 exige um método de conjunto ativo. | Mínimos quadrados não negativos no simplex (conjunto ativo de Lawson–Hanson). Em cada passo, Cholesky do sistema reduzido ao conjunto ativo (\|S\| ≤ 5), com regularização de Tikhonov λ·I. |
| 5 | BLAS/LAPACK para sistemas 2×2, 4×4 e ≤ 9×9 | Em matrizes desse tamanho, o custo da chamada Fortran (sem *inlining* e sem vetorização entre chamadas) domina o custo da conta. | `sla::linalg` com núcleos inline de tamanho fixo para n ≤ 16. LAPACK/BLAS (`dpotrf_`/`dpotrs_`, `dgesv_`, `dgemm_`, `dgemv_`) ficam para operações **em lote**, como montar de uma vez, com um `dgemm_`, as Gram de milhares de pixels da mesma combinação de cores. O caminho LAPACK também fica como implementação de referência nos testes de regressão. |
| 6 | Ajuste de Bézier sem continuidade de tangente | Fragmentos ajustados independentemente, com P1 e P2 livres, formam **quinas** visíveis nos pontos de quebra que não são cantos. | Nas quebras "lisas" (que não são canto perfurado nem junção), a tangente τ é compartilhada: P1 = P0 + α·τ0 e P2 = P3 − β·τ3, com incógnitas (α, β). Tangentes livres só em cantos e junções (§2.5.4). |
| 7 | Refinamento por Newton | A alternância "mínimos quadrados ↔ reparametrização" (Schneider/Hoschek) converge **linearmente**. | Newton **exato e conjunto** em (P1, P2, t_1..t_m), resolvido pelo complemento de Schur da Hessiana em seta, com custo **O(m)** por iteração e convergência quadrática, mais amortecimento de Levenberg–Marquardt. Em 8 iterações, a energia ficou em 0.034878 contra 0.053870 da alternância (§2.6). |
| 8 | `ANN` kd-tree para o fiscal anti-cruzamento | O ANN indexa **pontos**, mas cruzamento é propriedade de **segmentos**. Como os nós se movem a cada passo do CG, a árvore teria de ser reconstruída em O(n log n) a cada passo. | Interface `ISpatialIndex` com dois backends. O padrão é uma **grade uniforme de segmentos** (célula de 2 px, reconstrução em O(n), consulta em O(1), já que os segmentos são curtos e o domínio é limitado). O backend **ANN** fica para consultas de ponto (zona livre da mola de fidelidade, métricas Hausdorff e Chamfer contra gabaritos) e como linha de base nos benchmarks. |
| 9 | "Verificar se a troca partiria a região" | Um teste global (BFS) custa O(\|R\|) por pixel testado. | Teste local de **ponto simples** pelos números topológicos (T4, T8) em N8(p), em O(1). É condição suficiente: se p é simples, removê-lo não divide R nem cria ou destrói furos (§1.5). |
| 10 | Mediana por canal sob fusões | Recalcular a mediana do zero a cada fusão custa O(\|R\| log \|R\|). | Acumulador híbrido. Até 1024 amostras guarda os pixels brutos (4 KiB). Acima disso, usa um histograma de 4×256 posições (4 KiB). Fusão em O(256), mediana em O(256). A remoção (para "trocar") também é suportada. |
| 11 | k-means sobre pixels para K = 2..12, com 10 rodadas cada | Custo Σ_K K·R·N ≈ 770·N avaliações de distância. | k-means sobre o **coreset ponderado** formado pelas células ocupadas do histograma (média e peso de cada célula), com custo independente de N. Depois, refinamento de 1 a 2 rodadas sobre os pixels só para o K escolhido. As sementes saem dos picos do histograma, e as faltantes são sorteadas por k-means++ (amostragem D² com `myPrng`) em vez de uniformemente. |
| 12 | Union-Find "cadeia de ponteiros" e `std::set` de vizinhos | Sem compressão de caminho e união por tamanho, as cadeias degeneram para O(N). O `std::set` aloca um nó por elemento e tem má localidade de cache. | Compressão de caminho por *halving* e união por tamanho, com custo amortizado O(α(N)). A vizinhança vem de uma **contagem exata de contatos** (pares de pixels 4-adjacentes) por par de raízes, guardada num mapa de hash, com listas de vizinhos preguiçosas. Fusões custam O(d) e trocas O(1) por vizinho. Um conjunto de vizinhos sem contagem ficaria desatualizado depois de trocas de pixel e permitiria fundir regiões que já não se tocam (§1.10). |
| 13 | `closePixel` / `insertCornerNodes` (montar os sub-polígonos de cada pixel) | É frágil: muitos casos especiais (vértices dentro do pixel, cantos, várias arestas no mesmo pixel). | Acumulação exata de área assinada, pelo mesmo princípio dos rasterizadores de fonte de área exata. O resultado é **idêntico** ao recorte poligonal, com diferença de 1.3·10⁻¹⁴ [V], sem reconstruir polígonos. As "migalhas de pão" continuam: são os intervalos de parâmetro usados no gradiente (§2.2–2.3). |
| 14 | `dxflib` | A `dxflib` é **GPLv2 ou licença comercial** da RibbonSoft. Linkada num produto fechado, contamina a licença. | O `DxfWriter` fica atrás de `IShapeWriter`. Para uso industrial: comprar a licença comercial ou trocar por um escritor DXF próprio (ASCII R12/R2000 é simples). A ANN é LGPL 2.1 (linkagem dinâmica OK). O Qt 6 é LGPLv3. |
| 15 | Reprodutibilidade | Uso do PRNG sem semente registrada. | A semente do `myPrng` entra em **todas** as impressões digitais. Reduções paralelas têm particionamento fixo e soma ordenada: mesma entrada, mesma saída, bit a bit. |

### 0.1 Inconsistências a resolver na especificação

- **Perfis:** fala em 10, mas enumera 9 (3 de logo com AA, 3 de logo sem AA, 3 de foto). Provisoriamente, `ImageProfile::Custom` é o 10º.
- **Ajustes de usuário:** fala em 11, mas nomeia 5 (sensibilidade, complexidade, suavização, curvas, antialiasing).
- **`dxf_mode`:** os modos 1 e 2 "discretizam", mas a diferença entre eles não foi definida.
- **Fotos e `measurement_types`:** "mola simples apenas na 1ª fase". Nas fases 2 e 3 não há termo de fidelidade? Por enquanto, `MeasurementType::None` nessas fases.
- **Ordem do pré-processamento:** a classificação escolhe o perfil, mas o perfil poderia alterar o σ do desfoque, o que é circular. Decisão provisória: reamostragem e achatamento independem do perfil; o desfoque é aplicado depois da classificação.

### 0.2 Origem da especificação e propriedade intelectual

A especificação coincide em muitos pontos com o Vector Magic e com a tese que o originou: J. Diebel, *Bayesian Image Vectorization: the probabilistic inversion of vector image rasterization*, Stanford, 2008. Os pontos em comum são a análise por síntese com AA analítico, o traço de 0.09375 pt, o modo "bolo em camadas" e perfis como `EP_LOGO_HIG_AA`. Isso tem duas consequências práticas:

1. **Academicamente**, a tese deve ser citada como fonte do método de análise por síntese.
2. **Comercialmente**, implementar algoritmos a partir de uma descrição funcional é diferente de reaproveitar código ou constantes extraídas de um binário. Se essa especificação veio de engenharia reversa, os limiares das árvores de decisão (`0.867741`, `0.0134977`, `57.5`…) e os perfis calibrados devem ser **re-treinados no nosso próprio conjunto de dados**, o que também é o correto do ponto de vista científico (reprodutível). Antes de comercializar, vale uma análise jurídica de patentes e de termos de licença.

---

## 1. Arquitetura

### 1.1 Camadas

```
┌──────────────────────────── aplicação / GUI (Qt 6.8 LTS) ─────────────────────────────┐
│ EngineController : QObject — CoreEngine numa QThread; callbacks → sinais (QueuedConn.) │
│ QImageLoader : IImageLoader — decodificação via plugins QImage (sem ImageMagick)       │
└──────────────────────────────────────────┬──────────────────────────────────────────────┘
                                           │  API C++20 pura (nenhum header Qt no núcleo)
┌──────────────────────────────────────────▼──────────────────────────────────────────────┐
│ cnx::engine  CoreEngine (fachada) · estágios · Fingerprint · TaskContext · Parameters   │
├───────────┬───────────┬───────────┬──────────────┬──────────────────┬───────────────────┤
│preprocess │ classify  │ palette   │ segment      │ vector           │ fit · export      │
│Preprocess.│ImageFeat. │ColorHist. │PixelMap      │BoundaryTracer    │BezierFitter       │
│SignalImg  │ParamClass.│PaletteFnd │RegionForest  │VectorImage (DCEL)│Exporter           │
│EdgeImage  │ImageClass.│KMeans     │QuadTree      │ContourSmoother   │Svg/Eps/Ai/Pdf/Dxf │
│FlowField  │           │WeightsFnd │Pixel/Super/  │Optimizer (CG)    │                   │
│AADetector │           │           │SubPixelSeg.  │GenerativeModel   │                   │
│           │           │           │Trainer       │CrossingGuard     │                   │
├───────────┴───────────┴───────────┴──────────────┴──────────────────┴───────────────────┤
│ sla  Vec2 · Rgba · Array/Array2D · BezierCurve · linalg (inline + BLAS/LAPACK) · myPrng │
│      debugTimers (debugPrintAllTimers) · memoryUsed                                     │
└──────────────────────────────────────────────────────────────────────────────────────────┘
 terceiros: OpenBLAS (BLAS/LAPACK, ABI Fortran) · ANN 1.1.x · dxflib · Qt 6.8 (só o adaptador)
```

### 1.2 Diretórios

```
cnx-praia/
├─ CMakeLists.txt     (cnx_core + testes CTest)
├─ docs/ARQUITETURA.md
├─ include/
│  ├─ sla/            Vec.h  Array.h  BezierCurve.h  [linalg.h  Prng.h  Timers.h  Memory.h]
│  └─ cnx/
│     ├─ engine/      CoreEngine.h  Parameters.h (inclui ParameterFiler)  Fingerprint.h  TaskContext.h
│     ├─ preprocess/  Preprocessor.h  SignalImage.h  EdgeImage.h  AADetector.h  FlowField.h
│     ├─ classify/    Classifier.h (medidas + árvore)
│     ├─ palette/     ColorHistogram.h  KMeans.h  PaletteFinder.h  WeightsFinder.h
│     ├─ segment/     PixelMap.h  QuadTree.h  PixelSegmenter.h  SubPixelSegmenter.h
│     │               [SuperPixelSegmenter.h  CoOccurrenceMatrix.h  BipartiteMatcher.h
│     │                SegmentationTrainer.h]
│     ├─ vector/      VectorImage.h  BoundaryTracer.h  GenerativeModel.h  ContourSmoother.h
│     │               Optimizer.h  EnergyTerms.h  CrossingGuard.h   (a grade de segmentos mora em CrossingGuard)
│     ├─ fit/         BezierFitter.h
│     ├─ io/          ImageDecode.h  Inflate.h  PngDecoder.h  JpegDecoder.h  ImageFileLoader.h
│     └─ export/      Exporter.h  SvgWriter.h  ExportOutputStage.h  [EpsWriter.h  AiWriter.h
│                      CompatiblePdfWriter.h  DxfWriter.h]
├─ src/               (espelha include/; hoje: segment/PixelMap.cpp, QuadTree.cpp, PixelSegmenter.cpp,
│                      vector/VectorImage.cpp, BoundaryTracer.cpp, GenerativeModel.cpp, EnergyTerms.cpp,
│                      Optimizer.cpp, CrossingGuard.cpp, ContourSmoother.cpp; fit/BezierFitter.cpp;
│                      sla/BezierCurve.cpp; palette/*.cpp; segment/SubPixelSegmenter.cpp;
│                      engine/CoreEngine.cpp, Parameters.cpp, ParameterFiler.cpp; preprocess/Preprocessor.cpp,
│                      SignalImage.cpp, EdgeImage.cpp, AADetector.cpp, FlowField.cpp;
│                      classify/Classifier.cpp; export/ExporterSettings.cpp, Exporter.cpp, SvgWriter.cpp,
│                      ExportOutputStage.cpp;
│                      io/Inflate.cpp, PngDecoder.cpp, JpegDecoder.cpp, ImageDecode.cpp)
├─ qt/                EngineController.{h,cpp}  QImageLoader.{h,cpp}
├─ tests/
│  ├─ math/           validate_math.py   (validação numérica das fórmulas)
│  ├─ unit/           test_harness.h, test_support.h, header_smoke_test, pixel_map_test,
│  │                  pixel_segmenter_test, vector_image_test, boundary_tracer_test,
│  │                  generative_model_test, energy_terms_test, optimizer_test, crossing_guard_test,
│  │                  contour_smoother_test, bezier_curve_test, bezier_fitter_test,
│  │                  sub_pixel_segmenter_test, palette_test, weights_finder_test,
│  │                  parameters_test, preprocess_classify_test, core_engine_test, parameter_filer_test,
│  │                  edge_classifier_test, flow_field_test, signal_image_test, image_decoder_test,
│  │                  svg_export_test, contour_quality_test, svg_support.h (leitor e rasterizador de SVG),
│  │                  logo_scenes.h (cenas de logo com AA e o gravador de PNG)
│  ├─ data/           make_fixtures.py, make_gdiplus_fixtures.ps1, manifest*.txt e os arquivos de referência
│  └─ golden/         imagens + SVG de referência (regressão)
├─ bench/             (Google Benchmark: k-means, segmentação, CG, ajuste Bézier)
└─ third_party/       ann/  dxflib/
```

Entre colchetes: cabeçalhos a escrever nas próximas etapas.

### 1.3 Pipeline, impressões digitais e recálculo

| Estágio (`cnx::Stage`) | Equivale a (spec) | Produz (imutável) | Progresso | Paralelismo interno |
|---|---|---|---|---|
| `LoadImage` | `LoadImageThread` | `SourceImage` | 0 % | — |
| `Classify` | `ClassifyImageThread` | `PreprocessedImage`, `ImageClassification` | 0 % | faixas de linhas |
| `Palette` | `FindPalettesThread` | `Palette` | 0–5 % | um K por tarefa |
| `Segment` | `SegmentationThread` | `PixelMap` | 5–20 % | faixas de linhas (estatísticas) |
| `SmoothContours` | `ContourSmoothingThread` | `VectorImage` (poligonal) | 20–80 % | redução da energia e do gradiente |
| `FitBezier` | `BezierFittingThread` | `VectorImage` (com curvas) | 80–90 % | uma aresta por tarefa |
| `Output` | `Save/VectorRender/ThumbsThread` | arquivos, prévia, miniaturas | 90–100 % | três consumidores em paralelo |

**Impressão digital encadeada:**

```
fp_k = H( parâmetros_k , fp_{k−1} , semente )        H = FNV-1a 64 bits sobre a representação canônica
estágio k é "sujo"  ⇔  fp_k ≠ fp_k(cache)
```

Como `fp_k` inclui `fp_{k−1}`, sujar um estágio suja automaticamente toda a cadeia abaixo dele. Alterar só `FitParams` faz `needsToBezierFit()` virar verdadeiro e `needsToSegment()` continuar falso, e a execução recomeça em `FitBezier`. Uma edição manual da segmentação vira uma "pseudo-parametrização": o hash das pinceladas substitui `fp_Segment`.

**Snapshots imutáveis (*copy-on-write*):** cada estágio publica `shared_ptr<const T>`. Uma execução copia os ponteiros do snapshot anterior, substitui os artefatos dos estágios sujos e publica o novo snapshot atomicamente (sob mutex) **só se terminar**. Um cancelamento descarta o snapshot parcial, e a GUI continua mostrando o último completo.

### 1.4 Fluxo de dados em memória

```
 SourceImage                 Array2D<Rgba8>
     │  Preprocessor: achatamento α → reamostragem p/ área-alvo (s = √(A_alvo / (W·H)))
     ▼
 SignalImage (Array2D<Rgba>) ──► EdgeImage (nitidez), FlowField (orientação), perfil AA (6 px)
     │                                    │
     │                           ImageFeatures (20 medidas) ──► árvore de decisão ──► perfil EP_*
     ▼
 PaletteFinder: histograma 16⁴ ponderado por w = 1/(1+edge) → coreset → k-means(K=2..12) → Palette
     │                                 WeightsFinder: pixel de borda = Σ w_i c_i (NNLS no simplex)
     ▼
 ┌────────────────────────────── PixelMap ─────────────────────────────────┐
 │ labels_ : Array2D<RegionId>   id do superpixel inicial (QuadTree 16×16) │
 │              │ regionAt(x,y) = forest.find(labels_(x,y))   [sem reescrever pixels]
 │              ▼                                                          │
 │ RegionForest (Union-Find)                                               │
 │   parent_[r]   compressão por halving: parent[r] ← parent[parent[r]]    │
 │   size_[r]     união por tamanho  →  O(α(N)) amortizado                 │
 │   stats_[raiz] MedianAccumulator (amostras ≤ 1024 │ histograma 4×256)   │
 │   contatos     (raiz_a, raiz_b) → nº exato de pares 4-adjacentes        │
 └───────────────┬─────────────────────────────────────────────────────────┘
                 │ compact()  (= buildRegions): labels ← find(labels); ids densos 0..F−1
                 ▼
 BoundaryTracer  grade de quinas (W+1)×(H+1); fora da imagem = rótulo EXTERIOR
     quina com ≥ 3 rótulos distintos em seus 4 pixels ............ Junction
     quina 2×2 em xadrez (2 rótulos na diagonal) .................. Junction de grau 4 (sela)
     cadeia ortogonal entre junções, colineares removidos ......... Edge (left, right)
     região sem junção (ilha) ..................................... Edge fechada + LoopAnchor
                 ▼
 ┌─────────────────── VectorImage — mapa planar por meia-aresta (DCEL) ───────────────┐
 │ nós (SoA)  pos_[n]  origin_[n]  flags_[n]  quarantine_[n]   ← variáveis do CG       │
 │            nó interior: pertence a 1 aresta;  junção: compartilhada por referência  │
 │ Edge e     edgeNodes_[nodeBegin .. +nodeCount)  (junção … nós … junção)             │
 │            left, right (faces)    curveBegin/curveCount → curves_ (sentido direto)  │
 │ HalfEdge   h = 2e   (direto, pertence a left)  |  h = 2e+1 (inverso, pertence a right)
 │            twin(h) = h ^ 1     edgeOf(h) = h >> 1     next_[h] → próximo no laço    │
 │ Face f     color, outerLoop (área > 0), holeLoops (área < 0), parent                │
 └───────────────┬────────────────────────────────────────────────────────────────────┘
   ContourSmoother / GenerativeModel: movem pos_[] — cada nó compartilhado existe uma vez,
                                      portanto as duas faces vizinhas se movem juntas.
   BezierFitter: ajusta curves_[e] uma vez por aresta (quebras forçadas em junções e cantos).
                 ▼
 Exporter  para cada face em drawOrder(), percorre os laços:
           h direto  → curves_[e] em ordem,          cada curva (P0,P1,P2,P3)
           h inverso → curves_[e] em ordem reversa,  cada curva invertida (P3,P2,P1,P0)
           → IShapeWriter: SvgWriter │ EpsWriter │ AiWriter │ CompatiblePdfWriter │ DxfWriter
```

**Por que a meia-aresta é a estrutura certa:** a garantia de "zero frestas" (§8) não pode ser uma propriedade só da exportação. Se cada região tivesse o próprio contorno, a suavização e o ajuste moveriam as duas cópias de uma fronteira de forma independente, e as frestas surgiriam **antes** do `Exporter`. Com nós e curvas armazenados uma única vez por aresta, a propriedade vale do `BoundaryTracer` até o arquivo final.

**Invariantes**, verificados por `VectorImage::validate()` nos testes e no modo debug:

- **I1:** todo nó interior pertence a exatamente uma aresta; as junções só são referenciadas.
- **I2:** percorrendo todas as meias-arestas de uma face *f*, o laço externo tem área > 0 e os furos têm área < 0. Por isso os caminhos compostos (`cake_stack_mode = 1`) funcionam com a regra **nonzero**, o padrão de SVG, PDF e PostScript, sem declarar `fill-rule`.
- **I3:** fórmula de Euler V − E + F = 1 + C, com C = número de componentes conexas do grafo de arestas. Cada ilha soma +1 em V, E, F e C ao mesmo tempo.
- **I4:** Σ_f área(f) = W·H (exterior excluído). A violação dessa igualdade depois de uma iteração do CG indica um cruzamento que escapou do fiscal.

### 1.5 Teste topológico da troca de pixel

Seja N8*(p) a vizinhança de 8 de p, sem o próprio p, R a região de p e R̄ o seu complemento (pixels fora da imagem contam em R̄). Com R em 4-conectividade e R̄ em 8-conectividade, definem-se:

```
T4(p, R)  = nº de componentes 4-conexas de R ∩ N8*(p) que são 4-adjacentes a p
T8(p, R̄)  = nº de componentes 8-conexas de R̄ ∩ N8*(p)
p é simples  ⇔  T4(p, R) = 1  ∧  T8(p, R̄) = 1
```

Remover um ponto simples preserva a topologia de R: não o divide e não cria nem destrói furos. O teste é O(1): uma tabela de 256 entradas, indexada pela máscara de R ∩ N8*(p) e gerada em tempo de compilação (116 das 256 configurações são pontos simples).

Em 2D, a condição local é exata. Ela equivale a exigir que a remoção de p preserve o número de 4-componentes de R e o de 8-componentes de R̄. Isso foi verificado nas 256 configurações e em 80 mapas aleatórios de 7×7 (`pixel_map_test`). Portanto, a proteção recusa exatamente as trocas que mudariam a topologia de R.

Um detalhe observado na verificação exaustiva: calcular T4 com 8-adjacência gera a mesma tabela, porque a condição T8 = 1 já exclui os casos em que as duas adjacências divergem.

### 1.6 Concorrência

- **Orquestrador:** um único `std::jthread` executa os estágios sujos em ordem. `TaskContext::checkpoint()` verifica o `stop_token` e lança `CancelTaskException`, que é capturada pelo orquestrador.
- **Pool de tarefas:** cada estágio divide o trabalho em blocos com particionamento **fixo** e redução em ordem fixa, o que dá determinismo bit a bit.
- **Snapshots:** são lidos sem bloqueio e trocados sob um mutex curto.
- **Adaptador Qt:** os callbacks chegam na thread do worker, e o `EngineController` os reemite como sinais com `Qt::QueuedConnection`.

### 1.7 Complexidade por estágio

N = número de pixels, B = número de células ocupadas do histograma, P = número de pedaços de segmento por pixel (pixels de borda cruzados), n = número de nós do contorno, I = número de iterações.

| Estágio | Tempo | Memória extra |
|---|---|---|
| Pré-processamento (gaussiano separável de raio r) | O(N·r) | O(N) |
| Paleta: histograma + k-means no coreset | O(N + Σ_K K·R·B) | O(B) |
| WeightsFinder (NNLS, \|S\| ≤ 5) | O(N_borda · K²) | O(K²) |
| Segmentação: quadtree + rodadas de juntar/trocar | O(N + rodadas · N·α(N)) | O(N) rótulos + O(regiões) |
| Traçado de bordas | O(N) | O(nós) |
| Suavização, mola | O(I · n) | O(n) |
| Suavização, modelo gerador | O(I · P log P), independente de W·H (§2.2.3) | O(N) somas prefixadas, só na inicialização |
| Ajuste de Bézier | O(rodadas · n · iterações de Newton) | O(n) |
| Exportação | O(curvas) | O(1) por *stream* |

### 1.8 `BoundaryTracer` e `VectorImage`: decisões de implementação

**Faces são componentes 4-conexas.** Antes de traçar, o `BoundaryTracer` rotula as componentes 4-conexas de cada rótulo. Isso garante um único laço externo por face, mesmo quando a entrada não tem conectividade (no modo "paleta exata", o rótulo é o índice da cor). `TraceResult::faceRegion` devolve o rótulo de origem de cada face, e `pixelFaces` a face de cada pixel.

**Vértices.** Uma quina vira vértice quando tem grau ≥ 3 (inclusive selas) ou quando é canto da moldura (`Pinned | Corner`).
- Grau 3 implica 3 faces distintas.
- Grau 2 implica exatamente 2 faces, de modo que as faces dos dois lados não mudam ao longo de uma cadeia. O traçador confere isso a cada passo e lança `std::logic_error` se falhar.
- Nos trechos retos, as quinas colineares não viram nós.

**Laços sem junção.** Na varredura, a primeira quina de um laço é a mais alta à esquerda, sempre uma dobra com fendas Leste e Sul; ela vira o `LoopAnchor`. Um laço nunca toca a moldura, porque toda quina da moldura com fenda interna tem grau 3.

**Ligação dos laços.** Em cada vértice, as meias-arestas que saem são ordenadas pelo ângulo da direção inicial, e o laço segue a regra next(twin(o_k)) = o_{k−1}, cíclica. A regra não depende da grade: vale também depois da suavização. Os testes cobrem direções diagonais.

**Selas e beliscões.**
- Numa sela 2×2 em xadrez, as 4 fendas se encontram num único vértice de grau 4.
- Se uma face toca a si mesma nessa sela (pelos dois pixels da mesma diagonal), o contorno externo e o contorno do "furo" viram **um único passeio fechado** que passa duas vezes pelo vértice: um passeio beliscado. A área desse passeio é a área da face, e a regra nonzero o preenche corretamente. A face fica registrada com 0 furos.
- Regiões que só se tocam por uma quina geram arestas fechadas presas a uma junção (sem `LoopAnchor`) e furos em forma de 8.

**Pais (ordem de desenho).** O pai de uma face X é decidido assim:
- Se o laço externo de X encosta num furo de G, X está diretamente dentro de G, e o pai é G.
- Se encosta no laço externo de G, X e G são irmãs (agrupadas por Union-Find) e herdam o mesmo pai.
- As faces junto à moldura encostam no furo do exterior, portanto têm pai 0.
- Uma face dentro de um beliscão fica irmã da face beliscada. Isso basta para o modo "bolo", porque o passeio beliscado já exclui a área dela.

**Áreas.** `loopArea` usa, aresta por aresta, a fórmula fechada de Green das cúbicas (§2) quando a aresta tem curvas, e o shoelace sobre os nós quando não tem. As somas são feitas relativas ao centro da imagem, o que melhora o condicionamento. `validate()` recalcula as áreas em vez de confiar em `Face::area`; `updateFaceAreas()` as atualiza depois de mover nós ou trocar curvas.

**Desempenho medido** (MSVC 19.44, x64). Imagem de 1024×768 com 40 102 faces, 45 602 arestas e 152 154 nós:

| Build | Traçado | `validate()` |
|---|---|---|
| Release | 96 ms | 16 ms |
| Debug | 1,24 s | 0,60 s |

**Estratégia de teste.** Cada traçado de teste passa por `validate()`, que confere:
- I1–I4;
- que next é uma permutação;
- que cada laço está registrado em exatamente uma face;
- que os pais não formam ciclos.

Além disso, cada traçado passa por três verificações independentes da implementação:
1. a área de cada face é igual ao número de pixels da face (comparação exata, porque as coordenadas são inteiras);
2. o número de voltas de cada face em torno de cada centro de pixel é 1 para a face do pixel, 0 para as demais e −1 para o exterior;
3. simulando o modo "bolo" (laços externos pintados por profundidade), a última face que cobre cada centro de pixel é a face do pixel.

Cobertura:
- 14 testes de casos construídos: ilha, anéis concêntricos, xadrez, beliscão, laço preso a junção, diagonal, entre outros.
- 240 imagens aleatórias (ruído e discos sobrepostos, de 1×1 a 19×19).
- 1 imagem de 1024×768.
- 10 testes de `VectorImage` com mapas montados à mão, incluindo a detecção de violações de I1 e I4.

**Teste de mutação.** Injetei quatro defeitos, um por vez, numa cópia do código, e a suíte detectou todos:
- next escolhendo o sucessor angular em vez do antecessor;
- lados trocados ao percorrer a fenda Leste;
- pais ignorando os furos;
- junção de grau 3 tratada como se exigisse grau 4.

### 1.9 Build e testes

Num *Developer Prompt* do Visual Studio (x64), ou com GCC/Clang no PATH:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Validado com MSVC 19.44 em Debug e Release, com /W4 e sem avisos. GCC/MinGW (o alvo da especificação) ainda não foi testado nesta máquina.

### 1.10 `PixelMap`: decisões de implementação

**Contatos em vez de listas de vizinhos.** Para cada par de raízes, a `RegionForest` guarda quantos pares de pixels 4-adjacentes separam as duas regiões. Esse número é o comprimento da fronteira compartilhada, em lados de pixel. Fica num `unordered_map` cuja chave (raiz_a, raiz_b) é sempre formada por raízes atuais. As contagens são mantidas assim:
- `setLabel` soma 1 por 4-vizinho já atribuído de outra região.
- `tryMovePixel` atualiza, para cada um dos 4 vizinhos q, o par (origem, Q) com −1 e o par (destino, Q) com +1. O custo é O(1).
- `unite` re-indexa as entradas da raiz absorvida, com custo O(d) no grau dela, e descarta o contato entre as duas regiões, que vira fronteira interna.
- `neighbors(r)` canoniza a lista preguiçosa da região em O(d log d) e devolve só as raízes com contato > 0.

Um conjunto de vizinhos sem contagem não serviria: depois de uma troca, uma região pode deixar de tocar outra, e o segmentador fundiria regiões que não são mais vizinhas.

**Troca (`tryMovePixel`).**
- O destino precisa ser 4-vizinho do pixel; caso contrário, a função lança `std::invalid_argument`.
- Um pixel que não é ponto simples da região de origem é recusado, com retorno `false` e nenhum efeito. Em particular, uma região nunca desaparece por trocas: um pixel isolado não é simples.
- A caixa envolvente só cresce entre trocas (é um superconjunto) e é recalculada exatamente em `compact()`.

**Mediana.**
- `MedianAccumulator` guarda as amostras brutas até 1024 pixels e passa a um histograma de 4×256 posições acima disso, em ambos os casos com 4 KiB.
- A fusão aproveita o histograma já existente (da região absorvida, se for o caso).
- A mediana devolvida é a inferior, o valor de posição ⌊(n−1)/2⌋.
- `assignMedianColors(k)` usa a média quando a região tem menos de k pixels (por padrão, k = 3).

**Compactação (`compact()`, o `buildRegions` da especificação).**
- Os ids são renumerados na ordem em que aparecem na varredura, de forma determinística.
- Regiões sem pixels são descartadas.
- Os rótulos são reescritos, a floresta vira identidade, e caixas e contatos são recalculados do zero.
- `isCompact()` só é verdadeiro enquanto nenhuma fusão ou troca acontece depois disso.

**Ligação com o traçador.** `BoundaryTracer::trace(const PixelMap&)` exige `isCompact()`. A cor de cada face vem de `RegionStats::color` e o índice de paleta de `RegionStats::paletteIndex`; `faceRegion` usa os ids compactos. Uma região que não seja 4-conexa vira várias faces com o mesmo `faceRegion`.

**Testes** (`pixel_map_test`, 14 testes, e 3 testes de integração em `boundary_tracer_test`):
- Mediana, média, remoção e as 16 combinações de fusão entre modos, comparadas com força bruta.
- Ponto simples contra a definição global de topologia, de forma exaustiva e aleatória.
- Seis cenas com fusões e trocas aleatórias, conferindo a cada rodada contagens, medianas, contatos e vizinhos contra força bruta. Elas também confirmam que nenhuma região fica desconexa e que `tryMovePixel` concorda com `isSimplePoint` em todas as tentativas.
- Na integração, o traçado de um `PixelMap` compactado produz exatamente uma face por região, com a cor e a área da região.

**Teste de mutação** (6 defeitos injetados um por vez):
- Detectados: troca sem proteção topológica, `unite` perdendo contatos, troca esquecendo o contato entre origem e destino, e mediana superior em vez de inferior.
- Caixas antigas mantidas em `compact()`: escapou na primeira rodada. Acrescentei um teste determinístico em que uma região perde seu pixel mais à esquerda, e o defeito passou a ser detectado.
- T4 calculado com 8-adjacência: mutante equivalente, isto é, gera exatamente a mesma tabela. Nenhum teste poderia detectá-lo.

### 1.11 `PixelSegmenter` e `QuadTree`: decisões de implementação

**QuadTree.** Parte da imagem em blocos de `quadTreeBlock` × `quadTreeBlock` (16 por padrão); os da borda são truncados. Um bloco vira folha se tem 1 pixel ou se a variância de cor é ≤ `quadTreeVariance`. Caso contrário, é dividido ao meio em cada eixo e examinado de novo. A variância é a soma, nos 4 canais, da variância por canal (canais em [0, 1]). É calculada com somas inteiras exatas, como (n·Σv² − (Σv)²)/n², sem cancelamento numérico. Não uso imagens integrais: elas custariam 64 bytes por pixel, enquanto reler cada bloco em cada nível custa O(N · log₂ 16) = 5N operações simples, com memória O(folhas). As folhas particionam a imagem e saem em ordem determinística.

**Escala multiplicativa.** Os limiares são t₀, t₀·g, t₀·g², … até o teto, e o teto é sempre o último. Em cada limiar há uma varredura de juntar e uma de trocar. No teto o par se repete até estabilizar, no máximo `finalSweepCap` vezes.

**Juntar.** Para cada raiz, em ordem de id, ficam candidatas as vizinhas com dist²(cor, cor) ≤ t, ordenadas da mais parecida para a menos. A primeira que passa na simulação "e se" é fundida, e a raiz resultante é examinada de novo. A dist² usa os 4 canais RGBA em [0, 1].

A simulação "e se" mede o erro quadrático em torno da cor mediana (SSE) e aceita a fusão se

> SSE(A ∪ B; mediana da união) ≤ SSE(A; cor A) + SSE(B; cor B) + `whatIfFactor` · t · min(nₐ, n_b)

- Para duas regiões chapadas de cores a e b, com nₐ ≥ n_b, o lado esquerdo vale n_b·|a−b|². A condição vira |a−b|² ≤ t, a mesma regra de distância das medianas. Já regiões heterogêneas são tratadas com mais rigor.
- O SSE é aditivo na união (para o mesmo centro). A soma de `squaredDeviation` das duas partes é interrompida (aborto antecipado) assim que passa do orçamento, e o resultado "ultrapassou" é devolvido na hora.
- A mediana da união (`MedianAccumulator::unionMedian`) é calculada sem alterar os acumuladores, em O(256 + nₐ + n_b).

**Trocar.** Para cada pixel, em varredura por linhas, o candidato é o 4-vizinho de região diferente cuja cor está mais próxima da cor do pixel. A troca só acontece se a distância ao candidato for estritamente menor que a distância à própria região, e só se `tryMovePixel` aceitar, isto é, se o pixel for um ponto simples. **As cores das regiões ficam congeladas durante a varredura**: o cache é atualizado só no fim. Com isso cada troca reduz estritamente Σ dist²(pixel, cor da sua região), e a varredura termina. Recalcular a mediana a cada troca custaria até 2 mil operações por pixel de fronteira.

**Cache de cores.** Fica por raiz, com marcação de "suja" nas regiões alteradas por `unite` ou por troca; só estas são recalculadas. A cor é a mediana, ou a média abaixo de 3 pixels, como em `assignMedianColors`. `MedianAccumulator::median()` ganhou um caminho rápido: até 32 amostras são ordenadas em vez de varrer 4×256 contadores. A definição não muda (posição ⌊(n−1)/2⌋). Os superpixels 1×1 e 2×2 dominam a contagem de regiões em imagens com ruído.

**Passada diagonal (interpretação da especificação).** A especificação diz que "quinas onde pixels se tocam apenas na diagonal são resolvidas conectando a diagonal de cor mais parecida". Implementei assim:
- Uma sela é uma janela 2×2 em que uma diagonal pertence à mesma região e as outras duas células pertencem a regiões diferentes dela. A região toca a si mesma só pela quina; no `VectorImage` isso é um vértice de grau 4.
- Num mapa válido as duas diagonais nunca são ambas da mesma região. O caminho 4-conexo de uma delas separaria as células da outra. Por isso a condição acima não tem um caso simétrico duplo.
- Qualquer movimento de uma célula da janela para a região de um 4-vizinho dentro da janela desfaz a sela. Há até 6 candidatos. O custo é dist²(pixel, cor da região de destino) − dist²(pixel, cor da região de origem), e vale o menor custo que seja ponto simples.
- A passada repete até não restar movimento possível, no máximo `finalSweepCap` vezes. No ponto fixo, toda sela que sobra tem as 4 células como pontos não simples. Isso é verificado em teste.

**Regiões minúsculas (`mergeSmallRegions`).** Regiões com menos de `minRegionPixels` pixels são fundidas à vizinha de cor mais próxima. Em empate, vence o maior contato e depois o menor id. A região única da imagem não é fundida. O laço se repete até estabilizar e termina, porque cada fusão reduz o número de raízes. Como fundir pode criar selas e mover pixels pode criar regiões pequenas, o pipeline alterna `mergeSmallRegions` e `diagonalSweep` até nenhum dos dois alterar o mapa.

**Paleta.** `snapColorsToPalette` atribui a cada região a cor da paleta mais próxima da sua média (uma rodada de k-means, só o passo de atribuição). Empate → menor índice.

**Invariantes mantidos em todos os estágios.** Toda região é 4-conexa (fusões só entre vizinhas, trocas só em pontos simples), tem ≥ 1 pixel, e os contatos são exatos. O resultado é determinístico: não há aleatoriedade, e a ordem de varredura é fixa. `finish()` consome o mapa interno do segmentador (move-o para o resultado).

**Parâmetros novos** (`SegmentParams`): `whatIfFactor` (padrão 1) e `finalSweepCap` (padrão 8). Os demais valores continuam marcados como "a calibrar".

**Testes** (`pixel_segmenter_test`, 24 testes; o caso de ruído por pixel usa 512×384 em Release e 160×120 em Debug):
- Mediana pequena contra força bruta (cruzando o limite de 32 amostras); união e SSE contra força bruta em 64 combinações de tamanhos e modos, incluindo o aborto antecipado.
- QuadTree: as folhas particionam a imagem exatamente, em 6 formatos × 4 limiares; a imagem uniforme dá uma folha por bloco; a variância tem valor conhecido (0,25).
- Juntar (limiar, fator "e se", regiões não vizinhas); trocar (pixels mal rotulados são corrigidos, ponte recusada, regiões sempre conexas em 6 cenas aleatórias); diagonal (a sela com o movimento mais barato; ponto fixo em 8 cenas de percolação com 132 selas iniciais e 0 finais); regiões minúsculas (cor mais próxima e desempate por contato).
- Pipeline: logo limpo recuperado exatamente; logo com ruído ±3 recuperado, com o resultado passando por `BoundaryTracer` e `validate()` (I1–I4); 30 pixels intrusos isolados são absorvidos; 5 imagens aleatórias com todas as regiões conexas e traçado válido; determinismo; paleta; cancelamento.

**Desempenho** (MSVC Release, 1 thread): logo 640×480 com ruído, 4,6 mil superpixels → 4 regiões, ≈ 0,4 s. Ruído por pixel 512×384, o pior caso para a QuadTree (196 mil superpixels), ≈ 2,4 s. Em Debug os tempos são cerca de 12× maiores.

**Teste de mutação do PixelSegmenter** (13 defeitos injetados um por vez, em Release):
- Detectados: trocar escolhendo a região mais distante; juntar sem a simulação "e se"; diagonal escolhendo o movimento mais caro; regiões minúsculas com desempate invertido e com a região mais distante; QuadTree com `<` no lugar de `≤` e com a metade direita encolhida; `unionMedian` com mediana superior; sela aceitando uma célula igual à diagonal; mediana pequena com posição superior.
- Dois escaparam na primeira rodada: juntar e trocar sem marcar a cor em cache como suja. Acrescentei `colorCacheConsistent()` (autoverificação do cache) e o teste `color_cache_stays_consistent_through_every_stage`, que a chama depois de cada estágio. Os dois defeitos passaram a ser detectados.
- `squaredDeviation` sem o aborto antecipado: mutante equivalente quanto ao resultado. O contrato é "devolve algo > limite", e o total exato também o cumpre; o aborto só muda o tempo de execução. Nenhum teste funcional poderia detectá-lo.

### 1.12 `GenerativeModel`: decisões de implementação

**Escopo.** O modelo trabalha sobre as **poligonais entre os nós** do `VectorImage`, que é a representação otimizada pelo `ContourSmoother`. As curvas de Bézier ajustadas depois (`BezierFitter`) não entram. As cores das faces são fixas; as variáveis são as posições dos nós. A cor de cada face é `Face::color`, e o exterior (face 0) vale 0. A mistura é linear nos 4 canais RGBA, no espaço em que o alvo e as cores foram dados. O alvo é a imagem original, sem desfoque (§0).

**Cobertura exata por área (§2.2.2).**
- Um único passe pelas arestas, com o vetor dc = c_esq − c_dir por aresta. A face à esquerda soma o trecho no sentido direto e a da direita soma o trecho invertido, que dá o sinal oposto.
- Cada segmento é fatiado nas linhas inteiras da grade (DDA com duas progressões aritméticas intercaladas, sem ordenação) em "migalhas de pão". Cada migalha, com δy = (t_b − t_a)·Δy e x̄ no ponto médio, gera dois incrementos de acumulador: α = −δy·((i+1) − x̄) na coluna i e γ − α, com γ = −δy, na coluna i+1.
- Os incrementos são ordenados por (linha, coluna) e fundidos. A soma prefixada da linha devolve ĉ_p.
- **Bordas da imagem.** Partes com y fora de [0, H] ou x > W não afetam nenhum pixel e são descartadas. A parte com x < 0 cobre, na sua linha, tudo o que está à direita: soma γ na coluna 0. Isso vale para nós que a otimização leve para fora da moldura; é testado.
- **Segmentos sobre linhas inteiras.** A atribuição da migalha ao pixel (por floor do ponto médio) é irrelevante para a área: um segmento vertical em x = m dá o mesmo resultado nos dois pixels vizinhos. Só o gradiente é descontínuo nesse conjunto de medida nula.
- **Face exterior.** `faceCoverage(0)` devolve zeros. Os laços do exterior são a moldura invertida, e a soma de Green daria −1 em cada pixel.

**Energia esparsa (§2.2.3).** Entre duas colunas com incremento a cor é constante. Os pixels cruzados são avaliados um a um; cada trecho entre eles custa O(1), a partir das somas prefixadas por linha de w·I (4 canais) e de w·‖I‖², calculadas uma vez na construção do modelo. Custo por avaliação: O(P log P) na ordenação dos incrementos, sem depender de W·H. Memória do modelo: ≈ 48 bytes por pixel. Linhas sem incrementos custam O(1). O erro de uma sequência é n‖c‖² − 2c·S¹ + S², cortado em 0 para absorver o arredondamento; nos testes a diferença para a energia densa fica abaixo de 1e-9 em termos relativos.

**Gradiente analítico (§2.3).** Para cada migalha (pixel p, intervalo [t_a, t_b] do segmento k, com normal n_k = (Δy, −Δx)):
- escala s = 2·w_p·dc·(ĉ_p − I_p), com o ĉ_p do grupo (linha, coluna), achado por busca binária;
- nó inicial += s·n_k·∫(1−t)dt, nó final += s·n_k·∫t dt, com ∫(1−t) = (t_b − t_a) − ½(t_b² − t_a²) e ∫t = ½(t_b² − t_a²).

Os nós de junção e os de arestas fechadas acumulam todas as arestas incidentes sem tratamento especial, porque o índice é o `NodeId`. No modo com gradiente, as migalhas de segmentos horizontais mantêm um incremento de valor zero: a cobertura não muda, mas o pixel precisa existir como grupo para o ĉ_p ser consultado, e a normal do segmento não é nula. O gradiente tem a forma `std::span<Vec2>` com um elemento por nó; mascarar nós fixados e em quarentena é responsabilidade do chamador.

**Descontinuidade conhecida.** Um nó movido sobre a moldura direita ou inferior (x = W ou y = H) tem derivada lateral: a migalha cai fora de qualquer pixel e é descartada, o que corresponde à derivada pelo lado de fora. O CG com região de confiança de 0,5 px (§2.3) não depende desse ponto.

**API.** `energy`, `energyAndGradient`, `render` e `faceCoverage` (densos, para diagnóstico e testes) e `denseEnergy` (definição, O(W·H), usada como referência). Todos os métodos são const e sem estado mutável. `GenerativeStats` devolve migalhas, incrementos e pixels cruzados da última avaliação.

**Testes** (`generative_model_test`, 14 testes):
- Cobertura contra recorte poligonal exato (Sutherland–Hodgman por pixel, soma assinada de laços e furos): erro < 1e-12 em 5 cenas com nós deslocados, cobertura em [0, 1] e soma 1 em cada pixel.
- Escadinha sem deslocamento reproduz a cor de cada pixel; a imagem sintetizada é a mistura ponderada pelas coberturas.
- Energia esparsa = densa (< 1e-9 relativo), com nós deslocados até 3 px, fora da imagem, e com pesos incluindo zeros.
- Alvo = síntese: E ≈ 0 e gradiente ≈ 0.
- Gradiente contra diferenças centradas (h = 1e-5): 568 derivadas em 6 cenas, mais cenas com pesos e nós distantes, e uma ilha com aresta fechada; desvio relativo < 1e-5. Há também um caso de forma fechada de um único segmento.
- Segmentos fora da imagem: um retângulo cuja aresta esquerda é exatamente vertical em x = −2, e um que a atravessa, cobrem os pixels à direita.
- Complexidade: numa imagem de 1000×800 com moldura e ilha, a avaliação esparsa usa ~2,5 mil migalhas e leva ≈ 1,4 ms, contra ≈ 15 ms da densa.

**Teste de mutação** (15 defeitos injetados um por vez, em Release): pesos (1−t)/t trocados no gradiente; dc invertido; sem a parte à esquerda da imagem, ou com seu sinal trocado; α com a borda esquerda do pixel; normal com sinal trocado; varredura sem o trecho final da linha; incrementos repetidos sem somar; peso ignorado na energia e no gradiente; gradiente sem o fator 2; nó final com o peso do inicial; eventos nulos descartados no modo com gradiente; ĉ do pixel cruzado sem somar o incremento. Todos foram detectados. Um escapou na primeira rodada: ignorar um segmento exatamente vertical à esquerda da imagem, caso que o deslocamento aleatório dos nós nunca produz. O teste `edges_left_of_the_image_still_cover_the_pixels_to_their_right` passou a cobri-lo.

### 1.13 `ContourSmoother`: termos, CG e fiscal

**Energia por fase** (`EnergyTerms.h`; os pesos de cada fase vêm de `phaseWeights`):

E = λ_fid · (mola | E_gen | nada) + λ_ang · E_ang + λ_sp^(φ) · E_sp + λ_air · E_air + λ_brd · E_brd

| Termo | Definição | Observações |
|---|---|---|
| Fidelidade | mola Σ max(0, ‖P−O‖ − r)², ou E_gen do `GenerativeModel`, ou nenhuma (`measurement_types[φ]`) | λ_fid = `fidelityWeight` (novo, padrão 1, a calibrar) |
| Angular | Σ θ², θ = atan2(u×v, u·v), nos nós **interiores** das arestas e nos de arestas fechadas | nas junções a cadeia termina, então não há ângulo; a flag `Corner` desliga o termo no nó |
| Espaçamento | Σ s², s = (‖Δ_k‖ − ‖Δ_{k−1}‖) / (‖Δ_k‖ + ‖Δ_{k−1}‖) ∈ [−1, 1] | pesos 5,0 → 0,5 → 0,5; **relativo** (a forma absoluta não era invariante à escala, §1.25) |
| Pressão do ar | (A − A₀)² por contorno com < 8 nós | A₀ = área na escadinha original; contorno = laço de uma face (externo ou furo); o exterior não tem |
| Âncora de margem | (coordenada − moldura)² nos nós da moldura não fixos | ver a trava exata, abaixo |

Todos os gradientes são analíticos e conferidos contra diferenças finitas centradas (`energy_terms_test`), inclusive com cantos perfurados e arestas fechadas. O gradiente do prior angular segue a tabela do §2.4. Três observações sobre a energia vieram da execução:

1. **Trava exata da moldura.** Só a mola de margem não basta: com ela os nós da moldura derivam alguns milésimos para dentro, a moldura deixa de medir W×H e a invariante I4 (Σ áreas = W·H) falha. Por isso, além da mola, o gradiente perpendicular dos nós da moldura é zerado e a coordenada perpendicular é fixada exatamente no início da fase. Eles deslizam ao longo da moldura e nunca saem dela (`borderHardConstraint`, ligado por padrão). A mola de margem continua na energia e vale zero nesse regime.
2. **O espaçamento é necessário, não só estético.** Sem ele, nós livres sobre um trecho reto deslizam e se amontoam. O prior angular depende de 1/‖Δ‖², fica rígido demais perto da coincidência, e a busca em linha falha. Num teste, dois nós chegaram a 1e-6 px um do outro. O fiscal trata esse colapso como um cruzamento (ver abaixo).
3. **O prior angular não desloca polígonos com poucos nós.** Em qualquer polígono convexo Σ θ = 2π, então um retângulo (4 giros iguais) já é o mínimo de Σ θ². A perfuração de cantos só faz diferença quando há curvas ou lados em ângulos diferentes; é por isso que o teste usa um triângulo com hipotenusa em escada.

**Otimizador** (`Optimizer.h`): gradiente conjugado não linear **Polak–Ribière+** (β = max(0, g'·(g'−g)/‖g‖²)) com reinício de Powell (|g'·g| ≥ 0,2‖g'‖²) e reinício em d = −g se d não for de descida. A busca em linha é a de Nocedal & Wright (alg. 3.5 e 3.6) com **Wolfe forte** (c₁ = 1e-4, c₂ = 0,1), interpolação cúbica protegida e bisseção de reserva. A **região de confiança** limita o deslocamento de cada nó a `maxNodeStepPx` (0,5) por iteração; no teto basta a condição de Armijo. Funções indefinidas fora de um domínio (NaN) são tratadas como "passo grande demais". O passo inicial é 1/‖g‖ na primeira iteração e, depois, α_prev·(g·d)_anterior/(g·d) (N&W 3.60).

**Fiscal anti-cruzamento** (`CrossingGuard.h`): interseção própria entre segmentos sem nó em comum, com uma **grade uniforme de segmentos** no lugar da kd-tree da ANN (que indexa pontos, não segmentos). O resultado é idêntico ao da força bruta O(S²), verificado em cenas perturbadas com dezenas de cruzamentos. Cada par é testado uma só vez, na célula do canto inferior-esquerdo da interseção das caixas. O fiscal também detecta **segmentos colapsados** (< `minSegmentPx`, 0,05 px).

**Fluxo de uma fase.** Constrói a topologia estática, perfura os cantos (só na fase `punctureOnPhase`), cria o objetivo sobre os nós livres e itera. Depois de cada iteração aceita pelo CG, o fiscal confere o resultado. Havendo cruzamento ou colapso: desfaz a iteração, põe os nós envolvidos em quarentena por `quarantineRounds` iterações e recomeça o CG do ponto anterior (a energia continua não crescente). Quando a quarentena expira, o conjunto de nós livres muda e o CG recomeça de novo. Ao fim da fase a quarentena é limpa. Entradas que já se cruzam ou já têm segmentos colapsados são recusadas (`std::invalid_argument`).

**Canto perfurado (`isProbablyPuncturedNode`).** O vetor de 21 medidas da especificação não está definido, então implementei uma regra de duas medidas que usa exatamente os dois conceitos citados ("ângulo local e retidão dos trechos adjacentes"). Cada braço segue a poligonal por até `punctureArmPx` (4 px) de comprimento de arco. O nó é canto se |giro entre as cordas dos braços| ≥ `punctureCornerThresh` (0,8 rad) e **nenhum nó intermediário** de cada braço se afasta da corda em mais de `punctureFlatnessThresh` (0,3 px). Limiar ≤ 0 desativa. A regra deve ser recalibrada no conjunto de referência. **Atenção:** a regra é aplicada, em `ContourSmoother::puncture`, a uma poligonal filtrada ao longo do arco e em duas escalas de braço, e na fase 0, sobre a escadinha (§1.25); a descrição acima vale para a regra em si.

**Resultados medidos** (MSVC Release):
- Disco de raio 10,6 com antialiasing (supersampling 16×16), fidelidade generativa: o erro radial médio dos nós cai de 0,425 px na escadinha para 0,015 px, e E_gen de 14,7 para 0,004 (≈ 90 ms).
- Reta inclinada: o desvio máximo dos nós interiores cai de 0,657 px para 0,001 px.
- Ângulo reto de um triângulo (generativo): o nó fica a 0,085 px do vértice verdadeiro com a perfuração e a 0,861 px sem ela.
- Pressão do ar: numa região em L (6 nós), a área fica em 180,00 com a pressão e vai a 182,75 sem ela.
- Duas barras escuras com alvo de barra única: o fiscal desfez 1 iteração por fase e o resultado não tem cruzamentos.

**Limitações conhecidas.** Todos os pesos são provisórios (a calibrar): com a mola padrão (λ_fid = 1) um triângulo encolhe vários pixels sob o prior angular de 10, e os testes que dependem disso fixam os pesos explicitamente. O CG pode terminar por `LineSearchFailed` perto do arredondamento (gradiente ≈ 1e-3), o que é reportado em `PhaseStats::status`. O prior de suavidade "raiz quadrada" não foi implementado (`priorType` ≠ 3 lança `std::invalid_argument`).

**Testes** (`energy_terms_test` 12, `optimizer_test` 9, `crossing_guard_test` 7, `contour_smoother_test` 11; a qualidade contra imagem está em `contour_quality_test`, §1.25):
- Termos em forma fechada, topologia contra contagem independente, gradientes contra diferenças finitas (cada termo e o total, com as três medições).
- Detector de cantos em poligonais: L, reta, extremo de cadeia, arco, zigue-zague e quadrado fechado.
- Otimizador: quadrática mal condicionada, Rosenbrock (32 iterações), Wolfe forte e Armijo em cada passo aceito, β ≥ 0, região de confiança por nó, domínio com NaN, falha da busca em linha, determinismo.
- Fiscal: casos construídos e grade = força bruta em 12 cenas perturbadas.
- Suavizador: precisão subpixel, topologia (`validate()`), nós fixos imóveis, moldura exata, quarentena que não vaza, determinismo, cancelamento e validação de argumentos.

**Teste de mutação** (24 defeitos injetados um por vez, em Release):
- **Detectados (21).** Sinais e fatores dos gradientes (angular, espaçamento, pressão do ar, âncora); zona livre da mola ignorada; flag `Corner` ignorada; vizinho errado na aresta fechada; limite de nós da pressão do ar; PR sem reinício de Powell; Wolfe fraco; região de confiança ignorada; interseção testada de um lado só; grade sem a célula dona do par; segmento curto invertido; iteração com cruzamento não desfeita; colapso não checado; trava da moldura removida; perfuração na fase errada; espaçamento sempre da fase 0.
- **Dois escaparam na primeira rodada e agora são detectados:** o canto que ignora o braço da frente (acrescentei o zigue-zague visto do outro lado) e os nós do cruzamento fora da quarentena (sem ela o CG repete o mesmo passo inválido até esgotar as iterações; o teste das barras exige menos de 10 rejeições).
- **Três escapam e não os considero falhas dos testes:**
  - *PR sem o corte em zero (O1):* equivalente em efeito. Um β negativo exige g'·g > ‖g'‖², o que já dispara o reinício de Powell (|g'·g| ≥ 0,2‖g'‖²) antes de β ser usado. O corte fica por ser a definição padrão do PR+.
  - *Armijo sem o termo c₁ (O4):* só difere quando a queda de energia fica entre 0 e c₁·α·|g·d|. Nenhum teste produz esse caso, e os passos aceitos nos testes cumprem Armijo de qualquer modo. É uma lacuna real de cobertura, não verificada.
  - *Quarentena que nunca expira (S6):* só atrasa a convergência dentro da fase (os nós ficam congelados até o fim dela, quando a quarentena é limpa). Nenhum teste mede esse atraso.

### 1.14 `BezierFitter`: decisões de implementação

**Visão geral** (`include/cnx/fit/BezierFitter.h`). Cada aresta do `VectorImage` suavizado é ajustada uma vez e guarda suas curvas em `curves_`; as duas faces vizinhas as compartilham. `BezierFitter::run` substitui as curvas existentes; `fitChain` ajusta uma poligonal solta (usado pelos testes).

**Quebras e fragmentos.**
- Os nós com a flag `Corner`, `Junction` ou `Pinned` são quebras de tangente **livre**. Entre elas, cada trecho é cortado em fragmentos iniciais de `initialFragmentNodes` nós (no mínimo 4: fragmentos menores que isso virariam retas, e uma reta ao lado de uma cúbica quebra a continuidade G¹).
- Todos os fragmentos iniciais têm ≥ 4 pontos, exceto um trecho que já nasça com ≤ 3. Fragmentos de ≤ 3 pontos são retas (`SegmentKind::Line`), com E calculada na projeção de cada nó interior sobre a corda.
- **Aresta fechada.** O nó âncora (`LoopAnchor`) é uma quebra lisa: o primeiro e o último fragmento compartilham a tangente do âncora. Se o âncora tiver flag de canto ou junção, a ponta é livre.

**Modelo de um fragmento** (`FragmentModel`). A curva é B(t; p) = B_fixo(t) + J(t)·p, linear em p. Cada ponta é livre (P₁ ou P₂ inteiro entra em p, 2 coordenadas) ou G¹ (uma incógnita α ou β, com P₁ = P₀ + α·τ₀ e P₂ = P₃ − β·τ₃, e a parte constante vai para B_fixo). Isso dá k = 4, 3, 3 ou 2 incógnitas e uma única implementação das derivadas:

| Grandeza | Fórmula |
|---|---|
| E | ½ Σ_k ‖B(t_k) − Q_k‖² nos nós interiores |
| g_p | Σ Jᵀ r |
| g_t,k | r·B' |
| H_pp | Σ JᵀJ |
| h_k | Jᵀ B' + J'ᵀ r |
| D_k | ‖B'‖² + r·B'' |

O gradiente e a Hessiana em seta foram conferidos contra diferenças finitas centradas nos quatro modos de ponta (erro < 1e-6; a especificação numérica do §2.6 dá 1e-10 e 1e-8).

**Newton com complemento de Schur.** Partindo de t por comprimento de corda e p por mínimos quadrados (equações normais k×k por Cholesky, com uma crista mínima se forem quase singulares):
- S = H_pp + λI − Σ h h'ᵀ/(D_k + λ), S δp = −g_p + Σ h g_t/(D_k + λ), δt_k = −(g_t,k + h_kᵀ δp)/(D_k + λ);
- D_k + λ é mantido ≥ 1e-9 (Hessiana indefinida);
- λ ← λ/10 quando o passo é aceito (E diminui) e λ ← 10λ quando é rejeitado, até 14 tentativas;
- t é projetado em [0, 1] e feito crescente. Custo O(m) por iteração, contra O(m³) do Newton denso.

O passo de Schur coincide com a solução do sistema denso (k+m)×(k+m) (testado, erro < 1e-9). Um λ pequeno pode deixar S indefinido (o Cholesky falha e λ aumenta), o que o ajuste trata aumentando λ. Num arco ruidoso de 26 pontos, a energia cai de 0,288 (mínimos quadrados com t por corda) para 0,088 (Newton/Schur, 8 iterações).

**G¹ degenerado.** Se, no fim, α ≤ 0 ou β ≤ 0 (laço ou cúspide), o fragmento usa a heurística de Schneider (α = β = ‖P₃ − P₀‖/3), os t são reotimizados para essa curva (Newton 1D por nó) e o fragmento fica marcado `degenerate`. A fusão recusa uniões degeneradas, então a subdivisão permanece.

**Fusão e deslocamento** (por trecho, da esquerda para a direita):
- Funde A e B se ΔE = E*(A∪B) − E*(A) − E*(B) < τ_r, com τ_r = τ₀ (τ_f/τ₀)^{r/(R−1)}, r = 0..R−1 (R = 20: fator ≈ 2,42 por rodada). O ajuste da união parte dos t de A e B reescalados pelos comprimentos de corda (*warm start*).
- Se recusa, testa deslocar a quebra em ±1 nó e fica com a de menor E*(A) + E*(B). O deslocamento nunca deixa um fragmento com menos de min(4, tamanho original) pontos.
- Depois das R rodadas, `finalSweeps` varreduras extras no limiar final (até nada mudar). A especificação para em R rodadas; as extras deixam o resultado num ponto fixo.
- Como a fusão é gulosa, a energia final não é estritamente monótona no limiar. O teste do limiar usa folga de 20 % e exige que o número de curvas nunca aumente.

**O que ΔE < τ significa.** E soma ½d² sobre os nós do fragmento, então τ_f = 0,2 deixa o erro total de uma curva chegar a 0,2 px²: num disco de raio 10,6 com ~50 nós, o resultado são 2 curvas com desvio radial de 0,16 px; com τ_f = 0,003, 4 curvas com 0,019 px. Perfis que queiram precisão de pixel devem usar limiares menores que os 0,2 do perfil AA. O valor é ajustável em `FitParams::statThreshFinal`.

**`BezierCurve`.** Implementei os métodos que estavam só declarados: `secondDifferenceBound`, `flatteningSegments` (n = ⌈√(3M/(4·tol))⌉), `flatten`, `bounds` (raízes das quadráticas de B'_x e B'_y) e `length` (Gauss–Legendre de 5 pontos, adaptativo).

**Testes** (`bezier_curve_test` 6, `bezier_fitter_test` 23):
- Curva: bases e derivadas, divisão e inversão, caixa justa contra amostragem, comprimento contra polilinhas finas e o quarto de círculo, discretização dentro da tolerância.
- Fitter: derivadas contra diferenças finitas, passo de Schur contra o denso, mínimos quadrados estacionário, recuperação de uma cúbica com t desconhecidos, Newton contra mínimos quadrados, G¹, cúspide, retas, *warm start*, cronograma de limiares, colineares → 1 curva, círculo (3 curvas, desvio 0,014 px, G¹ inclusive na âncora), quadrado com cantos (4 curvas), limiar contra número de curvas, deslocamentos, integração com o `VectorImage` (cadeias contínuas, cantos viram pontas, `validate()` e áreas de Green), disco suavizado, determinismo e cancelamento.

**Teste de mutação** (22 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (17).** Limite de segmentos da discretização, raízes em y da caixa, peso de Gauss–Legendre, limite da curvatura; sinais do gradiente, de D, de h, do lado direito de Schur e de δt; coluna G¹ do fim e parte fixa G¹ do início; deslocamento que escolhe o pior; cantos que não quebram a cadeia; expoente do cronograma de limiares; sem G¹ nas quebras internas; Newton que nunca aceita o passo; λ sem nova tentativa; *warm start* sem reescala.
- **Escaparam e agora são detectados (3).** Fusão por erro absoluto em vez de ΔE (o teste do ziguezague de 15 pontos, cuja união tem E = 0,17 > τ_f = 0,1 e mesmo assim funde); tangente errada na âncora (agora comparada com a diferença central); Schneider sem a marca `degenerate` (o caso do ziguezague com tangentes a favor do percurso, que é determinístico, tem de ser marcado).
- **Escapam (2).** Não consegui construir um caso que os alcance, então são lacunas de cobertura reais:
  - *Sem a monotonia de t (F7):* nenhum dado dos testes, nem uma polilinha que volta sobre si mesma, leva o Newton a ordenar os t de forma errada.
  - *Fusão que aceita fragmento degenerado (F14):* a degeneração só ocorre com G¹ nas duas pontas, e nem a estrela serrilhada fechada produz uma união degenerada que o código original recuse.

### 1.15 `SubPixelSegmenter`: decisões de implementação

**Onde entra.** Roda depois do `PixelSegmenter`, sobre o mesmo `PixelMap` e a mesma imagem (as cores dos pixels alimentam as medianas). É o modo "com mistura nas bordas" da §4: pixels de antialiasing não podem virar regiões finas de cor intermediária. A saída já vem compactada, com cor e índice de paleta em cada região, e alimenta o `BoundaryTracer`.

**Estágios**, todos públicos para teste:
1. **Micro-regiões** (`mergeMicroRegions`). Para tamanho s = 1..`microRegionMaxPixels`, funde toda região com ≤ s pixels à vizinha de cor mais próxima (empate → maior contato → menor id), repetindo até estabilizar. A ordem crescente importa: no teste, uma região de 1 px prefere a de 2 px (distância 60) à grande (110), e a de 2 px prefere a grande (50). Em ordem crescente a primeira vai para a de 2 px, que passa a ter 3 pixels e sobrevive; uma varredura única por id absorveria a de 2 px e depois a de 1 px, e a região sumiria.
2. **Praias** (`splitAndMergeBeaches`). Uma região é praia quando:
   - tem pelo menos duas vizinhas e as duas de maior contato cobrem ≥ `beachNeighborShare` (0,8) do contorno;
   - o contraste entre elas é ≥ `beachMinContrast` (‖cor_B − cor_A‖² ≥ 0,02);
   - a cor da região projeta-se em t ∈ (0,02; 0,98) sobre a reta A→B e fica a ≤ `beachLineTolerance` · ‖cor_B − cor_A‖ dela (uma faixa de terceira cor não é praia);
   - é fina: nenhum pixel é interior (todo pixel toca outra região ou a borda da imagem), ou seja, tem 1 ou 2 px de largura (uma banda larga é uma região de verdade).

   Cada pixel da praia vai para o lado de sua própria cor (t < ½ → A, senão B), por `tryMovePixel` em varreduras repetidas: só se movem pixels que são pontos simples da praia e vizinhos do destino, então as pontas da faixa saem primeiro e as demais viram pontas em seguida. O último pixel de uma faixa de largura 1 não é ponto simples (removê-lo apagaria a região); ele é fundido inteiro (`unite`) ao lado da sua cor. O resultado tem a fronteira no meio da transição e preserva a 4-conexidade.
3. **Sombras** (`flagShadowSegments` e `rebuildShadowSegments`). Marca com `RegionFlag::Shadow` a região com ≥ `shadowMinPixels` pixels **interiores**, variância de cor ≥ `shadowVariance` (soma dos 4 canais) e variância da imagem desfocada ≥ `shadowSmoothRatio` · variância bruta, tudo medido só nos pixels interiores (os 4 vizinhos na região). O desfoque de caixa preserva um gradiente e elimina o ruído, então a razão separa as duas coisas: um gradiente tem razão ≈ 1 e um ruído de ±12 níveis, ≈ 0,04. A reconstrução classifica cada pixel da região pela cor da imagem **desfocada dentro da região**, entre os dois extremos da sombra (tintas das regiões vizinhas) e as cores de paleta sobre a reta entre eles; extrai as componentes 4-conexas de cada cor (a maior mantém o id; as outras viram regiões novas, via `reassignPixel`) e deixa a fronteira nas faixas do gradiente. Pela imagem bruta, o ruído fragmentaria a fronteira em ilhas (testado). Os três defeitos que esta regra tinha (pixels de borda na variância, desfoque que mistura as vizinhas e cor de paleta mais próxima em RGB, que cria uma terceira tinta) estão no §1.26.
4. **Paleta.** `snapColorsToPalette` dá a cada região a cor da paleta mais próxima da sua média; `mergeSamePaletteNeighbors` funde vizinhas 4-adjacentes com o mesmo índice (`clusterColors`); `enforceMinNumPixels` funde as regiões com menos de `minRegionPixels` à vizinha de cor mais próxima, e a cor e o índice da vizinha prevalecem (a união pode escolher a raiz da região pequena); `compact()` renumera (`buildRegions`).

**`PixelMap::reassignPixel`** (novo). Move um pixel para qualquer região, sem teste topológico, atualizando contagens, medianas, caixas e contatos com o mesmo código de `tryMovePixel` (extraído para `moveUnchecked`). Serve às reconstruções; quem o usa responde pela 4-conexidade. `SubPixelSegmenter::splitDisconnected` restaura essa garantia: divide cada região nas suas componentes (a maior mantém o id). No `run` ele é só uma verificação: os estágios acima preservam a conexidade, e o contador `splitRegions` fica em 0.

**Parâmetros novos** (`SegmentParams`, todos a calibrar): `microRegionMaxPixels` 2, `beachLineTolerance` 0,25, `beachMinContrast` 0,02, `beachNeighborShare` 0,8, `shadowVariance` 2e-3, `shadowSmoothRatio` 0,5, `shadowMinPixels` 16, `shadowBlurRadius` 2.

**Resultado medido.** Disco com antialiasing (raio 11,4, supersampling 16×16, imagem de 40×40 quantizada em 8 bits): o `PixelSegmenter` sozinho deixa 4 regiões (as duas cores e mais duas de transição, que o estágio de sombras marca). O `SubPixelSegmenter` reduz a 2 (fundo e disco), com cores exatas da paleta e 0 divergências contra "cobertura ≥ ½" (410 pixels escuros; a verdade analítica é 408,3). O mapa resultante passa em `validate()` (I1–I4) com uma face por região.

**Testes** (`sub_pixel_segmenter_test`, 22; os 7 de sombra e franja estão no §1.26): `reassignPixel` contra força bruta (tamanhos, medianas, contatos em 4 cenas com 300 reatribuições cada) e seus argumentos; micro-regiões em ordem crescente; praias de 1 e 2 pixels; quatro casos que não são praia (terceira cor, banda larga, contraste baixo, contorno repartido em três); sombra marcada só no gradiente (não no chapado nem no ruído); reconstrução em faixas, com e sem ruído; paleta, mínimo de pixels e a fronteira de exatamente 4 px; `splitDisconnected`; o disco com antialiasing; determinismo, cancelamento e validação de argumentos.

**Teste de mutação** (17 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (15).** Micro-regiões do maior para o menor; vizinha mais distante; pixel da praia no lado oposto; praia sem o teste de espessura, da reta A–B, da fração de contorno ou do contraste; sombra sem o teste de suavidade ou de variância; reconstrução pela imagem bruta (o teste com ruído); mínimo de pixels inclusivo (a região de exatamente 4 px); `splitDisconnected` ficando com a última componente; micro-regiões acima do tamanho da vez; paleta sem checar o índice; `reassignPixel` sem o caso "já pertence" e sem o contato do destino.
- **Escaparam e agora são detectados (2).** Regiões sem índice de paleta (−1) que se fundiriam entre si (o pipeline nunca chega a esse estado; um teste de estágio o cobre) e a cor da vizinha que deve prevalecer em `enforceMinNumPixels` quando a união escolhe a raiz da região pequena (empate de tamanho com o menor id; também é um teste de estágio).

### 1.16 `PaletteFinder` e `WeightsFinder`: decisões de implementação

**Pipeline da paleta** (`include/cnx/palette/`):
1. **Pesos de pixel.** w = 1/(1 + borda/`edgeScale`), com a borda dada pelo gradiente de Sobel nos 4 canais (RGBA em [0, 1]) dividido por 4, de modo que um degrau de amplitude a em um canal dá a (um degrau preto|branco em RGB dá √3). É o `EdgeImage` (§1.19), que a paleta usa via `EdgeImage::sharpnessWeights`. Os pixels de transição pesam menos, para o antialiasing não criar cores falsas.
2. **Histograma esparso** (`ColorHistogram`), 16 faixas por canal (8 no modo rápido): só as células ocupadas existem. Cada célula guarda o peso, a **média ponderada** das cores que caíram nela e o espalhamento interno Σ w‖c − média‖². Assim o erro de um agrupamento medido nas células é **exatamente** o erro nos pixels (testado: 1e-9), e o k-means tem custo independente de N (o coreset da §0, item 11).
3. **K = minColors..maxColors.** Sementes (`chooseSeeds`): os picos do histograma, pulando os que ficam a menos de 2 faixas de uma semente já escolhida; as que faltarem saem de k-means++ (amostragem D² ponderada pelo peso, com `myPrng` semeado por seed ⊕ K). Depois, Lloyd ponderado com no máximo `kmeansMaxRounds` (10) rodadas e distância em RGBA. Centros que ficam vazios são descartados.
4. **Nota** = log(erro + 1) + (cores − 2)·`colorPenalty`, com o erro quadrático médio por unidade de peso (RGBA em [0, 1]), e "cores" = centros não vazios. Fica a menor nota (empate → menos cores). A normalização por peso é uma escolha minha: com o erro total, a nota dependeria do tamanho da imagem.
5. **Imagem chapada** (`findOptimalNumColors`, parte das cores exatas): as cores exatas com ≥ `exactColorMinShare` (0,2 %) dos pixels, se forem ≤ `maxColors` e cobrirem ≥ 99 % da imagem, são a paleta, sem k-means (`exact = true`). Pixels de antialiasing, cada cor com poucos pixels, não contam. `forcedNumColors` desliga essa detecção.
6. **Refinamento.** Para o K escolhido, `refineRounds` (2) rodadas de k-means sobre os pixels reais (item 11 da §0). O erro final é medido nos pixels.

**Não implementado:** a segunda árvore de decisão de 62 medidas (2 a 9 cores ou modo "livre") não está definida na especificação. A escolha de K é só a menor nota. Os limiares `edgeScale`, `exactColorMinShare` e `colorPenalty` são provisórios (a calibrar).

**`myPrng`** (`sla/Prng.h`): SplitMix64, conferido contra os três primeiros valores de referência da semente 0 (0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F). `uniform()` com 53 bits e `below(n)` sem viés (rejeição de Lemire).

**`WeightsFinder`** (decomposição de um pixel em mistura das cores da paleta):

> w = argmin ‖Σ wᵢ cᵢ − p‖²  sujeito a  wᵢ ≥ 0, Σ wᵢ = 1

- **Duas cores:** projeção na reta com t ∈ [0, 1] (forma fechada).
- **Três ou mais:** conjuntos ativos, como Lawson–Hanson. Começa no vértice mais próximo; entra a variável de gradiente mais negativo; resolve o sistema KKT [CᵀC 1; 1ᵀ 0][z; ν] = [Cᵀp; 1] no suporte; se algum z sai ≤ 0, anda de w até z e para na primeira restrição, removendo o peso zerado. Uma cor afinamente dependente das do suporte (KKT singular) é banida.
- **Por que não o Cholesky da especificação.** Com 4 canais, CᵀC tem posto ≤ 4 e é singular para n ≥ 5. O KKT continua não singular enquanto as cores do suporte são afinamente independentes, e o ótimo sempre tem suporte de no máximo 5 cores (Carathéodory em ℝ⁴). É o item 12 da §0.
- **Candidatas.** Com mais de `maxCandidates` (9) cores na paleta, só as mais próximas do pixel entram.
- **Unicidade.** Com n ≤ 5 cores afinamente independentes, os pesos de uma mistura exata são únicos; com mais cores há várias misturas exatas (dependência afim), e o que se garante é o resíduo mínimo.

**Testes** (`palette_test` 17, `weights_finder_test` 6):
- `myPrng` contra a referência; histograma contra força bruta (média e espalhamento de 600 cores em 256 faixas); a identidade erro do coreset = erro dos pixels; sementes por pico com supressão e k-means++; centros na média ponderada do grupo; descarte de centros vazios; o refinamento sobre os pixels (rampa).
- Paleta: o gradiente de Sobel de um degrau (√3) e os pesos; 4 cores com ruído → K = 4 com a nota mínima ali e a fórmula conferida em cada K; imagem chapada (inclusive com 3 pixels de uma cor rara); K forçado; a banda de transição (com os pesos de borda K = 2, sem eles K = 3); o disco com antialiasing → 2 cores; determinismo e validação.
- Pesos: duas cores (projeção, saturação, cores iguais); mistura exata com 3 a 9 cores; **o resíduo contra a enumeração exaustiva de todos os suportes de até 5 cores** (125 casos, muitos fora do fecho convexo); suporte ≤ 5; paleta colinear; candidatas e subconjuntos.
- Desempenho: imagem 512×512 com ruído, 4 cores, ≈ 110 ms (Release).

**Teste de mutação** (22 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (19).** Histograma: espalhamento sem a média, chave sem o alfa, média dividida pelo nº de pixels, ordem das células invertida. K-means: sementes sem supressão, erro sem o espalhamento, centros vazios mantidos, centros fora de ordem. Paleta: nota com o nº de cores errado, chapada com cobertura frouxa, peso de borda sem a escala, Sobel com o peso errado, K forçado que continua buscando. Pesos: projeção de duas cores sem o corte, KKT sem a restrição de soma, entrada da variável de maior gradiente, pesos negativos aceitos, candidatas mais distantes, índice repetido. `myPrng` com 52 bits.
- **Escaparam e agora são detectados (2).** Os centros que nunca se movem para a média do grupo, e o refinamento sobre os pixels que não atualiza os centros. O primeiro agora tem um teste com valor analítico (o item de 0,3 e peso 4 move o centro de A para 1,2/14). O segundo, uma rampa vermelha 0..199 em que o k-means das células deixa os centros a 2 níveis do ótimo dos pixels (47,5 e 147,5) e o refinamento os leva a 49,0 e 149,0 (ótimo: 49,5 e 149,5).
- **Escapa (1), por ser inalcançável.** *Cor dependente não banida (W10):* o ramo em que o sistema KKT é singular só roda se uma cor afinamente dependente do suporte tiver gradiente estritamente menor que λ. No ótimo de um suporte, uma cor dependente tem gradiente igual a λ (empate) ou, se o suporte tem 5 cores, o resíduo é nulo e todos os gradientes são zero. Só o arredondamento poderia chegar lá. O ramo fica como proteção contra laço infinito.

### 1.17 `CoreEngine`: orquestrador do pipeline

**Responsabilidade.** Liga os módulos da §1.3 numa cadeia de sete estágios (`LoadImage → Classify → Palette → Segment → SmoothContours → FitBezier → Output`), decide **o que precisa ser refeito** pelas impressões digitais, publica o resultado como um `PipelineSnapshot` imutável e permite cancelar e acompanhar o progresso.

**Execução.** `start(alvo)` cancela e aguarda a execução anterior, copia a configuração sob mutex (`RunConfig`: parâmetros, entrada, pedidos de exportação, observador, snapshot anterior) e roda `runPipeline` numa `std::jthread`. A thread só **lê** a cópia; a GUI pode chamar `setParameters`, `snapshot()` e `firstDirtyStage()` durante a execução. Para cada estágio até o alvo:

1. calcula `fp_k` (a impressão encadeada da §1.3) com os parâmetros **efetivos**;
2. se o estágio não está sujo acima dele (`dirty` ainda falso), o artefato existe no snapshot anterior e `fp_k` coincide, ele é **reaproveitado** (o mesmo `shared_ptr`; o observador recebe `onStageFinished(…, reaproveitado = true)`);
3. senão, é recalculado e `dirty` passa a valer para os de baixo;
4. no fim, estágios **depois do alvo** com `dirty` verdadeiro têm artefato e impressão descartados (estariam obsoletos); sem `dirty` ficam como estavam.

O snapshot só é publicado se a execução termina (`Completed`). `Cancelled` e `Failed` deixam o último snapshot completo intacto. Exceções do observador são engolidas (o contrato as proíbe; um observador defeituoso não derruba o motor).

**Perfil e parâmetros efetivos.** `Classify` escolhe um `ImageProfile`; a partir dele `EngineParameters::applyProfile` fixa o tipo de medição do suavizador e `fit.statThreshFinal`, e os estágios de `Palette` em diante usam esses parâmetros. A impressão do `Classify` usa os parâmetros brutos (o perfil ainda não existe). As tabelas por perfil e as dos controles de usuário (`applyUserControl`, escala multiplicativa `multiplicativeScale`, fator em [¼, 4] com 50 = identidade) são **provisórias**: a especificação dá a faixa e não a tabela (§0.1).

**Estágios.**
- *LoadImage*: `IImageLoader` (o adaptador Qt fica fora do núcleo). Impressão = caminho + tamanho + data do arquivo + `revision` (`invalidateInput()` a incrementa para forçar a releitura).
- *Classify*: `preprocess` (alfa constante → opaco; redução por **média exata de área** quando passa de `targetAreaPx`, ampliação bilinear só com `allowUpscale`; cópia gaussiana `smooth` para paleta e segmentação, e `image` sem desfoque para o `GenerativeModel`) e `classify`.
- *Palette*: `PaletteFinder` (§1.16).
- *Segment*: paleta exata → componentes 4-conexas por cor mais próxima, varreduras de diagonais e fusão de regiões pequenas; senão `PixelSegmenter` e, nos perfis com AA, `SubPixelSegmenter` (§1.15). A faixa de progresso é dividida 70/30 nesse caso.
- *SmoothContours*: `BoundaryTracer` + `ContourSmoother`, com o `GenerativeModel` só se alguma fase usa medição gerativa.
- *FitBezier*: copia o `VectorImage` suavizado (a poligonal fica intacta), roda o `BezierFitter` e recalcula as áreas das faces pela fórmula de Green das cúbicas.
- *Output*: ponto de extensão `IOutputStage`. O `Exporter` e os escritores (SVG/EPS/AI/PDF/DXF) ainda não existem; o estágio só chama o plugin quando há pedidos de exportação. Sua impressão cobre arquivo, formato, `ExporterSettings` (inclusive a `BackgroundSelection` com desfazer/refazer) e a do estágio anterior.

**Edição manual da segmentação.** `replaceSegmentation(mapa, hashDasEdições)` exige um snapshot que já passou pela paleta e um `PixelMap` compacto. A impressão do `Segment` passa a ser `H(hash das edições, fp da paleta)`, uma pseudo-parametrização: vale enquanto a paleta for a mesma, e `setInput` a descarta. Mudar a paleta recalcula a segmentação.

**Desvios da especificação (por escolha minha).**
- **Uma thread orquestradora**, não uma por estágio: os estágios são sequenciais por dependência de dados, e o paralelismo útil está **dentro** deles (§1.6). `workerThreads` fica reservado.
- O cancelamento é cooperativo (`std::stop_token` + `CancelTaskException` nos pontos de verificação dos `TaskContext`), com a mesma latência de checagem dos módulos.
- O **classificador** usa medidas reais e uma árvore escrita à mão no lugar da árvore treinada da especificação (§1.19). A primeira versão, de três regras sobre cores exatas, foi substituída.

**Testes** (`parameters_test`, `preprocess_classify_test`, `core_engine_test`):
- *Parâmetros:* sensibilidade de cada campo de cada struct à impressão (um campo esquecido reaproveitaria cache indevido); canonicalização de −0,0, NaN e separação de strings; tabela dos nove perfis; multiplicatividade e saturação dos controles; `BackgroundSelection` (desfazer/refazer, descarte do ramo, equivalência de impressão).
- *Pré-processamento e classificador:* média de área exata (blocos 2×2, razão não inteira, conservação da massa); gaussiana (normalização, simetria, bordas replicadas); alfa, área-alvo, ampliação; validação e cancelamento; casos básicos do classificador e o perfil forçado (o corpus e as medidas estão em `edge_classifier_test`, §1.19).
- *Motor:* execução completa e progresso monótono dentro da faixa de cada estágio; alterar só o ajuste refaz só do `FitBezier` para baixo (identidade dos `shared_ptr`); cada grupo de parâmetros suja exatamente o seu estágio; execução sem nada sujo; alvo parcial; cancelamento (no carregador, entre estágios, dentro do suavizador, e sem estragar o cache); `start` cancelando a execução anterior; falhas; exceções do observador; disco com AA → `LogoMedAA` e área do disco a 0,2 % do círculo; perfil forçado; edição manual; estágio de saída; leitores concorrentes.
- Resultados: 19 alvos de CTest, Debug e Release, sem avisos do MSVC /W4.

**Teste de mutação** (48 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (32).** Motor: cache que ignora a impressão digital; estágio refeito que não suja os de baixo; ausência da invalidação depois do alvo; edição manual válida com outra paleta; revisão e caminho fora da impressão da entrada; saída sem as configurações; classificação sem os parâmetros do classificador; suavizador sem o modelo gerador; estágio em cache informado como refeito; `start` que não cancela a execução anterior (detectado por *timeout*); perfil ignorado nos estágios seguintes; AA sem `SubPixelSegmenter`; `firstDirtyStage` que ignora diferenças; `needsToBezierFit` fora por um; saída rodando sem pedidos. Pré-processamento: pesos de área sem normalizar, escala sempre 1, ampliação sem permissão, gaussiana sem normalizar, cópia suave que não é desfocada, alfa constante não achatado. Classificador: limiar de logo chapado frouxo, extensão sensível a maiúsculas, perfil forçado marcado como automático, formato com perda ignorado. `BackgroundSelection`: ramo de refazer não descartado, passo registrado ao ocultar o já oculto, visibilidade invertida. Parâmetros: tabela de perfil alterada, sensibilidade invertida, fotos com mola nas três fases, campos (`allowUpscale`, `energyTolerance`, `enforceG1`) fora da impressão, impressões sem encadear o estágio anterior, espessura fora da impressão de exportação.
- **Escaparam e agora são detectados (13).** Saída sem o formato na impressão (teste: mudar só o formato); curvas sem recalcular as áreas (compara com `updateFaceAreas()` de novo); ausência do teste de cancelamento entre estágios (cancelar ao fim do ajuste não pode chegar ao plugin de saída, que não tem pontos de verificação); `setInput` mantendo a edição manual (reaplicar o mesmo caminho a descarta); `needsToSegment` fora por um; gaussiana com borda circular (rampa), raio curto (a 3 px do impulso), ampliação bilinear sem o meio pixel (2 → 4 colunas dá 0, 25, 75, 100); limiares do classificador (70 % chapado → foto) e limite de cores (exatamente `maxColors` ainda é logo, uma a mais é foto); impressão do fundo sem o conjunto (mesmo tamanho, faces diferentes); limpar o que já está vazio; limiar final sem o piso do inicial.
- **Equivalentes (2).** *Segmentação sem o perfil na impressão (C4)* e *paleta sem a semente (C11)*: o perfil e a semente já entram na impressão do `Classify`, que está encadeada em todas as de baixo; remover o termo duplicado não muda nenhuma decisão. Os termos ficam como cinto e suspensório (o perfil também decide o modo de segmentação).
- **Escapa (1).** *Paleta exata que para no primeiro passo (C21):* só difere quando uma varredura de diagonais cria outra quina diagonal, o que exigiria uma imagem construída para isso. O limite `finalSweepCap` protege contra laço infinito.
### 1.18 `ParameterFiler`: perfis `.apf`

**Formato.** Texto plano, uma entrada por linha, `Classe::parametro valor` (a especificação dá o exemplo `ContourSmoother::prior_strengths 10`). Linhas vazias e iniciadas por `#` são ignoradas; o escritor abre com `# cnx .apf v1` e agrupa as chaves por classe, cada grupo sob um comentário com o nome da classe. Esses comentários são só para humanos: um leitor que siga a especificação os ignora.

| Classe | Chaves (`EngineParameters`) |
|---|---|
| `CoreEngine` | `seed` (inteiro de 64 bits sem sinal) |
| `Preprocessor` | `target_area_px blur_sigma_px flatten_alpha allow_upscale` |
| `ParameterClassifier` | `auto_classify forced_profile` |
| `PaletteFinder` | `min_colors max_colors kmeans_max_rounds histogram_bins color_penalty forced_num_colors edge_scale exact_color_min_share refine_rounds` |
| `PixelSegmenter` | `quad_tree_block quad_tree_variance merge_thresh_start merge_thresh_growth merge_thresh_max min_region_pixels what_if_factor final_sweep_cap` |
| `SubPixelSegmenter` | `micro_region_max_pixels beach_line_tolerance beach_min_contrast beach_neighbor_share shadow_variance shadow_smooth_ratio shadow_min_pixels shadow_blur_radius` |
| `ContourSmoother` | `measurement_types spacing_weight` (3 valores, um por fase) `fidelity_weight prior_type prior_strengths air_pressure_weight air_pressure_max_nodes border_anchor_weight border_hard_constraint fidelity_free_radius_px quarantine_rounds min_segment_px puncture_corner_thresh puncture_flatness_thresh puncture_arm_px puncture_on_phase max_cg_iterations max_node_step_px gradient_tolerance energy_tolerance` |
| `BezierFitter` | `stat_thresh_initial stat_thresh_final max_rounds initial_fragment_nodes newton_max_iterations final_sweeps enforce_g1` |

Os nomes `prior_strengths`, `air_pressure_weight`, `measurement_types`, `puncture_corner_thresh`, `puncture_flatness_thresh` e `stat_thresh_final` são os da especificação; os demais seguem o mesmo padrão `snake_case` dos campos de C++. A classe é a do módulo que usa o parâmetro, e não a do struct (o `SegmentParams` se divide entre `PixelSegmenter` e `SubPixelSegmenter`). O que a especificação não nomeia (por exemplo as chaves do `SubPixelSegmenter`) é escolha minha, sem compatibilidade a preservar.

**Valores.** Inteiros em decimal; reais em decimal ou científico, com ponto e sem *locale* (`std::from_chars`/`std::to_chars`); booleanos `true|false|1|0`; enumerações pelo nome: `EP_LOGO_HIG_AA … EP_PHOTO_LOW, EP_CUSTOM` (os nomes da especificação; `Custom` é a décima entrada, a que a especificação não enumera) e `spring|generative|none`; `forced_num_colors` aceita `none`. Listas: valores separados por espaço ou tabulação, exatamente um por fase. Tudo é sensível a maiúsculas.

**Escrita sem perda.** O `to_chars` sem precisão gera a *menor* representação que reconstrói exatamente o mesmo `double`; assim `ler(escrever(p)) == p` bit a bit, inclusive com `-0`, subnormais e `DBL_MAX` (testado com 300 conjuntos aleatórios de mantissa cheia). Escrever duas vezes dá o mesmo texto.

**Semântica de leitura.**
- *Parcial:* o que o arquivo não cita fica como estava em `params`. Um `.apf` pode ser só `BezierFitter::stat_thresh_final 0.9`.
- *Transacional:* a leitura trabalha numa cópia e só a publica se o arquivo inteiro for aceito. Erro em qualquer linha, ou regra violada no fim, deixa `params` intacto.
- *Chave repetida é erro* (nos dois modos): com duas linhas para o mesmo campo, "a última vence" esconderia um engano de edição.
- *Modos:* `Strict` (padrão) rejeita chave desconhecida; `Lenient` as devolve (sem repetição, na ordem de aparição) para o chamador avisar, o que permite abrir um perfil de versão futura. Sintaxe malformada, valor ruim de chave conhecida e repetição são erro nos dois.
- *Robustez de entrada:* BOM UTF-8 no início, `\r\n`, tabulações, última linha sem quebra; caractere de controle ou linha maior que 4096 caracteres são recusados.
- *Erros:* `ParameterFileError` com a linha (1-based) e uma mensagem que cita a chave e o texto ofensor, por exemplo `PaletteFinder::max_colors: inteiro inválido: 'oito'`. Linha 0 é erro de E/S ou incoerência que não vem de nenhuma linha do arquivo.

**Validação única.** `ParameterFiler::validate` reúne as faixas e as relações entre campos (`max_colors ≥ min_colors`, `merge_thresh_max ≥ merge_thresh_start`, `stat_thresh_final ≥ stat_thresh_initial`, `histogram_bins ∈ [2, 256]`, `prior_type = 3`, `puncture_on_phase ∈ [0, 2]`, reais sempre finitos…). Elas espelham o que cada módulo já exige no construtor (PaletteFinder, PixelSegmenter, ContourSmoother, BezierFitter, `preprocess`), de modo que **um `.apf` aceito nunca faz o motor falhar na construção** (o teste constrói os módulos com 300 conjuntos aleatórios). O leitor aplica as regras depois de ler e atribui o erro à linha do **último** campo envolvido que o arquivo definiu; o escritor as aplica antes de escrever e lança `std::invalid_argument` em vez de produzir um arquivo que o leitor recusaria.

**Escrita em arquivo.** Grava `perfil.apf.tmp` e renomeia sobre o destino, de modo que uma falha no meio nunca deixa o arquivo anterior pela metade (e parâmetros inválidos nem tocam no disco).

**Uma tabela só.** Cada campo é uma entrada `{chave, leitor, escritor}` construída por `integerField / doubleField / boolField / optionalIntField / enumField / …`, a partir de um acessor genérico `[](auto& p) -> auto& { return p.palette.maxColors; }` que serve ao struct com e sem `const`. Leitor e escritor não podem divergir, e esquecer um campo é pego por `every_field_has_exactly_one_line…` (confere que cada campo muda exatamente uma linha, e que o número de linhas é o de campos).

**Desvios e limites.**
- A especificação diz apenas "texto plano no formato `Classe::parametro valor`". O cabeçalho de versão, a lista fechada de chaves, as regras de validação e os modos são decisões minhas.
- Os valores dos perfis embutidos (`applyProfile`) e dos controles de usuário (`applyUserControl`) não são lidos de `.apf`: continuam em código até que a especificação dê as tabelas; um `.apf` define `Custom`.
- Não há `include` entre arquivos nem expansão de variáveis: um `.apf` é só dados.

**Testes** (`parameter_filer_test`, 14):
- *Ida e volta:* padrões; 300 conjuntos aleatórios (escrever → ler → escrever idêntico, e construção de todos os módulos); extremos (`DBL_MAX`, subnormal, `-0`, `0.1+0.2`, `UINT64_MAX`, `INT_MAX`); cada campo muda exatamente uma linha.
- *Sintaxe:* comentários, BOM, `\r\n`, tabulações, arquivo parcial, vazio, ordem livre; todas as formas de valor (booleano, `none`, as dez enumerações, listas, científico).
- *Erros:* linhas malformadas e dezenas de valores ruins (inteiro, real, booleano, enumeração, listas curtas ou longas), cada um com a linha esperada, o texto da mensagem e `params` intacto; mais de 30 violações de faixa ou de relação entre campos e os valores-limite aceitos; incoerência vinda de valores anteriores (linha 0); leitura que falha no meio (inclusive `badbit`).
- *Modos:* `Strict` e `Lenient`.
- *E/S:* arquivo ida e volta, substituição, destino inválido, diretório como destino, arquivo ou diretório inexistente, *stream* em falha.

**Teste de mutação** (45 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (45).** Leitor: chave repetida aceita, BOM e CR final não removidos, limite de linha frouxo, `#` não reconhecido, caracteres de controle aceitos, modo estrito aceitando chave desconhecida, chaves ignoradas repetidas, valor ausente, leitura que escreve direto em `params` (perde a transacionalidade), regras não aplicadas, erro atribuído à primeira linha em vez da última, chaves definidas não registradas, tabulação que não separa, indentação não removida, `badbit` ignorado. Valores: real impresso com 6 dígitos, real não finito, inteiro ou real com lixo no fim, `1` como booleano, `none`, listas e escalares com valores a mais, nome do parâmetro ou identificador começando por dígito. Escritor: não validar, cabeçalho por classe, erro de `rename` ignorado. Regras: 16 limites trocados (`>=` por `>`, faixas ampliadas em 1, `prior_type`, `puncture_on_phase`, `stat_thresh_final = initial`, `max_colors = min_colors` etc.), cada um com um teste no valor-limite dos dois lados.
- **Não injetados, por serem equivalentes ou inobserváveis (2).** A segunda checagem de enumerações (`findProblem(work)`) depois das regras na leitura: o leitor só produz enumerações da tabela, então ela nunca dispara (ela vale para o escritor, onde há teste). A troca de `perfil.apf.tmp` por escrever direto no destino: só difere se o processo morrer no meio da escrita, o que um teste de unidade não reproduz de forma portátil.

### 1.19 `EdgeImage`, `AADetector` e o classificador de perfil

**`EdgeImage`** (`include/cnx/preprocess/EdgeImage.h`). Por pixel, Sobel 3×3 em cada um dos 4 canais (bordas replicadas):

> m = √(Σ_c gx_c² + gy_c²) / (4·255),   n = (cos θ, sin θ),   θ = ½·atan2(2·Jxy, Jxx − Jyy),   J = Σ_c ∇_c ∇_cᵀ

- A magnitude é a mesma da paleta (um degrau de amplitude a num canal dá m = a; preto|branco dá √3). Agora ela é calculada em **inteiros** sobre os 8 bits: uma região chapada dá exatamente 0 (em `double` sobrava 1e-17, que virava normal e crista falsas), e o resultado não depende da ordem das somas. A paleta passou a usar `EdgeImage::sharpnessWeights` (pesos 1/(1 + m/edgeScale)), e `PaletteFinder::edgeMagnitude` virou um atalho para ela.
- A normal sai do **tensor de estrutura**, não da soma dos gradientes: em vermelho→verde os gradientes dos canais têm sinais opostos e se cancelariam. n é um eixo (o sinal é arbitrário).
- **Cristas** (`ridges`): m ≥ limiar e máximo local ao longo de n, com m **>** m(p − n) e m **≥** m(p + n) (vizinhos por arredondamento). A assimetria escolhe exatamente um dos dois pixels de um degrau, que têm a mesma magnitude. Num disco duro de raio 20 saem 133 cristas para um perímetro de 126, todas a menos de 1,6 px do contorno e com a normal radial.

**`AADetector`** (§2 da especificação: "perfis de 6 pixels consecutivos atravessando perpendicularmente cada borda forte"). Em cada crista com m ≥ 0,2, lê os pixels nos offsets −3..+2 ao longo de n, **por vizinho mais próximo**: interpolar criaria rampas falsas em bordas duras oblíquas. Com as pontas a = v₀ e b = v₅, cada amostra interna é projetada no eixo a→b (t) e medida a distância a ele (r):

| Perfil | Condição |
|---|---|
| Outro | ‖b − a‖ < 0,15; r > 0,15·‖b − a‖ (uma terceira cor, não a mistura das duas tintas); t fora de [−0,15; 1,15] (ringing); retrocesso > 0,15 (textura); ou perfil cortado pela moldura |
| Degrau | todas as amostras internas num patamar (t ≤ 0,12 ou ≥ 0,88) |
| Rampa | ≥ 1 amostra intermediária; o número delas (1–4) é a largura da transição |

Medido num disco de raio 26: duro, 170 degraus e 0 rampas; com AA por superamostragem 8×8, 53 degraus e 125 rampas (largura média 0,70, porque cruzamentos que caem no limite do pixel são degraus legítimos); desfocado com σ = 1,5, 176 rampas (largura 2,86).

**As 20 medidas** (`ImageMeasures`), nas famílias que a especificação enumera:

| Família | Medidas |
|---|---|
| Cores reais | `significantColors` (cores exatas com ≥ 0,2 %), `flatShare`, `distinctColorRatio` |
| Histograma 16⁴ | `mainPeaks` (células com ≥ 1 % dos pixels, a > 0,06 uma da outra: ruído leve perto de um limite de faixa não multiplica a tinta), `peakShare` (pixels a ≤ 0,06 de uma *tinta*), `betweenShare` (pixels fora das tintas, a ≤ 0,04 do segmento entre dois picos distintos e longe das pontas: a fração "entre picos" da especificação) |
| Bordas | `edgeFraction` (m > 0,05), `strongEdgeFraction` (cristas por pixel), `meanEdgeMagnitude`, `interiorNoise` (RMS de m abaixo do limiar de crista: a "energia de sinal") |
| Antialiasing | `edgeProfiles`, `stepShare`, `rampShare`, `otherShare`, `meanTransitionWidth` (o "perfil de nitidez das bordas") |
| Erro | `quantizationRmse` contra a paleta de tintas, e `edgeErrorShare`, a parte dele nos pixels de borda (o "erro nas bordas × no interior") |
| Segmentos preliminares | `quadTreeLeavesPerBlock` (folhas da QuadTree 16×16 por bloco inicial) |
| Arquivo | `lossyFormat` (.jpg/.jpeg), `hasAlpha` |

*Tinta* = um pico do histograma, ou uma cor exata significativa longe de todos os picos. Assim uma tinta de área pequena (< 1 %) num arquivo sem perdas continua sendo tinta, e o erro dela não parece "espalhado".

**A árvore** (`decideProfile`), escrita à mão. Cada limiar vem do que a medida vale nas famílias do corpus:

1. `mainPeaks > maxColors` → **foto** (a paleta não comportaria as tintas).
2. `peakShare + betweenShare < 0,90` (JPEG: 0,95) → **foto**. Num logo, quase todo pixel é tinta ou mistura de duas tintas (corpus: logos ≥ 0,950; fotos ≤ 0,059; ruído forte ≤ 0,888).
3. `quantizationRmse > 0,05` e `edgeErrorShare / edgeFraction < 1,5` → **foto**. Sobra erro, e ele não se concentra nas bordas: é textura ou gradiente. Num logo com AA o erro está todo nas bordas (lift ≥ 5,1 no corpus); ruído de até ~±10 níveis fica abaixo de 0,05.
4. `betweenShare ≥ 0,01`, ou `rampShare ≥ 0,25` com ≥ 16 perfis válidos → **logo com AA** (o mesmo modo serve a borda desfocada: "com AA ou desfoque", §4 da especificação).
5. Senão → **logo sem AA**.

**Corpus sintético** (`the_classifier_gets_every_image_of_a_synthetic_corpus_right`, 12 imagens aleatórias 80×72 por família): **120 de 120 corretas**.

| Família | Esperado | Cobertura | Rampas | Lift mín. | rmse máx. |
|---|---|---|---|---|---|
| logo chapado (e o mesmo como .jpg) | LogoMed | 1,000 | 0,00 | — | 0,000 |
| logo com ruído ±5 | LogoMed | 0,985–1,000 | 0,00–0,08 | 0,93 | 0,076 |
| logo com AA (6×6) | LogoMedAA | 0,992–1,000 | 0,66–1,00 | 5,87 | 0,067 |
| logo com AA (.jpeg) | LogoMedAA | 0,989–1,000 | 0,65–1,00 | 5,14 | 0,047 |
| logo desfocado (σ 0,8–1,5) | LogoMedAA | 0,950–1,000 | 1,00 | 2,90 | 0,074 |
| foto (e .jpg) | PhotoMed | 0,005–0,059 | — | 1,00 | 0,42 |
| logo com ruído ±40 | PhotoMed | 0,010–0,888 | — | 1,00 | 1,15 |
| gradiente 2D | PhotoMed | 0,012–0,018 | 0,00 | — | 0,66 |

No ruído ±5, as amostras com rmse acima de 0,05 têm lift de 4 a 16 (faixas finas de uma tinta pequena junto às bordas); as de lift baixo têm rmse ≈ 0,022 (o ruído puro). As duas regras se complementam.

**Ligação ao motor.** O estágio `Classify` do CoreEngine reparte a faixa de progresso entre `preprocess` (0–40 %) e `classify` (40–100 %), que agora recebe o `TaskContext` e pode ser cancelado no meio das medidas. Com `autoClassify = false`, nada é medido.

**Desvios e limites.**
- **A árvore não é treinada.** A especificação fala de limiares aprendidos (`0.867741`, `0.0134977`, `57.5` cores) num conjunto de referência que não existe aqui. As medidas são reais; os limiares estão em `ClassificationRules`, que uma árvore treinada pode substituir sem tocar nas medidas, e não são parâmetros do usuário (não entram em impressão digital nem em `.apf`).
- **O nível HIG/MED/LOW não é medido:** a especificação não diz como escolhê-lo. O classificador escolhe a família e devolve o nível MED; os outros nove perfis continuam disponíveis por `forcedProfile`.
- **A extensão não decide sozinha:** um PNG salvo como .jpg sem ruído é logo. O JPEG só endurece o limiar de cobertura. Artefatos de bloco 8×8 (um JPEG salvo como PNG) não são medidos.
- **Gradiente suave e de baixo contraste** (poucas faixas do histograma, cada uma um pico, erro < 0,05) passaria como logo posterizado; o gradiente amplo do corpus vai para foto pela regra 1.
- O `FlowField` (campo de orientação para traços finos) está na §1.20, e a `SignalImage` na §1.22; o classificador não usa nenhum dos dois. A "energia de sinal" da especificação continua sendo `interiorNoise`.

**Testes** (`edge_classifier_test`, 12 casos):
- *EdgeImage:* imagem chapada e vazia; magnitude, normal e uma crista por cruzamento em degraus vertical, horizontal, diagonal e de canais opostos; borda só no alfa; igualdade com os pesos da paleta; cristas de um disco.
- *Perfis:* degrau, rampa de 1, 2 e 4 amostras, limites da tolerância, terceira cor, mistura de duas cores, ringing, retrocesso, contraste.
- *Detector:* disco duro, com AA e desfocado; retângulo de bordas inteiras; diagonal com meio-tom (amostragem arredondada); perfis cortados.
- *Medidas:* valores exatos (cristas, magnitude média, frações, `stepShare`/`otherShare` com perfis cortados, folhas da QuadTree, `hasAlpha`, tinta pequena, imagem sem picos), logo chapado, com AA e foto; agrupamento de picos sob ruído; "entre picos" só para misturas reais (margem das pontas, tubo, separação mínima).
- *Árvore e classificação:* cada regra e cada fronteira, com limiares diádicos para que a fronteira seja exata; o corpus; o perfil forçado e o cancelamento.

**Teste de mutação** (41 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (34).** EdgeImage: Sobel sem o peso central, sinal trocado numa coluna de `gy`, escala da magnitude, alfa ignorado, ângulo inteiro em vez de meio ângulo, `Jxx` e `Jyy` trocados, supressão não estrita dos dois lados, sem supressão de não máximos, limiar de crista ignorado, peso multiplicando pela escala, vizinho por truncamento na supressão. Perfis: contraste mínimo ignorado, terceira cor aceita, perfil não monótono aceito, patamar baixo contado como intermediário, larguras não contadas, largura média só sobre as rampas. Medidas e árvore: picos vizinhos não agrupados, erro de borda em todo pixel, ruído interior incluindo as bordas, `>` por `>=` em "cores demais", JPEG sem limiar próprio, sem a regra do erro espalhado, *lift* no limite rejeitado, AA sem as misturas, AA sem mínimo de perfis, perfil forçado ignorado, cores exatas pequenas que não viram tinta, folhas sem dividir pelos blocos, `otherShare` zerado, `hasAlpha` não copiado, cristas e magnitude média sem dividir pelos pixels, erro nulo contado como erro todo na borda.
- **Sobreviventes fechados com testes (7):**
  - *Ringing em passos pequenos:* `{0; −0,1; −0,2; 1; 1; 1}` e `{0; 0; 1,3; 1,2; 1,1; 1}`. Cada passo cabe na tolerância de retrocesso, então só o limite absoluto rejeita o perfil.
  - *Amostra do perfil truncada em vez de arredondada:* degrau diagonal com uma linha de meio-tom. Na crista do meio-tom há exatamente 1 amostra intermediária; truncar dá 2.
  - *"Entre picos":* três testes de valor exato, um por condição:
    - sem a margem das pontas: cinza escuro com t ≈ 0,08, fora do raio do preto;
    - com o tubo dobrado: cor a 0,058 do segmento, quando o tubo é de 0,04;
    - aceitando picos a menos de `minPeakSeparation`: picos a 0,19 de distância.
    Cada um tem o controle positivo de um cinza médio, que conta como mistura.
  - *`stepShare` sobre as cristas em vez de sobre os perfis válidos:* imagem com um degrau inteiro e uma borda cortada pela moldura. O valor esperado é `stepShare = 1`, `otherShare = 0,5`.
  - *Pico de reserva fixo em vez da célula mais cheia (quando nenhuma célula chega a 1 %):* 1585 cores distintas e 15 pixels pretos. O rmse esperado é exatamente a distância de cada pixel ao preto.

  Reexecutados (`-Only`), os 7 foram detectados.

### 1.20 `FlowField` e a ligação de traços finos

A especificação descreve o `FlowField` como um campo de direções que calcula a orientação das bordas "para preservar traços finos durante a segmentação e suavização". Antes de escrever o campo, medi o que acontece com um traço fino sem ele.

**O problema, medido.** Pense num traço preto de 1 px em diagonal sobre fundo branco. Os pixels dele só se tocam pela quina. Em 4-conectividade, que é o invariante do `PixelMap`, cada pixel é uma região de 1 px:
- `joinSweep` não os une, porque eles não são vizinhos;
- `diagonalSweep` não os vê, porque uma sela exige a mesma região nas duas pontas da diagonal;
- `mergeSmallRegions` os apaga um a um.

Resultado: um traço de 26 px numa imagem 32×32 termina em **1 região**, e o traço desaparece. Pelo motor (logo chapado, caminho da paleta exata), o mesmo traço vira **10 regiões**: os fragmentos preto+branco recebem, pela média, a cor mais próxima da paleta, que é o vermelho. Traços com inclinação qualquer (escadinha de Bresenham) e círculos finos sofrem o mesmo em cada degrau diagonal. O defeito está na segmentação. A suavização só recebe o que sobra dela.

**O campo** ([FlowField.h](../include/cnx/preprocess/FlowField.h)):
- *Integração do tensor.* O tensor de estrutura de Sobel nos 4 canais é integrado por uma gaussiana: J_σ = G_σ * Σ_c ∇I_c ∇I_cᵀ, separável, com raio ⌈3σ⌉, pesos exp(−k²/2σ²) normalizados e bordas replicadas. As unidades são as de m², então tr J₀ = m² do `EdgeImage`.
- *Medidas por pixel:*
  - energia λ₁ + λ₂;
  - coerência (λ₁ − λ₂)/(λ₁ + λ₂) = √((Jxx − Jyy)² + 4Jxy²)/(Jxx + Jyy);
  - tangente: o autovetor de λ₂, isto é, a direção ao longo da borda;
  - alinhamento cos 2α com uma direção d, em forma fechada, sem trigonometria: −[(Jxx − Jyy)(dx² − dy²) + 2Jxy·2dxdy] / ((λ₁ − λ₂)‖d‖²).
- *Por que integrar:*
  - No centro de um traço de 1 px o gradiente de Sobel é **exatamente nulo**, porque os dois flancos se cancelam, e o `EdgeImage` não dá direção ali.
  - O tensor soma ∇∇ᵀ, que não depende do sinal. Com σ = 1, o centro herda a direção do traço: coerência 1,0000, tangente (0,7071; 0,7071).
  - Um xadrez de 1 px tem Sobel exatamente nulo, então não há campo.
  - Um cruzamento tem coerência 0,000, e o canto de um quadrado tem 0,374.
- *Sem duplicação de conta.* O Sobel inteiro saiu do `EdgeImage` para `sobelTensors`, que os dois módulos usam. Os valores são inteiros menores que 2²⁴, portanto exatos em float, e o `EdgeImage` dá os mesmos resultados bit a bit.

**A ligação** (`PixelSegmenter::linkStrokeSweep`). Considere cada par diagonal p, q = p + (±1, 1) de regiões diferentes. As duas **pontes** (os 4-vizinhos comuns aos dois) precisam ser de outras regiões. O par é de traço quando três condições valem:
1. **Cores:** dist²(p, q) ≤ `mergeThreshMax`, e a distância de p e de q a cada ponte é maior que esse limite.
2. **Não é antialiasing:** a cor do traço não é mistura das duas pontes. O teste de reta é o mesmo das praias (`beachMinContrast`, `beachLineTolerance`). Assim, a escadinha cinza de uma borda a 45° entre preto e branco não é traço. O mesmo cinza entre vermelho e azul é: ele é um contorno.
3. **Campo do par:** sobre a **soma dos tensores** de p e de q, a coerência é ≥ `strokeMinCoherence` e o alinhamento com a diagonal é ≥ cos(2·`strokeMaxAngle`). Exigir isso de cada pixel isoladamente falhava nas pontas: o pixel extremo de um traço tem coerência ≈ 0,48–0,50, em cima do limiar. A soma é a integração sobre as duas vizinhanças. Na ponta, o vizinho firma a direção; num canto, os dois pixels são isotrópicos e a soma também.

Para ligar um par, uma ponte passa para a região de p, desde que seja ponto simples (`tryMovePixel`), e p e q são unidos. A ponte escolhida é, nesta ordem de preferência:
1. a que não cria contatos novos (os outros dois 4-vizinhos dela são da sua própria região);
2. a de cor mais próxima do traço (num traço com AA, a ponte que tem mais tinta);
3. a horizontal.

**Duas passadas.** A primeira passada só marca os pixels de traço, e a segunda liga, com uma regra extra: uma ponte nunca sai de um traço. Entre duas linhas paralelas a 1 px de distância (x − y = 0 e 2), a fileira branca do meio também "parece" traço. Com uma passada só, ligá-la roubava pixels das linhas e as partia. As três cadeias não cabem juntas em 4-conectividade, e ficam as duas que têm uma ponte livre.

**Pontes fora da cor.** O primeiro teste pelo motor mostrou que o traço sobrevivia, mas **vermelho**. A região tinha 15 px de tinta e 14 pontes brancas; a média dava um cinza de ≈ 0,49, e a paleta o levava ao vermelho. A ponte é um pixel de fundo que só existe para a topologia. Por isso o `PixelMap` ganhou `excludeFromColorStats`: o pixel continua na região (contagem, contatos, ponto simples), mas sai do `MedianAccumulator`, e nenhuma amostra o acompanha quando ele é movido depois. Toda decisão de cor (mediana, média da paleta, `ColorCache` do SubPixelSegmenter) sai das amostras, então todos os caminhos passam a ver só a tinta.

**Onde roda:**
- `PixelSegmenter::run`, no laço final: ligar → fundir minúsculas → diagonal, até nenhum dos três mudar. A ligação vem antes de `mergeSmallRegions`.
- O caminho da paleta exata do CoreEngine, no mesmo lugar.
- O SubPixelSegmenter recebe o mapa já ligado.

O campo é calculado uma vez por segmentação, sobre a imagem que o segmentador usa.

**Parâmetros** (`SegmentParams`, chaves `PixelSegmenter::…` no `.apf`, todos a calibrar):

| Parâmetro | Padrão | Faixa |
|---|---|---|
| `link_strokes` | `true` | — |
| `flow_sigma_px` | 1,0 | ≥ 0 |
| `stroke_min_coherence` | 0,5 | [0, 1] |
| `stroke_max_angle` | 0,7 rad (≈ 40°) | (0, π/4): em π/4 as duas diagonais empatam |

**Medido** (traços de 1 px pretos sobre branco, PixelSegmenter completo):

| Traço | Pixels | Ligações | Regiões |
|---|---|---|---|
| diagonal `\` e `/` (26 px) | 26 + 25 pontes | 25 | 2 |
| inclinação 1/3 | 36 | 12 | 2 |
| inclinação 2/3 | 36 | 23 | 2 |
| inclinação 3 | 34 | 11 | 2 |
| inclinação −1/2 | 36 | 18 | 2 |
| inclinação 1/8 | 38 | 4 | 2 |
| círculo r = 12 | 76 | 15 | 3 (dentro, anel, fora) |

Pelo motor, com o logo chapado mais o traço: 4 regiões, o traço preto e área de 29,67 px suavizada e 29,73 px em curvas (15 px de tinta + 14 pontes). A orientação de uma borda reta com AA erra no máximo 1,5° em 10 ângulos. Num disco duro, a tangente fica a ≥ 78,9° do raio, com coerência ≥ 0,80.

**Desvios e limites:**
- **Suavização.** O `ContourSmoother` não usa o campo. A mola preserva a área do traço ligado (29,67 px contra 29). Os dois lados de uma faixa de 1,4 px não se cruzam, porque o fiscal impede, e não colapsam em linha reta, porque o prior angular não encolhe uma reta. Um termo de largura guiado pelo campo só se justifica com uma falha medida, por exemplo nas fases sem fidelidade das fotos (`None`).
- **Traço mais grosso.** A faixa ligada tem largura √2 ≈ 1,41 px, contra 0,71 px da tinta: um traço ideal de 1 px a 45° dá 1,0. Só o modelo gerador corrigiria isso.
- **Inclinações rasas.** Um traço a menos de ≈ 5° de um eixo (α > 40° da diagonal) não é ligado. Ele vira trechos retos de comprimento ≥ 1/inclinação, que sobrevivem a `mergeSmallRegions` mas ficam separados pela quina.
- **Teste de ponto simples conservador.** Ele é local, e às vezes recusa as duas pontes de um par na ponta de um traço.
- **Traço com AA fraco.** Se o cinza varia mais que `mergeThreshMax` ao longo da diagonal, o traço não é ligado. Traços com AA costumam ser 4-conexos e não precisam da ligação.
- **Encontrado no caminho.** Na paleta exata, fragmentos que juntavam duas cores exatas recebiam, pela média, uma **terceira** cor (os pedaços vermelhos do traço preto sem ligação). Isso foi corrigido na §1.21.

**Testes:**
- `flow_field_test` (14 casos):
  - *Campo:* campo chapado, vazio e σ inválido; campo bruto igual ao tensor de Sobel e ao `EdgeImage`; integração como gaussiana normalizada (perfil exato de um degrau, raio exato, conservação da energia); centro de um traço de 1 px e identidade cos 2α = 2(t·d̂)² − 1; bordas em 10 ângulos e um disco; xadrez, cruzamento e canto.
  - *Ligação:* traço diagonal nas duas direções, com e sem ligação; 5 inclinações e um círculo; casos que não podem ligar (desligado, coerência mínima, pontes quase da cor do traço, xadrez, quadrados pela quina, cores alternadas, escadinha de AA, ângulo máximo) e o contorno entre duas cores, que liga; paralelas ligadas ao longo e nunca através; ponte com tinta; ponte que encostaria em outra região; pontes fora da cor e contrato de `excludeFromColorStats`; validação.
- `core_engine_test`: o traço pelo motor inteiro, com e sem ligação.
- `parameters_test` e `parameter_filer_test`: os 4 campos novos, com impressão digital, uma linha cada no `.apf` e as regras nos dois lados dos limites.

**Teste de mutação** (45 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (36).**
  - *Campo:* raio de 2σ, núcleo não normalizado, peso com σ no lugar de σ², sem a passada vertical, tensor integrado e tensor bruto sem a unidade de m², coerência ao quadrado, tangente igual à normal, anisotropia sem o fator 2 de Jxy, alinhamento com o sinal trocado ou sem o fator 2 de sin 2ψ, σ infinito aceito, tangente sem o teste de isotropia, magnitude do `EdgeImage` só a partir de Jxx.
  - *Ligação:* `linkStrokes` ignorado, cores de p e q não comparadas, mistura de AA aceita, alinhamento ignorado ou medido contra a outra diagonal, ângulo máximo sem dobrar, campo só de p, ponte tirada de um traço, ponte contada na cor, `run` e o caminho da paleta exata sem ligação, uma diagonal só, mistura sem a tolerância de reta, campo sem integração, parâmetros não validados, contador de ligações.
  - *`PixelMap`:* amostra acompanhando o pixel excluído, exclusão sem tirar a amostra, exclusão repetida.
  - *`.apf`:* π/4 aceito, coerência > 1 aceita, σ negativo aceito.
- **Sobreviventes fechados com testes (5):**
  - borda circular no lugar de replicada: um degrau vertical a duas colunas da moldura, com o valor exato na coluna 0 e zero na última;
  - contraste com as pontes ignorado: pontes quase da cor do traço, que então já tem 2 px;
  - coerência ignorada: com mínimo de 0,95, as pontas deixam de ligar (17 ligações em vez de 19);
  - ponte sem a preferência pela limpa: um quadradinho quase branco encostado na ponte horizontal de um único par;
  - ponte sem a preferência pela de tinta: as pontes cinza passaram para o lado que a ordem padrão não escolhe.

  Reexecutados (`-Only`), os 5 foram detectados.
- **Equivalentes, mantidos (4):**
  - o `min(1, ·)` na coerência e o `clamp(−1, 1)` no alinhamento são proteções contra arredondamento. Numa soma de produtos externos, xy² ≤ xx·yy (Cauchy–Schwarz), o que já limita os dois.
  - a saída antecipada "a ponte já é de p ou de q" está implicada pelo teste de contraste (a ponte teria a própria cor do traço).
  - o contraste mínimo entre as pontes no teste de mistura: com pontes de cores quase iguais, t cai fora de (0,02; 0,98), a menos que o traço tenha a cor delas, e aí o contraste recusa o par.

### 1.21 Cor das regiões no modo paleta exata

**O defeito.** Na paleta exata (imagem chapada), o mapa inicial é exato: cada região é uma componente 4-conexa de uma única cor da paleta. Depois, `mergeSmallRegions` e a passada diagonal juntam fragmentos de cores diferentes. A cor final vinha de `snapColorsToPalette`, que escolhe a cor da paleta mais próxima da **média** da região: é o passo de atribuição do k-means (§4), certo para uma paleta de k-means. Para cores exatas, a média de um fragmento com 2 pixels pretos e 2 brancos é o cinza 0,5, e a cor da paleta mais próxima desse cinza era o **vermelho** (0,42 contra 0,75). O resultado era uma cor que nenhum pixel da região tem. Pelo motor, com um traço diagonal preto sem ligação, saíam 7 fragmentos vermelhos.

**A regra.** `snapColorsToExactPalette` faz uma votação. Cada pixel vota na cor da paleta mais próxima da sua, a mesma regra que montou o mapa inicial, e a região recebe a cor mais votada. Os desempates, em ordem:
1. entre as cores empatadas, a mais próxima da média;
2. o menor índice.

A cor escolhida é sempre a de pelo menos um pixel da região. Os pixels fora das estatísticas de cor (as pontes de traço, §1.20) não votam, a menos que a região só tenha pontes; aí elas votam como reserva. O custo é O(N·K), e a memória O(regiões·K), com um índice denso por raiz.

`PixelSegmenter::finish` escolhe a regra pelo tipo da paleta: `palette->exact` usa a maioria, e uma paleta de k-means continua com a média. O SubPixelSegmenter (logos com AA) só recebe paletas de k-means e não muda.

**Medido:**
- Em 20 cenas de 16 cores exatas, com fusões aleatórias, havia 1370 regiões mistas.
  - Com a regra da média, 177 delas receberiam uma cor que nenhum pixel seu tem.
  - Com a maioria, nenhuma recebe, e a cor de cada região confere com a contagem feita à parte no teste.
- Pelo motor, sem a ligação de traços, a área vermelha volta a ser exatamente a do retângulo (266,01 px). Os fragmentos do traço desfeito ficam pretos ou brancos, conforme a maioria dos seus pixels.

**Testes** (`pixel_segmenter_test`):
- *Casos pequenos:*
  - 2 + 2 pixels: a média dá vermelho; a maioria dá branco, pelo empate exato e pelo menor índice;
  - maioria preta e maioria branca;
  - votos de pixels que não são exatamente cores da paleta;
  - empate decidido pela média;
  - pontes que não votam, e uma região só de pontes;
  - validação.
- *Propriedade:* a maioria depois de fusões aleatórias.
- *`finish`:* maioria só para a paleta exata.
- *`core_engine_test`:* a área vermelha igual à do retângulo, com e sem a ligação de traços.

**Teste de mutação** (11 defeitos injetados um por vez, em Release): todos detectados na primeira rodada.
- *Despacho:* `finish` sem a regra exata.
- *Votos:* pontes votando, sem a reserva, voto sem a cor do pixel, `nearest` sem trocar o melhor.
- *Desempates:* empate sem a média, empate exato indo para o maior índice, maioria não estrita.
- *Resultado e validação:* cor sempre a primeira da paleta, dimensões não conferidas, paleta vazia aceita.

### 1.22 `SignalImage`: o sinal pré-multiplicado

A especificação só dá o nome do módulo. Na arquitetura (§0, item 2, e §1.4), ele é a imagem achatada e reamostrada **antes** do desfoque, que serve de alvo ao modelo gerador. Ao implementá-lo, encontrei três defeitos no pré-processamento que existia. A média de área, o desfoque e a ampliação eram feitos em RGBA **straight** (não pré-multiplicado), em 8 bits:
1. **Franja.** A média mistura o RGB escondido sob alfa 0, que não é visível. Vermelho opaco ao lado de um transparente vira, ao reduzir 2 → 1, rosa (255, 128, 128, 128) se o RGB escondido é branco, vermelho-escuro (128, 0, 0, 128) se é preto, e (136, 100, 45, 128) se é um verde qualquer. O certo é (255, 0, 0, 128): meia cobertura de vermelho puro.
2. **Fundo transparente partido.** Pixels invisíveis com RGB diferentes por baixo eram cores diferentes. Num logo com fundo transparente em que cada pixel invisível esconde um RGB aleatório, como editores costumam deixar, o fundo não era uma cor exata: a paleta exata falhava, e o fundo se dividia em regiões.
3. **Alvo quantizado.** O alvo do modelo gerador era a imagem de 8 bits já arredondada. A média exata de uma redução (por exemplo 100,5 níveis) se perdia.

**O módulo** ([SignalImage.h](../include/cnx/preprocess/SignalImage.h)):
- Cada pixel é (r·a, g·a, b·a, a) em float. É a grandeza que se soma fisicamente: cobertura, média de área e desfoque são combinações lineares dela, e todo pixel invisível é (0, 0, 0, 0).
- `fromStraight` pré-multiplica, em double, e guarda em float. `straight` e `toStraight8` desfazem a pré-multiplicação. Um alfa que arredonda para 0 vira (0, 0, 0, 0). A ida e volta é exata para todos os 256 × 256 pares (valor, alfa) com alfa > 0.
- Os três filtros saem de um filtro separável em double, com pesos por índice ("taps"):
  - média de área exata: conserva Σ sinal · área até 1e-5;
  - bilinear com centros de pixel: t limitado a [0, 1], sem extrapolar nas bordas;
  - gaussiana de raio ⌈3σ⌉, com pesos exp(−k²/2σ²) normalizados e bordas replicadas.

**Ligação:**
- O `preprocess` achata o alfa constante, como antes, e então pré-multiplica, reamostra e desfoca **no sinal**.
  - `PreprocessedImage::signal` é o sinal sem desfoque.
  - `image` é `signal.toStraight8()`.
  - `smooth` é o sinal desfocado, convertido para 8 bits.
- `resampleArea` e `gaussianBlur` (8 bits) continuam como atalhos sobre a `SignalImage`, sem cópia da conta.
- **GenerativeModel.** O CoreEngine usa `signal.premultiplied()` como alvo: float, sem quantizar. O modelo passou a **pré-multiplicar as cores das faces**: ĉ_p = Σ_f a_p(f)·(r·a, g·a, b·a, a)_f. É o espaço em que a mistura por cobertura é linear, e o mesmo do alvo. `targetFrom` também pré-multiplica. Para faces e imagens opacas nada muda; os testes de cobertura, energia e gradiente foram reescritos com as expectativas pré-multiplicadas (com o alfa = 1 nos casos de forma fechada).
- Paleta, segmentação, classificador e exportação continuam em straight. A cor exportada de uma face é a cor straight, que é o que SVG, PDF e EPS esperam.

**Medido:**
- Borda vertical exata em x = 10,5 entre vermelho opaco e transparente: energia **0,000055** com o alvo pré-multiplicado (só o arredondamento de 128/255 contra ½), contra **68,9** com o alvo straight.
- Pelo motor, o logo chapado com o fundo trocado por transparente com RGB escondido aleatório dá perfil LogoMed, paleta exata de 3 cores e 3 regiões, com uma face (0, 0, 0, 0).

**Limites:**
- O espaço de cor continua sendo o sRGB codificado. Média e AA em luz linear (gama) seriam mais físicos, mas os rasterizadores comuns misturam no sRGB codificado. Mudar isso exige medir contra imagens de referência.
- A paleta e a segmentação medem distâncias em straight. Para alfas parciais, a distância em pré-multiplicado seria mais coerente, mas não há caso medido que a justifique.

**Testes** (`signal_image_test`, 6 casos):
- *Pré-multiplicação:* ida e volta de 8 bits exaustiva; `straight` e o invisível; imagem vazia.
- *Média de área:* sem vazamento do RGB escondido (branco, preto e um verde), contra a média straight; média ponderada pelo alfa; conservação, valor exato de um pixel com razão não inteira, cópia no mesmo tamanho, validação e o meio nível que o 8 bits perderia.
- *Gaussiana e bilinear:* desfoque sem halo branco; conservação de um impulso; σ ≤ 0, NaN e ∞; raio exato; bilinear entre opaco e transparente, sem extrapolar nas bordas.
- *`preprocess`:* sinal e 8 bits derivados dele; transparente canônico; bordas sem franja; mesmos pixels sem reamostragem; σ infinito recusado.
- *Modelo:* alvo pré-multiplicado contra straight; `targetFrom` igual ao sinal; face semitransparente uniforme com energia 0.
- *`core_engine_test`:* fundo transparente com RGB escondido aleatório.

**Teste de mutação** (19 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (13):**
  - *Pré-multiplicação:* vermelho sem pré-multiplicar, 8 bits sem desfazer a pré-multiplicação, invisível não canônico, `straight` do invisível, arredondamento por truncamento.
  - *Filtros:* área sem dividir por r, área sem o último pixel parcial, bilinear extrapolando, gaussiana não normalizada, sem a passada vertical, σ infinito aceito.
  - *Modelo:* face sem pré-multiplicar, `targetFrom` sem pré-multiplicar.
- **Sobreviventes fechados com testes (4):**
  - raio de 2σ e σ no lugar de σ²: perfil exato da gaussiana sobre um impulso (eixo e diagonal) e zero a partir do raio;
  - área aceitando ampliar na altura: a validação de `h`;
  - `preprocess` reduzindo pelos 8 bits: o meio nível (100,5) que tem de sobreviver no sinal.

  Reexecutados (`-Only`), os 4 foram detectados.
- **Equivalentes, mantidos (2):**
  - o limite [0, 1] de `straight`: com c = 255, r·a e a arredondam para o mesmo float, e os filtros usam os mesmos pesos em todos os canais. Por isso r ≤ a vale, e o limite é só uma proteção.
  - o CoreEngine usar o alvo de 8 bits no lugar do sinal: a diferença é a quantização (≤ ½ nível), abaixo da tolerância dos testes do motor. A exatidão está coberta no teste do `preprocess` (o meio nível), que é o sinal que o CoreEngine repassa.

### 1.23 Leitura de imagens: PNG, JPEG e zlib próprios

**Decisão.** Os decodificadores são próprios, em C++20 puro (camada `cnx/io`), sem dependência nem arquivo de terceiros. As alternativas avaliadas foram o stb_image (domínio público, mas sem DPI e sem cancelamento no meio da decodificação) e o WIC do Windows (não portável). O núcleo continua portável. O adaptador Qt previsto (`QImageLoader`) fica opcional.

**Peças:**
- **`Inflate`** (RFC 1950/1951):
  - blocos armazenados, Huffman fixo e dinâmico;
  - Huffman canônico com tabela rápida de 9 bits e, para os códigos longos, o caminho bit a bit do `puff.c`;
  - códigos superdimensionados são recusados, e os incompletos só no caso que a RFC permite;
  - teto exato de saída: o PNG sabe o tamanho, e passar dele é erro, o que barra as bombas de descompressão;
  - Adler-32 e CRC-32 (tabela gerada em tempo de compilação).
- **`PngDecoder`:**
  - os 5 tipos de cor em todas as profundidades válidas, os 5 filtros, Adam7, tRNS (chave e alfa da paleta) e pHYs;
  - CRC errado num chunk crítico é erro; num chunk ancilar, o chunk é ignorado;
  - 16 → 8 bits por arredondamento; profundidades menores que 8 são escaladas exatamente.
- **`JpegDecoder`:**
  - baseline, sequencial estendido e progressivo (DC e AC, primeira passada e refinamento), com reinícios;
  - todos os modos decodificam para um buffer de coeficientes. A tabela de quantização é fixada no primeiro scan de cada componente, como no libjpeg; depois vêm a IDCT em double e o croma por interpolação linear centrada;
  - cores: cinza, YCbCr, RGB (pelo marcador Adobe ou pelos ids 'R', 'G', 'B') e CMYK/YCCK na convenção Adobe;
  - DPI pelo JFIF ou, sem ele, pelo EXIF; a orientação EXIF é aplicada;
  - bytes estranhos entre segmentos são pulados, como faz o libjpeg; falta de EOI com os scans completos é aceita; dados entrópicos truncados são erro.
- **`ImageFileLoader : IImageLoader`:**
  - o formato vem da assinatura, não da extensão;
  - limites de tamanho do arquivo e de pixels, conferidos antes de alocar;
  - o `stop_token` é consultado na leitura, a cada linha do PNG, a cada linha de MCUs do JPEG e a cada 64 KiB descomprimidos.

**Validação independente** (`tests/data`):
- `make_fixtures.py` (só a biblioteca padrão do Python) gera:
  - PNGs com todos os tipos de cor, profundidades, Adam7 (com passadas vazias), filtros variados, IDAT repartido e chunks ancilares (inclusive um com CRC errado). O esperado são as amostras de origem convertidas pela norma;
  - fluxos zlib e DEFLATE em vários níveis e estratégias;
  - JPEGs feitos por um codificador próprio em Python: DCT pela definição, tabelas de Huffman ótimas (anexo K.2), e o progressivo pelo algoritmo de codificação do libjpeg (EOBRUN e bits de correção). O esperado é a reconstrução dos coeficientes que o próprio codificador quantizou, com a orientação EXIF aplicada por composição de giros.
- `make_gdiplus_fixtures.ps1` gera um JPEG e um PNG pelo GDI+ do Windows (um codificador de terceiros), com a decodificação do próprio GDI+ como referência.

**Medido:**
- Os **36 JPEGs** decodificam **bit a bit iguais** à reconstrução de referência: 6 amostragens, cinza, reinícios, scans separados, DQT de 16 bits, 4 roteiros progressivos, RGB, CMYK, YCCK, DPI pelo JFIF e pelo EXIF e as 8 orientações.
- O JPEG do GDI+ difere em no máximo **2 níveis** (média de 0,21), por causa de outra IDCT e de outra ampliação do croma.
- Os **36 PNGs** e os **11 fluxos zlib/DEFLATE** são idênticos.
- Velocidade, 4000 × 3000 (Release, uma thread): PNG em 0,41–0,49 s e JPEG em 0,92 s.

**Limites:**
- **Gerência de cor:** não há. gAMA, cHRM, sRGB, iCCP e perfis ICC do JPEG são ignorados, e os valores saem como estão no arquivo.
- **JPEG não suportado:** codificação aritmética, sem perdas, hierárquico, 12 bits e DNL. Todos dão erro com o motivo.
- **Outros formatos:** só PNG e JPEG (nada de GIF, BMP, TIFF ou WebP), e do APNG só a imagem padrão. O EXIF de PNG (eXIf) não é lido.
- **Velocidade do JPEG:** a IDCT em double e a ampliação do croma por pixel são cerca de 10–20× mais lentas que o libjpeg-turbo. Para o motor isso é aceitável, porque a vetorização leva segundos ou minutos; se for preciso, os caminhos óbvios são uma IDCT AAN em inteiros e linhas inteiras de croma.

**Segurança de memória.** A build Debug (asserções da biblioteca padrão) achou um defeito que a Release escondia. Numa tabela de Huffman JPEG superdimensionada, o código passava de 2^len *antes* da verificação do fim do comprimento, e o preenchimento da tabela rápida escrevia fora dela. Um arquivo malformado chegava a isso. A verificação agora vem antes de cada uso. Para os erros que ninguém pensou em montar, há um fuzz determinístico: 3600 variações corrompidas (bytes trocados, cortes, FF inseridos) de 9 arquivos de referência. Em Debug, todas terminam em `ImageDecodeError` ou numa imagem, sem nenhuma outra exceção nem asserção (3238 recusadas, 362 decodificadas).

**Testes** (`image_decoder_test`, 8 casos):
- os 72 arquivos de referência, com pixels, dimensões, DPI, alfa e formato;
- os fluxos zlib e as somas de verificação conhecidas;
- fluxos zlib e DEFLATE inválidos, montados bit a bit, cada um com o motivo;
- PNGs inválidos montados à mão: assinatura, CRC, IHDR, dados a mais ou a menos, filtro, chunk crítico desconhecido, paleta, tRNS, IDAT não consecutivos, limite;
- JPEGs inválidos derivados de um válido: SOI, modos não suportados, 12 bits, DNL, tabelas ausentes ou superdimensionadas, truncamento, componente, dois quadros, RST ausente, limite;
- o fuzz de 3600 arquivos corrompidos;
- o carregador: assinatura contra extensão, arquivo inexistente, limites e cancelamento;
- o CoreEngine vetorizando um PNG lido do disco, e falhando com a mensagem certa num arquivo que não é imagem.

**Teste de mutação** (82 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (77):**
  - *zlib:* NLEN, bits extras de comprimento e de distância, distância antes do início, teto de saída, cópia com sobreposição, ordem e bases das repetições do código de comprimentos, literais incompletos, Adler-32, tamanho exato, FCHECK, janela, dicionário, módulo do Adler, polinômio do CRC.
  - *PNG:* CRC crítico, filtros Average, Sub e Paeth, 16 bits e profundidades baixas, ordem dos bits, Adam7, tRNS de cinza e da paleta, IDAT não consecutivos, chunk crítico desconhecido, pHYs, alfa por tRNS, passadas vazias, combinações inválidas, índice fora da paleta, PLTE exigido.
  - *JPEG:* zigue-zague, EXTEND, predição do DC, ZRL, EOBRUN nas duas passadas, sinal e contagem do refinamento, bit do refinamento do DC, deslocamentos progressivos, reinícios (predição e marcador), IDCT, nível 128, quantização fixada no primeiro scan, constantes do YCbCr, croma centrado, Adobe RGB, ids RGB, CMYK, YCCK, orientação (ignorada, 6 como 8, DPI sem troca), unidades do JFIF e do EXIF, precedência do JFIF, EXIF big-endian, precisão, Huffman superdimensionado, truncamento, bytes estranhos, enchimento, falta de EOI.
  - *Carregador:* assinatura de 3 bytes, limite do arquivo, parada, limite de pixels nos dois caminhos.
- **Sobreviventes fechados com testes (2):**
  - chave tRNS de RGB comparada só no vermelho: três pixels que diferem só no verde e no azul;
  - tabela AC não exigida: a tabela AC 0 levada para outro destino tem de dar "AC ausente", e não ser decodificada como lixo.

  Reexecutados (`-Only`), os 2 foram detectados.
- **Equivalentes, mantidos (3):**
  - *empate do Paeth* (`pa ≤ pb` contra `pa < pb`): pa = pb com a ≠ b obriga c = (a + b)/2, logo pc = 0 e c vence; o empate só ocorre com a = b;
  - *refinamento AC de um coeficiente com o bit já ligado:* num fluxo válido esse bit é sempre 0, porque os scans anteriores pararam em Al + 1. A condição só protege contra fluxo corrompido, como no libjpeg;
  - *croma limitado a n − 1:* o maior f possível é menor que n, e i1 já é limitado a n − 1, então o limite superior não muda nada.

### 1.24 `Exporter` e `SvgWriter`

**Exporter** ([Exporter.h](../include/cnx/export/Exporter.h)): é independente de formato e emite o protocolo `IShapeWriter`.
- **Laços.** Para cada meia-aresta, a direta escreve as curvas da aresta em ordem; a inversa escreve as mesmas curvas em ordem reversa, cada uma invertida (P3, P2, P1, P0). Como a fronteira existe uma vez e o texto sai dos mesmos doubles, as duas faces escrevem os **mesmos números**, mesmo arredondados a 2 casas: zero frestas e zero sobreposição.
  - Uma aresta sem curvas (antes do ajuste) sai como a poligonal dos nós, só com `L`.
  - Uma reta ajustada (`SegmentKind::Line`) sai como `L`, e não como uma cúbica degenerada.
- **Modos** (`cake_stack_mode`):
  - *bolo em camadas:* só o laço externo de cada face;
  - *caminhos compostos:* laço externo mais furos, com orientação oposta, que a regra nonzero resolve sem `fill-rule` (invariante I2);
  - *compostos agrupados por cor:* um grupo por cor exata, na ordem da 1ª aparição.
- **Ordem de desenho:** profundidade na árvore de `Face::parent`, depois o id. O pai vem antes dos filhos, o que o modo bolo exige.
- **Fundo removido** (`BackgroundSelection`): a face não é escrita. Nos compostos ela vira buraco; no bolo, aparece o pai. Faces com alfa 0 também não são escritas.
- **Arquivo atômico:** `writeFile` grava num temporário ao lado do destino e renomeia. Em erro ou cancelamento, o temporário é apagado e o destino fica intacto. O título do SVG é o nome do arquivo.
- **`ExportOutputStage`:** é o estágio Output do CoreEngine. Grava cada `ExportRequest`, com as configurações do pedido, a partir de `snapshot.fitted`.

**SvgWriter** ([SvgWriter.h](../include/cnx/export/SvgWriter.h)):
- `<svg width="{W·ppp}pt" height="{H·ppp}pt" viewBox="0 0 W H">`: tamanho em pontos e coordenadas em pixels.
- Caminhos `M`/`L`/`C`/`Z` com `coordinate_decimals` casas fixas, pelo `std::to_chars`: independente de locale, sem notação científica e nunca "-0".
- `opacity` com 2 casas, como pede a especificação (1.00 quando opaco); o alfa fica quantizado a ±0,005.
- Traço anti-fresta da cor do preenchimento, `stroke-width` = 0,09375 pt convertido para unidades do viewBox (pt / ppp), na menor representação exata.
- **Desvio da especificação:** o id do grupo é `c_rrggbbaa`, não `#rrggbbaa`. Um id XML não pode começar por '#' nem por dígito; com '#' o arquivo falharia num validador.

**Formatos:** EPS, AI, PDF e DXF continuam com `makeWriter` lançando `ExportError` com o motivo. São escritores do mesmo protocolo e entram depois.

**Medido:**
- **Cenas de blocos aleatórios** (junções e ilhas) e uma **cena com 3 níveis de furos e uma sela**: em 5 pontos por pixel, cada ponto cai em **exatamente um** caminho composto, e da face certa; no modo bolo, a última pintura é a face certa.
- **Curvas reais** (discos e anéis suavizados e ajustados, 26–28 cúbicas por cena):
  - 4000 pontos ao acaso, com 2 e com 6 casas decimais: nenhuma fresta e nenhuma sobreposição;
  - área de cada forma igual à área de Green das cúbicas (`loopArea`) com erro relativo ≤ 7,6·10⁻⁵, que é o da discretização do rasterizador de teste.
- **Pelo motor:** uma imagem com retângulo, anel e furo dá 4 caminhos, 7 anéis e 10 cúbicas. Os 1420 pixels fora da borda saem com a cor exata. Na borda, o círculo vetorial e a escadinha de pixels diferem em meio pixel por natureza, e 96,35 % de todos os pixels batem.
- **Verificação visual num navegador real:** a 10x sobre fundo verde, nenhuma fresta.

**Achado da verificação visual (fora do escopo deste módulo).** Vetorizei um logo de 240×160 com AA ("cnx" em branco sobre azul, anel vermelho) e exportei cada estágio:
- a **segmentação** está certa: as letras saem em escadinha, mas inteiras;
- a **suavização** (`ContourSmoother`, perfil LogoMedAA) **deforma as formas**: as letras viram bolhas, a perna esquerda do "n" se inclina e se separa, os cantos do "x" e do retângulo somem, e a borda do retângulo junto ao anel incha;
- o ajuste de Bézier segue a poligonal suavizada e acrescenta um esporão no pé do "n".

Os testes de suavização, que olham energia, cruzamentos e área, não pegam isso. Os pesos dos priors contra a fidelidade e a perfuração de cantos estão marcados "a calibrar" (§1.13), e este era o próximo problema de qualidade do motor. **Corrigido no §1.25.**

**Teste de mutação** (30 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (26):**
  - *Geometria:* curva inversa sem inverter, curvas inversas na ordem direta, compostos sem furos, bolo com furos.
  - *Seleção:* fundo não removido, alfa 0 escrito, sem agrupar por cor, sem traço.
  - *Arquivo:* cancelamento ignorado, temporário não apagado, validações (casas > 9, traço negativo, escala não positiva), título.
  - *SVG:* "-0", cor truncada, traço e tamanho sem a conversão de unidades, "&" sem escapar, grupo sem alfa, opacidade com 3 casas, espaço antes do primeiro `M`, recuo, coordenada não finita.
  - *Estágio:* snapshot sem ajuste aceito, configurações do pedido ignoradas.
- **Sobreviventes fechados com testes (4):**
  - *Poligonal inversa começando no nó errado:* nenhuma cena do tracer tem um laço que comece numa meia-aresta inversa de aresta aberta. Um teste procura cenas assim (acha várias) e confere cobertura e face certa nelas.
  - *Reta escrita como curva:* a cena dos retângulos ajustada tem de produzir mais de 10 `L`.
  - *Ordem sem profundidade:* nas cenas do tracer o pai sempre tem o id menor, então a ordenação nunca mudava nada. Um teste com uma imagem montada à mão, em que o filho tem o id menor, exige pai antes do filho, e o modo bolo exige o azul antes do vermelho.
  - *Cópia no lugar da renomeação:* depois de todas as gravações, o diretório só pode ter o destino.

  Reexecutados (`-Only`), os 4 foram detectados.

**Testes** (`svg_export_test`, 9 casos):
- *Escritor:* o texto exato de um documento com grupo, traço, opacidade, título escapado, `-0`, casas fixas e cores.
- *Compostos:* cobertura única e face certa (6 cenas aleatórias e a aninhada), inclusive laços que começam numa meia-aresta inversa.
- *Ordem:* pai antes do filho com ids invertidos.
- *Bolo e agrupado:* a face pintada certa, um grupo por cor, ids válidos e únicos.
- *Curvas:* sem frestas com 2 e 6 casas, e áreas de Green.
- *Opções:* traço, opacidade, alfa 0, tamanho em pt, fundo removido nos dois modos, ordem pai-filho, validação.
- *Arquivo:* atômico, formato não implementado sem tocar o destino, cancelamento, diretório inexistente, nenhum temporário sobrando, substituição.
- *Motor:* `ExportOutputStage` com as configurações do pedido, o SVG redesenhando a imagem, e a recusa de um snapshot sem ajuste.

### 1.25 Correção da suavização de contornos: letras e cantos de logos com AA

O achado do §1.24 (letras viram bolhas, cantos somem, o fio engorda) foi tratado como defeito de medição antes de defeito de código: os testes de suavização olhavam energia, cruzamentos e área, e nenhum comparava o **resultado** com a imagem.

**Como se mede** (`contour_quality_test`, com `svg_support.h` e `logo_scenes.h`). Três cenas de logo com AA por superamostragem exata são vetorizadas pelo motor, cada estágio é exportado em SVG (`Exporter` + `SvgWriter`), lido de volta e rasterizado (4×4 pontos por pixel). O rasterizado é comparado com o sinal pré-multiplicado, que é o alvo do modelo gerador. Três números por estágio: rmse, maior erro e pixels com erro > 0,2. Nas letras, mede-se também a distância dos 16 vértices verdadeiros ao contorno vetorial. As cenas: *letras* (L, T, I e um bloco, com 16 cantos conhecidos), *formas* (fio preto de 34 × 2,4 px, triângulos, disco amarelo) e *fios* (traços finos, inclusive diagonais).

**Antes e depois** (MSVC Release, rmse contra o alvo):

| Cena | Estágio | Antes | Depois |
|---|---|---|---|
| letras | suavizado | 0,0201 | **0,0129** (escadinha: 0,0339) |
| letras | Bézier | 0,0224 | **0,0180** |
| formas | suavizado | 0,1064 | **0,0341** (escadinha: 0,0507) |
| formas | Bézier | 0,1085 | **0,0342** |

Nas letras, o vértice verdadeiro fica a 0,27 px do suavizado e a 0,23 px do Bézier (máx. 0,94); sem a perfuração de cantos seria 0,77 px (máx. 1,00). O fio preto de 34 × 2,4 px sai com caixa de 34,4 × 3,0 px (a escadinha tem 34 × 3), e o disco amarelo tem desvio radial máximo de 0,09 px. Antes, o fio virava um bloco quase quadrado e os cantos das letras ficavam arredondados.

**Quatro causas, cada uma com a correção que a trata**

1. **Espaçamento não invariante à escala.** O termo `(‖b‖ − ‖a‖)²` custava (34 − 3)² = 961 por canto de um fio de 34 × 3 px. O tracer remove os nós colineares, então segmentos de comprimentos muito diferentes são normais, e o suavizador igualava tudo achatando o fio. *Correção:* diferença relativa s = (lb − la)/(la + lb) ∈ [−1, 1], com gradiente analítico (conferido contra diferenças finitas). Num fio de 34 × 3 px com 4 cantos, a energia de espaçamento passa de 4 · 961 = 3844 para 2,81 (s = ±31/37 em cada nó), e o termo mantém o papel que o §1.13 lhe dá (evitar o amontoamento de nós).
2. **Os cantos eram perfurados tarde demais.** A perfuração só rodava na fase 1, depois de a fase 0 já ter arredondado cada canto, e então o detector via curvas, não ângulos. *Correção:* `punctureOnPhase` passa a 0, e a detecção roda sobre a escadinha original.
3. **A escadinha reprova o teste de retidão.** Os degraus de uma borda inclinada se afastam da corda em ≥ 0,35 px (limiar: 0,3), e um fio ou a ponta de um triângulo têm lados menores que o braço de 4 px. *Correção:* `smoothChain` (EnergyTerms) calcula a média da posição ao longo do arco numa janela [s − h, s + h] (integral exata por segmento; extremos fixos em cadeia aberta, janela periódica em cadeia fechada), com h = max(braço/4, 1 px). A detecção usa **duas escalas** de braço (o inteiro e a metade) e marca o nó se qualquer uma o aprovar. O piso de 1 px é o degrau do pixel; com 0,5 px, discos pequenos geravam cantos falsos.
4. **O ajuste de Bézier só vê os nós.** Depois do tracer, uma borda reta de 18 px tem 2 nós, e uma cúbica pode se afastar de 1 px no meio dela sem custo. *Correção:* `BezierFitter::run` insere, em segmentos com mais de 2 px, pontos intermediários que servem só de amostras (não são nós), e uma cúbica cujos controles ficam a ≤ 0,02 px da corda, em ordem, sai como `Line`. Num "D" com reta de 18 px, o desvio máximo da poligonal cai de 0,872 px para 0,168 px.

**Limites e achados fora do escopo (não corrigidos)**
- **Seleção de K da paleta.** Em cenas com 3 ou 6 tintas, o `PaletteFinder` escolhe K = 2. Os testes forçam `forcedNumColors = inks`. Defeito da paleta, não da suavização.
- **Franja de cor na sombra.** A regra de sombras do `SubPixelSegmenter` transforma uma linha cinza de AA numa terceira tinta (franja vermelha ou amarela em torno de um traço preto).
- **Traços diagonais finos (2,8 px).** As pontas não são detectadas como cantos e a cúbica ultrapassa o traço nas pontas. Na cena *fios* o Bézier piora em relação ao suavizado (0,0557 contra 0,0475) e chega a erro máximo de 0,863, e o teste só exige `Bézier < 1,6 × suavizado`.
- Os limiares de canto (0,8 rad, 0,3 px, braço de 4 px) continuam provisórios e seguem a calibrar num conjunto de referência.

**Teste de mutação** (39 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (33).**
  - *Espaçamento:* forma absoluta, sem o quadrado, e quatro erros nos termos do gradiente (fator 2 de ∂s, sinal do nó seguinte, termo de `a` no nó do meio, `la` no lugar de `lb`).
  - *Filtro de arco:* cadeia fechada sem o segmento de fechamento, janela aberta sem encolher nas pontas, só no início, ou com o extremo movido, volta seguinte da janela periódica, divisão por `h` no lugar de `half`, trapézio sem a metade, fração inicial do segmento, comprimento de arco errado.
  - *Cantos:* uma escala só de braço, filtro sem a janela mínima (e com 0,5 px ou 3 px), sem filtrar a escadinha, janela igual ao braço inteiro, perfuração na fase 1 (no código e no padrão), só a escala grande decidindo.
  - *Ajuste de Bézier:* amostras a cada 4 px, sem amostras, com `floor`, amostras viradas em quebras, desvio lateral da reta ignorado, reta sempre ou nunca, contador de retas, controles da reta errados.
- **Seis escaparam; cinco foram fechados:**
  - *Segmento nulo aceito (S7):* nós coincidentes dão 0/0. O teste novo exige energia e gradiente finitos nesse caso.
  - *Janela periódica sem a volta anterior (M4):* a volta só importa quando o segmento de fechamento é menor que a meia janela, e o quadrado do teste anterior não tinha isso. Um polígono com segmento de fechamento de 0,5 px agora exige que girar o nó inicial gire o resultado.
  - *Tolerância de reta de 0,5 px (B6), ordem (B8) e fim (B9):* o `isStraight` estava dentro de um espaço de nomes anônimo e só era exercitado por ajustes reais. Passou a ser `BezierFitter::isStraight` e ganhou um teste direto (limites de 0,02 px, arco raso de 0,3 px, P1 atrás de P0, P1 e P2 fora de ordem, P2 depois de P3, corda nula, corda fora dos eixos).
  - Reexecutados com `-Only`, os cinco foram detectados.
- **Um escapa (B4):** espaçar as amostras intermediárias por `k/(pieces+1)` no lugar de `k/pieces`. As amostras continuam sobre o segmento; só o último trecho passa de ≤ 2 px para até 2·len/(pieces+1). Nenhum caso testado muda de forma observável, e não consegui construir um que mude. É uma lacuna de cobertura real e não verificada.
- **Armadilha do método.** Na primeira reexecução, S7 e M4 sobreviveram porque a cópia dos arquivos preservou datas antigas e o `ninja` não recompilou o teste novo. Ao reexecutar mutantes depois de mudar testes, as datas dos arquivos copiados têm de ser atualizadas (o script, que reaproveita uma cópia compilada, não faz isso sozinho).

**Testes novos ou alterados**
- `contour_quality_test` (3 casos), com limiares sobre rmse, maior erro, cantos, caixa do fio e raio do disco. Os limiares ficam acima do medido com folga, para o teste detectar regressão e não ruído.
- `energy_terms_test` (12): o espaçamento tem valor fechado (4/81), é invariante à escala e limitado a 1; o filtro de arco remove a escada de 45° (0,707 px → 0,000 px) e mantém um canto de 90°.
- `contour_smoother_test` (11): cantos sobre a escadinha (discos e anel 0; retângulos 4; triângulo 3; losango 4 a 8, porque cada ponta cega sai como 2 nós vizinhos), perfuração em qualquer fase e padrão na primeira.
- `bezier_fitter_test` (26): sem barriga em retas longas com nós esparsos, trechos retos viram `Line`, e `BezierFitter::isStraight` testado diretamente.
- `energy_terms_test`: além do descrito, nós coincidentes (energia e gradiente finitos) e um polígono fechado com segmento de fechamento curto (a rotação do nó inicial não muda o resultado).

### 1.26 Franja de terceira cor em volta de formas com AA: a regra de sombra

O §1.25 deixou como achado que a regra de sombra do `SubPixelSegmenter` cria uma franja de cor em volta de traços com AA. Aqui está a medição, a causa e a correção.

**Como se mede.** Um pixel tem **franja** quando a tinta final da sua região (índice da paleta) não existe em nenhum ponto dele: o pixel de borda entre vermelho e branco só pode ser vermelho ou branco. A verdade vem da própria cena (8 × 8 pontos por pixel, `fringePixels` em `sub_pixel_segmenter_test`), sem tolerância de cor nem comparação com o rmse. A ferramenta de diagnóstico que rodou os estágios um a um sobre o pixel afetado ficou fora do repositório; o que ela mostrou está abaixo.

| Cena | Antes | Depois |
|---|---|---|
| formas (retângulo, anel, triângulo, fio, disco) | 11 regiões, **100 pixels com franja** | **7 regiões** (as verdadeiras), **0** |
| letras | 5 regiões, 0 | 5 regiões, 0 |

**Causa, estágio a estágio**, no pixel de AA da borda do retângulo vermelho (rosa 238,148,148):

1. O `PixelSegmenter` o deixa numa faixa de 33 px, com a tinta amarela (o amarelo é a cor da paleta mais próxima do rosa em RGB: distância² 16 628 contra 23 187 para o branco e 23 652 para o vermelho).
2. As **praias** o corrigem: movem a faixa para o fundo branco (praias 14, 198 pixels movidos). Até aqui está certo.
3. A regra de sombra então marca o **fundo branco inteiro** (4709 px) como sombra. Os pixels de AA que as praias entregaram ao fundo ficam todos na borda dele, e somados à variância do fundo chapado dão a variância de um gradiente (≥ 2e-3), que o desfoque de caixa ainda preserva.
4. A reconstrução refaz o fundo "pela cor de paleta mais próxima da imagem desfocada". O desfoque de caixa mistura as tintas vizinhas na borda, e o rosa que sai dele volta a ser, em RGB, mais amarelo que branco: 29 regiões novas, entre elas a franja amarela do retângulo.

**Três correções**, uma por defeito (`SubPixelSegmenter.cpp`):

1. **Detecção só pelos pixels interiores** (os 4 vizinhos na mesma região). O pixel de borda de uma região é, por construção, a mistura com a vizinha; sozinho ele dá a um fundo chapado a variância de um gradiente. `shadowMinPixels` passa a contar só os interiores. Um gradiente verdadeiro tem interior (testado, `shadow_detection_still_flags_a_gradient_that_only_its_interior_carries`).
2. **Desfoque restrito à região** (`maskedBlur`: a média da janela só usa pixels da própria região). O desfoque de caixa antigo misturava a tinta vizinha na borda de qualquer sombra legítima; agora o cinza de um gradiente branco–preto não recebe o vermelho que toca o pé dele.
3. **Tintas candidatas da reconstrução** (`shadowInks`). Cada pixel era classificado pela cor de paleta mais próxima em RGB, e para um tom intermediário essa pode ser uma terceira tinta. Agora a sombra vai de uma tinta vizinha a outra:
   - as candidatas são as tintas das regiões vizinhas;
   - com mais de duas, fica o **par de extremos** cuja reta melhor ajusta as cores desfocadas da região (um teste mostrou o porquê: com o vermelho tocando só o pé do gradiente, o cinza do meio fica de fato mais perto do vermelho que do branco ou do preto);
   - entram também as cores da paleta que estão **sobre a reta** entre os dois extremos, com a tolerância da praia (`beachLineTolerance`): um cinza 128 na paleta é mistura legítima de branco e preto e ganha a sua faixa;
   - sem vizinhas (região única), vale a paleta inteira.

**Efeito no rmse.** O rmse de "formas" **sobe**: suavizado 0,0341 → 0,0390 e escadinha 0,0507 → 0,0542. É a métrica que engana aqui, não a correção: a faixa amarela, errada, ficava mais perto do rosa em RGB que o branco, certo. Por forma (suavizado): anel 0,0062 → 0,0065, triângulo 0,0041 → 0,0043, fio 0,1058 → 0,1068 e disco 0,0099 → 0,0098 não mudam. O retângulo vai de 0,0678 a 0,0828, e a causa **não é a franja**:
- as quatro bordas dele ficam a meio pixel exato, e as linhas de AA (50 % de cobertura) empatam e vão todas para fora; a escadinha sai com 31 × 23 e a forma verdadeira tem 32 × 24;
- a **pressão do ar** (contornos com menos de 8 nós, aqui um retângulo de 4) segura a área em 713 contra os 768 verdadeiros, e a suavização só consegue deslocar o retângulo inteiro em ~0,25 px (de (9,7)–(40,30) para (9,23; 7,24)–(40,26; 30,25));
- antes, as faixas amarelas faziam o contorno ter 12 nós e escapar da pressão do ar, o que escondia isso.

O Bézier do disco passa de 0,0116 a 0,0163 (desvio radial máximo 0,09 → 0,16 px); não investiguei. O gradiente conjugado é conjunto, então remover contornos alheios muda a trajetória dos demais.

**Limites e achados fora do escopo (não corrigidos)**
- **Pressão do ar por contagem de nós.** O critério "< 8 nós" é da especificação, mas um retângulo grande de 4 nós não é uma forma pequena que possa sumir. A pressão do ar ancorada na área da escadinha viesa retângulos grandes quando a escadinha perde os pixels de AA. A correção natural é aplicá-la por tamanho (área ou perímetro) e não por nós.
- **Empate de 50 % nas praias.** `t < ½ → A` manda todo pixel de cobertura exatamente ½ para o mesmo lado. Aqui só aparece em cenas construídas com bordas em meio pixel.
- Os pixels de AA que não são praia (cantos, junções, fios de 1 px) ainda recebem, pelo `snapColorsToPalette`, a tinta mais próxima em RGB, que pode ser uma terceira. Medido: 0 em "formas" e "letras" depois desta correção, mas a regra continua a mesma.

**Testes** (`sub_pixel_segmenter_test`, 25; `contour_quality_test`):
- `a_flat_background_with_beach_halo_is_not_flagged_as_a_shadow`: retângulo com AA sobre branco e uma terceira tinta na paleta; depois das praias, o fundo não é sombra.
- `antialiased_logos_come_out_without_a_third_ink_fringe`: "formas" e "letras" pelo pipeline completo, 0 pixels de franja, 7 e 5 regiões, todas conexas.
- `a_fade_between_white_and_black_never_gets_a_band_in_a_third_ink`: gradiente com amarelo na paleta, duas faixas com a fronteira no meio.
- `a_palette_gray_on_the_line_between_the_neighbor_inks_still_gets_its_band`: o cinza sobre a reta ganha as três faixas.
- `the_blur_does_not_bleed_a_neighbor_ink_into_the_fade` e `the_pair_of_extremes_is_chosen_from_all_the_colors_of_the_region`: o vermelho que toca o pé ou o topo do gradiente não entra nele, em oito disposições (em cima ou embaixo, criado antes ou depois das outras regiões, espelhado ou não); a segunda usa um platô de branco puro para que o último pixel da região empate dois pares de extremos e só as demais cores desempatem.
- `the_blur_radius_parameter_controls_the_reconstruction` (raio 2 dá uma faixa clara, raio 8 nenhuma) e `a_lone_fade_region_is_rebuilt_with_the_whole_palette` (região única, sem vizinhas).
- `shadow_detection_counts_only_interior_pixels` (inclusive uma região que ocupa a imagem inteira, cujo anel externo não é interior) e `shadow_detection_still_flags_a_gradient_that_only_its_interior_carries` (o mínimo de pixels conta só os interiores).
- `contour_quality_test`: o único caminho amarelo da cena "formas" é o disco.
- Seis dos testes da primeira versão falham no código antigo (conferido); o do cinza sobre a reta passa nos dois, como esperado.

**Teste de mutação** (20 defeitos injetados um por vez, em Release):
- **Detectados na primeira rodada (13).**
  - *Detecção:* pixels de borda na variância, interior com 3 vizinhos, mínimo de pixels sobre a região inteira.
  - *Desfoque:* janela assimétrica, divisão por constante, reconstrução pela imagem bruta, sem a máscara da região.
  - *Tintas:* candidatas só do par de extremos com 4 ou mais vizinhas, qualquer cor de paleta valendo, cor sobre a reta nunca valendo, tolerância dobrada, tintas das vizinhas pela cor da própria região, candidata mais distante.
- **Sete escaparam; seis foram fechados com testes:**
  - *Fora da imagem contando como interior (D3):* nenhuma região tocava a borda da imagem. O teste do anel externo de uma região que ocupa tudo exige raio 0, porque com raio 1 o suavizado já reprovava o anel por si só.
  - *Par de extremos nunca escolhido (D8), ajuste do par só com a última cor (D9), cores da região fora do ajuste (D16):* a cena de um só vermelho embaixo escolhia o par certo por acaso (a ordem das vizinhas). Passou a haver oito disposições, e o platô de branco puro faz o último pixel empatar dois pares.
  - *Sem vizinhas usando só a primeira cor (D14):* o teste da região única.
  - *Raio do desfoque fixo (D20):* o teste do raio 2 contra 8.
  - Reexecutados com `-Only`, os seis foram detectados; a rodada final com os 20 confirmou 19 detectados.
- **Um escapa (D12):** a distância à reta em vez do segmento (sem o `clamp`), ou seja, aceitar como mistura uma cor da paleta que está na extensão da reta além dos dois extremos. Exigiria uma sombra cujas cores passam dos extremos (um degradê de 255 a 0 entre vizinhas de cinza 100 e 200), o que não é uma cena realista; não construí o caso. É uma lacuna de cobertura, que também é uma escolha de projeto: "entre os extremos" foi o critério adotado.


---

## 2. Formulação matemática

### 2.1 Convenções

```
Coordenadas: x → direita, y → baixo. Pixel p = (i, j) ocupa [i, i+1) × [j, j+1).
Área assinada (shoelace):  A(P) = ½ Σ_k (x_k·y_{k+1} − x_{k+1}·y_k)
Laço externo de uma face: A > 0.  Furos: A < 0.
Segmento orientado s_k = [P_k, P_{k+1}],  Δ_k = P_{k+1} − P_k
Normal externa não normalizada:  n_k = (Δy_k, −Δx_k)   (‖n_k‖ = ‖Δ_k‖; aponta para fora da face de A > 0)
Ponto de s_k:  P(t) = P_k + t·Δ_k,  t ∈ [0, 1]
```

Uma aresta com faces (L, R) é orientada de modo que L fique do lado de área positiva: n_k aponta para fora de L e para dentro de R.

### 2.2 `GenerativeModel`: modelo direto exato com AA de filtro-caixa

Cor sintetizada do pixel p, onde a_p(f) é a área de f ∩ p:

```
ĉ_p = Σ_f a_p(f) · c_f,     a_p(f) = |f ∩ p| ∈ [0, 1],     Σ_f a_p(f) = 1
```

#### 2.2.1 Migalhas de pão (`dropBreadCrumbs`)

Para cada segmento s_k, os parâmetros onde ele cruza as linhas inteiras da grade são

```
{ (m − x_k)/Δx_k : m ∈ ℤ }  ∪  { (m − y_k)/Δy_k : m ∈ ℤ },   restritos a (0, 1)
```

São duas progressões aritméticas, intercaladas por DDA (Amanatides–Woo) sem ordenação, em O(número de cruzamentos). Cada subintervalo [t_a, t_b] fica inteiramente dentro de um único pixel, identificado pelo ponto médio. Cada migalha é a tripla (p, t_a, t_b).

#### 2.2.2 Área exata por pixel (Green), substituindo `closePixel`

Aplicando o teorema de Green ao campo F = (0, x − (i+1)) em f ∩ p, a parte da fronteira sobre o lado direito do pixel contribui zero e o lado esquerdo contribui com o comprimento coberto. Para cada pedaço de fronteira dentro do pixel, com

```
δy = (t_b − t_a)·Δy_k     e     x̄ = x_k + ½(t_a + t_b)·Δx_k
```

a cobertura do pixel (i, j) é

```
a_{(i,j)}(f) =   Σ_{pedaços de ∂f em (i,j)}  −δy · ((i+1) − x̄)        ← trapézio até o lado direito
               + Σ_{pedaços de ∂f na linha j com coluna < i}  −δy     ← cobertura do lado esquerdo
```

Isso se implementa com um acumulador por linha:

```
acc_j[i]   += α,    α = −δy·((i+1) − x̄)
acc_j[i+1] += γ − α,  γ = −δy
a_{(i,j)}  = Σ_{i' ≤ i} acc_j[i']                                    [V: 1.3·10⁻¹⁴ vs. recorte]
```

**Acumulação direta no espaço de cor.** Cada aresta e aparece com orientações opostas no contorno de L(e) e no de R(e). Por isso, basta **uma** passada pelas arestas, acumulando o vetor RGBA (c_{L(e)} − c_{R(e)}) multiplicado por α e por (γ − α), com a cor do exterior igual a 0. A soma prefixada da linha já devolve ĉ_p. Não é preciso uma passada por face.

#### 2.2.3 Energia por varredura esparsa, independente de W·H

Dentro de uma linha, entre duas colunas cruzadas consecutivas, a cor acumulada é **constante** (= c_run). Pré-calculando uma vez, por linha, as somas prefixadas S¹ = Σ I_p (4 canais) e S² = Σ ‖I_p‖², o erro de uma sequência de n pixels não cruzados sai em O(1):

```
E_run = n·‖c_run‖² − 2·c_run·S¹_run + S²_run
E_gen(P) = Σ_{p cruzados} w_p ‖ĉ_p − I_p‖² + Σ_runs E_run
```

Cada avaliação custa **O(P log P)**, para ordenar os pedaços por coluna dentro de cada linha, ou O(P) com balde por coluna. Esse custo não depende da área da imagem, o que importa porque a busca em linha do CG avalia a energia muitas vezes por iteração.

### 2.3 Gradiente analítico subpixel (`findPixelGradient`, `findGradient`)

**Derivada de forma (Hadamard/Reynolds).** Se a fronteira de f se move com campo de velocidade v, então

```
d a_p(f) = ∫_{∂f ∩ p} v · n̂ ds
```

Na poligonal, o ponto de parâmetro t em s_k se desloca (1−t)·δP_k + t·δP_{k+1}, e n̂ ds = n_k dt. Portanto:

```
∂a_p(f)/∂P_k = n_k · ∫_{T_k(p)} (1 − t) dt  +  n_{k−1} · ∫_{T_{k−1}(p)} t dt              [V: 6.3·10⁻⁹]

  ∫_{t_a}^{t_b} (1 − t) dt = (t_b − t_a) − ½(t_b² − t_a²)
  ∫_{t_a}^{t_b} t dt       = ½(t_b² − t_a²)
```

Aqui T_k(p) é a união dos intervalos de migalha de s_k dentro de p. O gradiente só é diferente de zero nos pixels cruzados pelos **dois** segmentos incidentes em P_k: é estritamente local.

**Prova de consistência.** Somando sobre todos os pixels, ∫₀¹(1−t)dt = ∫₀¹ t dt = ½, logo

```
Σ_p ∂a_p/∂P_k = ½(n_k + n_{k−1}) = ½( y_{k+1} − y_{k−1} ,  x_{k−1} − x_{k+1} ) = ∂A/∂P_k   [V: 2.2·10⁻¹⁶]
```

que é exatamente a derivada do shoelace.

**Gradiente da energia.** Para uma aresta e com faces (L, R), ∂ĉ_p/∂P_k = (c_L − c_R) ⊗ g_{p,k}, onde g_{p,k} é o vetor de derivada de forma acima. Então:

```
∂E_gen/∂P_k = Σ_{e ∋ k} Σ_{p ∈ migalhas(e, k)} 2 w_p (ĉ_p − I_p) · (c_{L(e)} − c_{R(e)}) · g^{(e)}_{p,k}      [V: 3.0·10⁻⁹]
```

- **Junções** (grau ≥ 3) somam a contribuição de cada aresta incidente, cada uma com o seu próprio (c_L − c_R). A mesma fórmula serve para todos os nós.
- **Custo:** O(P) por gradiente completo, na mesma passada das migalhas.
- **Regularidade:** a_p é C¹ em P exceto quando um segmento é colinear a uma linha da grade (conjunto de medida nula). Por isso usamos CG não linear **Polak–Ribière+** com reinício e busca em linha de **Wolfe forte**. O passo é limitado a ‖δP_k‖ ≤ 0.5 px (região de confiança), para que o fiscal anti-cruzamento continue válido entre avaliações.

### 2.4 Demais termos da energia de suavização

```
E^(φ) = λ_fid·E_fid  (ou E_gen, conforme measurement_types[φ])
      + λ_ang·E_ang + λ_sp^(φ)·E_sp + λ_air·E_air + λ_brd·E_brd
λ_sp^(φ) = 5.0 → 0.5 → 0.5;   λ_ang = prior_strengths = 10;   λ_air = 5
```

| Termo | Definição | Gradiente |
|---|---|---|
| Fidelidade com zona livre | E_fid = Σ_k max(0, ‖P_k − O_k‖ − r)² (C¹; r = 0 é a mola simples) | 2·max(0, d − r)·(P_k − O_k)/d |
| Suavidade angular (`prior_types = 3`) | E_ang = Σ_k θ_k², com θ_k = atan2(Δ_{k−1} × Δ_k, Δ_{k−1}·Δ_k) | com u = Δ_{k−1} e v = Δ_k: ∂θ/∂u = (u_y, −u_x)/‖u‖² e ∂θ/∂v = (−v_y, v_x)/‖v‖². Daí ∂θ_k/∂P_{k−1} = −∂θ/∂u, ∂θ_k/∂P_k = ∂θ/∂u − ∂θ/∂v, ∂θ_k/∂P_{k+1} = ∂θ/∂v [V: 3.7·10⁻⁹] |
| Espaçamento | E_sp = Σ_k s_k², s_k = (‖Δ_k‖ − ‖Δ_{k−1}‖) / (‖Δ_k‖ + ‖Δ_{k−1}‖) | ∂s/∂a = −2b/(a+b)², ∂s/∂b = +2a/(a+b)², com a = ‖Δ_{k−1}‖ e b = ‖Δ_k‖ (§1.25) |
| Pressão do ar (contornos com < 8 nós) | E_air = (A(P) − A₀)² | 2(A − A₀)·½(y_{k+1} − y_{k−1}, x_{k−1} − x_{k+1}) |
| Âncora de margem | E_brd = Σ_{k ∈ borda} dist⊥(P_k, moldura)² | normal à moldura |
| Canto perfurado | remove θ_k de E_ang nos nós com a flag `Corner` (fase 2) | — |
| Quarentena | nós em quarentena têm gradiente mascarado por 2 iterações | — |

O prior de suavidade "raiz quadrada" fica documentado e **desativado**. Interpreto-o como E = Σ √(θ_k² + ε); a forma exata deve ser confirmada. Seu gradiente θ/√(θ² + ε) tem curvatura 1/√ε perto de θ = 0, o que mal condiciona a busca em linha. É a instabilidade numérica relatada na especificação.

### 2.5 `BezierFitter`

**Modelo.** B(t; p) = b₀(t)P₀ + b₁(t)P₁ + b₂(t)P₂ + b₃(t)P₃, com bases de Bernstein b_k(t) = C(3,k)(1−t)^{3−k} t^k. P₀ e P₃ ficam fixos nas pontas do fragmento, e p = (P₁, P₂) ∈ ℝ⁴. Para os nós interiores Q₁..Q_{m−1}:

```
E(p, t) = ½ Σ_k ‖B(t_k; p) − Q_k‖²,     r_k = B(t_k; p) − Q_k
```

Se o fragmento tem ≤ 3 pontos, ele vira um segmento de reta (`SegmentKind::Line`).

#### 2.5.1 Inicialização e mínimos quadrados em p

A inicialização usa comprimento de corda: t_k = L_k / L_m. Com t fixo, E é quadrática em p, e as equações normais 2×2 são **compartilhadas** pelas coordenadas x e y:

```
⎡ Σb₁²   Σb₁b₂ ⎤ ⎡P₁⎤   ⎡ Σ b₁ R_k ⎤
⎣ Σb₁b₂  Σb₂²  ⎦ ⎣P₂⎦ = ⎣ Σ b₂ R_k ⎦ ,     R_k = Q_k − b₀(t_k)P₀ − b₃(t_k)P₃
```

São 2 lados direitos (x e y): solução inline por Cramer, ou `dgesv_` com NRHS = 2 no caminho de referência.

#### 2.5.2 Newton exato e conjunto em (p, t) [V: gradiente 2.2·10⁻¹⁰, Hessiana 1.1·10⁻⁸]

Com J_k = [b₁(t_k)·I₂ | b₂(t_k)·I₂] (2×4):

```
g_p      = Σ_k J_kᵀ r_k
g_{t,k}  = r_k · B'(t_k)
H_pp     = Σ_k J_kᵀ J_k                                            (4×4; = Gram ⊗ I₂)
h_k      = H_{p,t_k} = J_kᵀ B'(t_k) + [ b₁'(t_k)·r_k ; b₂'(t_k)·r_k ]    (4×1)
D_k      = H_{t_k,t_k} = ‖B'(t_k)‖² + r_k · B''(t_k)
```

A Hessiana tem estrutura em **seta**: um bloco denso 4×4, uma borda 4×(m−1) e uma diagonal. Pelo complemento de Schur, com amortecimento de Levenberg–Marquardt λ:

```
S   = H_pp + λI − Σ_k h_k h_kᵀ / (D_k + λ)                      (4×4, SPD → Cholesky: dpotrf_/dpotrs_ ou inline)
S δp = −g_p + Σ_k h_k g_{t,k} / (D_k + λ)
δt_k = −(g_{t,k} + h_kᵀ δp) / (D_k + λ)
```

- **Controle de λ:** passo aceito (E diminuiu) faz λ ← λ/10; passo rejeitado faz λ ← 10λ e o passo é recalculado.
- **Restrições em t:** t_k é projetado em [0, 1], e a monotonia t_k ≤ t_{k+1} é imposta por projeção.
- **Custo:** **O(m)** por iteração, contra O(m³) do Newton denso. É essa a Hessiana a que a especificação se refere.

#### 2.5.3 Fusão iterativa e deslizamento da quebra

```
ΔE(A, B) = E*(A ∪ B) − (E*(A) + E*(B))         (E* = energia após o ajuste)
funde se ΔE < τ_r,     τ_r = τ₀ · (τ_f / τ₀)^{r/(R−1)},   r = 0..R−1
τ₀ = 10⁻⁸;   τ_f = stat_thresh_final (0.2 nos perfis de AA);   R ≤ 20
→ fator por rodada (0.2/10⁻⁸)^{1/19} ≈ 2.42
```

Quando a fusão é recusada, a quebra entre A e B é deslocada em {−1, 0, +1} nó e fica a posição de menor E*(A) + E*(B). O ajuste da união reaproveita (*warm start*) os t das duas partes, reescalados para [0, 1]. O custo total é O(R · n · iterações de Newton), porque os fragmentos particionam os nós.

#### 2.5.4 Variante G¹ nas quebras lisas

Numa quebra que não é canto nem junção, estima-se a tangente τ no contorno suavizado (diferença central normalizada) e impõe-se

```
P₁ = P₀ + α·τ₀,   P₂ = P₃ − β·τ₃,   p = (α, β) ∈ ℝ²,   J_k = [ b₁(t_k)·τ₀ | −b₂(t_k)·τ₃ ]  (2×2)
```

A mesma maquinaria de Schur se aplica, agora com S de tamanho 2×2. Se α ≤ 0 ou β ≤ 0 (curva com laço ou cúspide), usa-se a heurística de Schneider α = β = ‖P₃ − P₀‖/3 e o fragmento é marcado para subdivisão.

### 2.6 Validação numérica

Script: [`tests/math/validate_math.py`](../tests/math/validate_math.py), em Python puro, determinístico (semente 7), com diferenças finitas centrais (h = 10⁻⁶).

| Verificação | Erro máximo |
|---|---|
| Cobertura por acumulação vs. recorte Sutherland–Hodgman | 1.3·10⁻¹⁴ |
| ∂a_p/∂P_k analítico vs. diferenças finitas | 6.3·10⁻⁹ |
| Σ_p ∂a_p/∂P_k vs. ∂A/∂P_k do shoelace | 2.2·10⁻¹⁶ |
| ∂E_gen/∂P_k (síntese RGBA com 2 faces) vs. diferenças finitas | 3.0·10⁻⁹ |
| Gradiente do ajuste de Bézier vs. diferenças finitas | 2.2·10⁻¹⁰ |
| Hessiana do ajuste de Bézier vs. diferenças finitas | 1.1·10⁻⁸ |
| Prior angular ∂E_ang/∂P_k vs. diferenças finitas | 3.7·10⁻⁹ |

Ajuste de um arco ruidoso de 26 pontos; energia E por iteração:

| Iteração | Newton + Schur + LM | Alternância LS ↔ reparametrização |
|---|---|---|
| 0 | 0.5493557838 | 0.5493557838 |
| 2 | 0.0392879662 | 0.2374701414 |
| 4 | 0.0352665920 | 0.1236193921 |
| 6 | 0.0348815399 | 0.0750600091 |
| 8 | **0.0348784066** | 0.0538701964 |

O Newton converge quadraticamente: entre as iterações 7 e 8, a energia muda só 10⁻⁸. A alternância ainda está 54 % acima do ótimo na iteração 8.

---

## 3. Estado da arte e posicionamento

| Abordagem | Representação | Fidelidade em logos com AA | Topologia (frestas) | Determinismo |
|---|---|---|---|---|
| Potrace (Selinger 2003) | bitmap binário por cor, camadas | média: ignora o AA | sobreposição em camadas | sim |
| Depixelizing Pixel Art (Kopf & Lischinski 2011) | grafo de similaridade + splines | só para pixel art | compartilhada | sim |
| DiffVG (Li et al. 2020) + LIVE (Ma et al. 2022) | caminhos sobrepostos, otimização por rasterizador diferenciável | alta, mas o gradiente vem de amostragem e prefiltragem | **sem** fronteira compartilhada: caminhos se sobrepõem | estocástico |
| StarVector (Rodriguez et al., 2023/2025) e LLMs geradores de SVG | código SVG gerado | semântica boa, sem exatidão de pixel | não garantida | estocástico |
| **cnx** (análise por síntese, Diebel 2008) | mapa planar por meia-aresta + Bézier por aresta | alta: gradiente **exato** e analítico da cobertura de área | **zero frestas por construção** (I1, I2) | bit a bit |

Para logos de cores chapadas com AA, a combinação "mapa planar + síntese de área exata" é superior aos métodos diferenciáveis genéricos. Estes otimizam caminhos sobrepostos com gradientes estimados e não garantem a partição do plano. Abordagens neurais entram melhor como **componentes**: classificador de perfil, priors de canto aprendidos, inicialização. Não servem como substitutas do ajuste geométrico.

---

## 4. Referências

- J. Diebel. *Bayesian Image Vectorization: the probabilistic inversion of vector image rasterization*. Tese de doutorado, Stanford University, 2008.
- P. J. Schneider. *An Algorithm for Automatically Fitting Digitized Curves*. Graphics Gems, 1990.
- J. Hoschek. *Intrinsic parametrization for approximation*. Computer Aided Geometric Design 5(1), 1988.
- W. Wang, H. Pottmann, Y. Liu. *Fitting B-spline curves to point clouds by curvature-based squared distance minimization*. ACM TOG 25(2), 2006.
- A. Rosenfeld. *Connectivity in digital pictures*. Journal of the ACM 17(1), 1970.
- S. Yokoi, J. Toriwaki, T. Fukumura. *An analysis of topological properties of digitized binary pictures using local features*. Computer Graphics and Image Processing 4(1), 1975.
- J. Amanatides, A. Woo. *A Fast Voxel Traversal Algorithm for Ray Tracing*. Eurographics, 1987.
- C. L. Lawson, R. J. Hanson. *Solving Least Squares Problems*. Prentice-Hall, 1974 (NNLS).
- J. Nocedal, S. J. Wright. *Numerical Optimization*, 2ª ed. Springer, 2006 (CG não linear, Wolfe, LM).
- R. Levien. *Inside the fastest font renderer in the world* (font-rs), 2016 (acumulação de área assinada).
- P. Selinger. *Potrace: a polygon-based tracing algorithm*, 2003.
- J. Kopf, D. Lischinski. *Depixelizing Pixel Art*. ACM TOG 30(4) (SIGGRAPH), 2011.
- T.-M. Li, M. Lukáč, M. Gharbi, J. Ragan-Kelley. *Differentiable Vector Graphics Rasterization for Editing and Learning*. ACM TOG 39(6) (SIGGRAPH Asia), 2020.
- X. Ma et al. *Towards Layer-wise Image Vectorization* (LIVE). CVPR, 2022.
- J. A. Rodriguez et al. *StarVector: Generating Scalable Vector Graphics Code from Images and Text*. arXiv:2312.11556.
