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
│     ├─ engine/      CoreEngine.h  Parameters.h  Fingerprint.h  TaskContext.h  [ParameterFiler]
│     ├─ preprocess/  [Preprocessor.h  SignalImage.h  EdgeImage.h  FlowField.h  AADetector.h]
│     ├─ classify/    [ImageFeatures.h  DecisionTree.h  ImageClassification.h]
│     ├─ palette/     [ColorHistogram.h  KMeans.h  PaletteFinder.h  WeightsFinder.h]
│     ├─ segment/     PixelMap.h  [QuadTree.h  PixelSegmenter.h  SuperPixelSegmenter.h
│     │                SubPixelSegmenter.h  CoOccurrenceMatrix.h  BipartiteMatcher.h
│     │                SegmentationTrainer.h]
│     ├─ vector/      VectorImage.h  BoundaryTracer.h  GenerativeModel.h  ContourSmoother.h
│     │               Optimizer.h  EnergyTerms.h  CrossingGuard.h   (a grade de segmentos mora em CrossingGuard)
│     ├─ fit/         BezierFitter.h
│     └─ export/      Exporter.h  [SvgWriter.h  EpsWriter.h  AiWriter.h
│                      CompatiblePdfWriter.h  DxfWriter.h]
├─ src/               (espelha include/; hoje: segment/PixelMap.cpp, QuadTree.cpp, PixelSegmenter.cpp,
│                      vector/VectorImage.cpp, BoundaryTracer.cpp, GenerativeModel.cpp, EnergyTerms.cpp,
│                      Optimizer.cpp, CrossingGuard.cpp, ContourSmoother.cpp; fit/BezierFitter.cpp;
│                      sla/BezierCurve.cpp)
├─ qt/                EngineController.{h,cpp}  QImageLoader.{h,cpp}
├─ tests/
│  ├─ math/           validate_math.py   (validação numérica das fórmulas)
│  ├─ unit/           test_harness.h, test_support.h, header_smoke_test, pixel_map_test,
│  │                  pixel_segmenter_test, vector_image_test, boundary_tracer_test,
│  │                  generative_model_test, energy_terms_test, optimizer_test, crossing_guard_test,
│  │                  contour_smoother_test, bezier_curve_test, bezier_fitter_test
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
| Espaçamento | Σ (‖Δ_k‖ − ‖Δ_{k−1}‖)² | pesos 5,0 → 0,5 → 0,5 |
| Pressão do ar | (A − A₀)² por contorno com < 8 nós | A₀ = área na escadinha original; contorno = laço de uma face (externo ou furo); o exterior não tem |
| Âncora de margem | (coordenada − moldura)² nos nós da moldura não fixos | ver a trava exata, abaixo |

Todos os gradientes são analíticos e conferidos contra diferenças finitas centradas (`energy_terms_test`), inclusive com cantos perfurados e arestas fechadas. O gradiente do prior angular segue a tabela do §2.4. Três observações sobre a energia vieram da execução:

1. **Trava exata da moldura.** Só a mola de margem não basta: com ela os nós da moldura derivam alguns milésimos para dentro, a moldura deixa de medir W×H e a invariante I4 (Σ áreas = W·H) falha. Por isso, além da mola, o gradiente perpendicular dos nós da moldura é zerado e a coordenada perpendicular é fixada exatamente no início da fase. Eles deslizam ao longo da moldura e nunca saem dela (`borderHardConstraint`, ligado por padrão). A mola de margem continua na energia e vale zero nesse regime.
2. **O espaçamento é necessário, não só estético.** Sem ele, nós livres sobre um trecho reto deslizam e se amontoam. O prior angular depende de 1/‖Δ‖², fica rígido demais perto da coincidência, e a busca em linha falha. Num teste, dois nós chegaram a 1e-6 px um do outro. O fiscal trata esse colapso como um cruzamento (ver abaixo).
3. **O prior angular não desloca polígonos com poucos nós.** Em qualquer polígono convexo Σ θ = 2π, então um retângulo (4 giros iguais) já é o mínimo de Σ θ². A perfuração de cantos só faz diferença quando há curvas ou lados em ângulos diferentes; é por isso que o teste usa um triângulo com hipotenusa em escada.

**Otimizador** (`Optimizer.h`): gradiente conjugado não linear **Polak–Ribière+** (β = max(0, g'·(g'−g)/‖g‖²)) com reinício de Powell (|g'·g| ≥ 0,2‖g'‖²) e reinício em d = −g se d não for de descida. A busca em linha é a de Nocedal & Wright (alg. 3.5 e 3.6) com **Wolfe forte** (c₁ = 1e-4, c₂ = 0,1), interpolação cúbica protegida e bisseção de reserva. A **região de confiança** limita o deslocamento de cada nó a `maxNodeStepPx` (0,5) por iteração; no teto basta a condição de Armijo. Funções indefinidas fora de um domínio (NaN) são tratadas como "passo grande demais". O passo inicial é 1/‖g‖ na primeira iteração e, depois, α_prev·(g·d)_anterior/(g·d) (N&W 3.60).

**Fiscal anti-cruzamento** (`CrossingGuard.h`): interseção própria entre segmentos sem nó em comum, com uma **grade uniforme de segmentos** no lugar da kd-tree da ANN (que indexa pontos, não segmentos). O resultado é idêntico ao da força bruta O(S²), verificado em cenas perturbadas com dezenas de cruzamentos. Cada par é testado uma só vez, na célula do canto inferior-esquerdo da interseção das caixas. O fiscal também detecta **segmentos colapsados** (< `minSegmentPx`, 0,05 px).

**Fluxo de uma fase.** Constrói a topologia estática, perfura os cantos (só na fase `punctureOnPhase`), cria o objetivo sobre os nós livres e itera. Depois de cada iteração aceita pelo CG, o fiscal confere o resultado. Havendo cruzamento ou colapso: desfaz a iteração, põe os nós envolvidos em quarentena por `quarantineRounds` iterações e recomeça o CG do ponto anterior (a energia continua não crescente). Quando a quarentena expira, o conjunto de nós livres muda e o CG recomeça de novo. Ao fim da fase a quarentena é limpa. Entradas que já se cruzam ou já têm segmentos colapsados são recusadas (`std::invalid_argument`).

**Canto perfurado (`isProbablyPuncturedNode`).** O vetor de 21 medidas da especificação não está definido, então implementei uma regra de duas medidas que usa exatamente os dois conceitos citados ("ângulo local e retidão dos trechos adjacentes"). Cada braço segue a poligonal por até `punctureArmPx` (4 px) de comprimento de arco. O nó é canto se |giro entre as cordas dos braços| ≥ `punctureCornerThresh` (0,8 rad) e **nenhum nó intermediário** de cada braço se afasta da corda em mais de `punctureFlatnessThresh` (0,3 px). Limiar ≤ 0 desativa. A regra deve ser recalibrada no conjunto de referência.

**Resultados medidos** (MSVC Release):
- Disco de raio 10,6 com antialiasing (supersampling 16×16), fidelidade generativa: o erro radial médio dos nós cai de 0,425 px na escadinha para 0,015 px, e E_gen de 14,7 para 0,004 (≈ 90 ms).
- Reta inclinada: o desvio máximo dos nós interiores cai de 0,657 px para 0,001 px.
- Ângulo reto de um triângulo (generativo): o nó fica a 0,085 px do vértice verdadeiro com a perfuração e a 0,861 px sem ela.
- Pressão do ar: numa região em L (6 nós), a área fica em 180,00 com a pressão e vai a 182,75 sem ela.
- Duas barras escuras com alvo de barra única: o fiscal desfez 1 iteração por fase e o resultado não tem cruzamentos.

**Limitações conhecidas.** Todos os pesos são provisórios (a calibrar): com a mola padrão (λ_fid = 1) um triângulo encolhe vários pixels sob o prior angular de 10, e os testes que dependem disso fixam os pesos explicitamente. O CG pode terminar por `LineSearchFailed` perto do arredondamento (gradiente ≈ 1e-3), o que é reportado em `PhaseStats::status`. O prior de suavidade "raiz quadrada" não foi implementado (`priorType` ≠ 3 lança `std::invalid_argument`).

**Testes** (`energy_terms_test` 10, `optimizer_test` 9, `crossing_guard_test` 7, `contour_smoother_test` 9):
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
| Espaçamento | E_sp = Σ_k (‖Δ_k‖ − ‖Δ_{k−1}‖)² | regra da cadeia sobre ‖Δ‖ |
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
