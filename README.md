# CUDA-Ising 骨質疏鬆基因/標的探勘 — QUBO 特徵選擇骨架

把「從公共表現量資料找出與骨質疏鬆相關、彼此不冗餘的一組基因」寫成
QUBO / Ising 最佳化問題,在 DGX-B200 上用內建 GPU 求解器求解。

## 在 DGX-B200 上安裝(兩條命令)

```bash
# 0) 把壓縮檔傳上 DGX 後解壓
unzip cuda-ising-osteo.zip && cd cuda-ising-osteo

# 1) 安裝(建立 venv、裝相依)。加 --with-torch 會一併裝 B200 用的 PyTorch cu128
bash install.sh --with-torch
source .venv/bin/activate

# 2) 冒煙測試(離線、免 GPU,用合成資料跑完整條 pipeline)
bash smoke_test.sh          # 印出「✅ 冒煙測試通過」就代表裝好了
```

B200 是 Blackwell(sm_100),**需要 CUDA 12.8 + PyTorch cu128 wheel**(較舊的 cu124
不能用)。兩個安裝路徑二選一:

- **NGC 容器(DGX 上最穩)**:直接用 `nvcr.io/nvidia/pytorch:<YY.MM>-py3`,裡面
  CUDA 與 PyTorch 已對好 Blackwell;進容器後跑 `bash install.sh`(不用 `--with-torch`,
  torch 已內建)。
- **原生 venv**:`bash install.sh --with-torch`,它會執行
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`。

裝完可驗證 GPU:`python -c "import torch; print(torch.cuda.is_available())"` 應印 `True`。
沒有 GPU / 沒裝 torch 也能跑——`solve_gpu.py` 會自動退回 NumPy(只是慢、規模小)。

## 目標函數

以 `x_i ∈ {0,1}` 表示是否選中基因 i:

```
E(x) = -alpha * Σ_i R_i x_i          # 相關性:越相關越想選
       + beta  * Σ_{i<j} C_ij x_i x_j # 冗餘:兩基因太像就懲罰
       + lam   * (Σ_i x_i - k)^2      # 基數:大約選 k 個
```

`R_i` = 基因對表型(low vs high BMD)的辨別力;`C_ij` = 基因間表現量相關。
展開後轉成 Ising `E(s)=Σ h_i s_i + Σ J_ij s_i s_j`(推導見 `step03`)。

## 流程

```
step01_download_geo.py   # 取得表現量矩陣 + 標籤 -> expression.npz
step02_features.py       # 算 R_i、C_ij         -> features.npz
step03_build_qubo.py     # 組 QUBO->Ising,匯出  -> ising_coupling.txt / .npz / _bqm.json
step04_param_sweep.py    # 掃 alpha/beta/lam/k   -> sweep/<tag>/...
solve_gpu.py             # ★ GPU 求解器(DGX-B200)-> states.txt   ← 這就是「CUDA-Ising 求解器」
step05_postprocess.py    # 讀解 + 彙整候選基因   -> candidates.csv
step06_enrich_and_novelty.py  # 富集 + 已知/新穎標註 -> candidates_annotated.csv, enrichment.csv
step07_baselines.py           # 經典基線對照        -> baseline_comparison.csv, method_gene_sets.json
step08_replicate.py           # 外部世代複製        -> replication_panel.csv, replication_summary.txt
```

## 資料來源:`data.loader`

`config.yaml` 用 `data.loader` 切換 step01 的三條路:

| loader | 說明 | 需網路 |
|--------|------|--------|
| `demo` | 合成資料,帶已知「真基因」(命名 `TRUE_xx`),離線驗證整條流程 | 否 |
| `gse56815` | **已寫死欄位解析的真實 osteoporosis series**(預設) | 是 |
| `generic` | 泛用 GEO 解析,欄位命名不保證正確,通常要自己微調 | 是 |

### 真實資料:GSE56815(已寫死,開箱即用)

Zhou et al. 循環單核球研究,GPL96 (Affymetrix HG-U133A),80 例:
40 high BMD + 40 low BMD(各含 20 pre + 20 postmenopausal 婦女)。
**病例(y=1)= low BMD;對照(y=0)= high BMD。**

`step01` 的 `load_gse56815()` 已對照 GEO 逐筆核對欄位,直接寫死解析:

```
characteristics_ch1:
  "gender: Female"
  "bone mineral density: high BMD"  /  "low BMD"   <- 分組依據
  "state: postmenopausal"           /  "premenopausal"
  "cell type: monocytes"
```

流程:解析 `bone mineral density` 定病例/對照 → 可選 `restrict_state` 只留某停經狀態
→ log2 轉換 → 用 GPL96 註解把 probe 併成 gene symbol(同 symbol 多探針取平均,
`A /// B` 取第一個)。

```bash
pip install -r requirements.txt        # 真實資料需要 GEOparse
cd pipeline
python step01_download_geo.py ../config.yaml   # 首次會從 NCBI 下載 GSE56815(~數十 MB)
python step02_features.py    ../config.yaml
python step03_build_qubo.py  ../config.yaml
python step04_param_sweep.py ../config.yaml    # 選用:掃參數
python step05_postprocess.py ../config.yaml    # 用內建 SA 當基線;正式改餵 CUDA-Ising 解
```

### 離線先驗證流程(不需網路)

把 `config.yaml` 的 `data.loader` 改成 `demo` 再跑同樣五步即可;
合成資料裡的 `TRUE_xx` 應該會排在候選榜前段。

### 換成其他 series

最穩的做法是「照抄 `load_gse56815` 再改」:先用 GEOparse 看
`gse.gsms['GSM...'].metadata['characteristics_ch1']` 印出實際字串,
把分組欄位的 key 與 high/low(或 case/control)判斷改掉,平台註解欄位換成該
GPL 的 symbol 欄名即可。`generic` loader 只適合欄位剛好對得上關鍵字時的快速嘗試。

## 與你的 CUDA-Ising 對接

`step03` 會輸出三種格式(在 `config.yaml` 的 `export.formats` 選):

- `coo_txt`：第一行 `N M`,其後 `i j value`;`i==j` 為場 `h_i`,`i<j` 為耦合 `J_ij`。
- `npz`：`h`(向量)、`J`(上三角矩陣)、`genes`、`offset`。
- `dimod_json`：D-Wave `dimod` BQM 的 JSON(OpenJij / dwave-neal 可直接讀)。

## 在 DGX-B200 上求解:`solve_gpu.py`(內建 GPU 求解器)

你不需要另外找 CUDA-Ising 求解器 —— `solve_gpu.py` 就是。它針對本問題(**任意稠密
耦合圖**,不是 2D 晶格)寫的 GPU 平行回火(parallel tempering):一次在 GPU 上跑
上千條 replica,正好吃滿 B200 的平行度。有 PyTorch+CUDA 就自動走 GPU,沒有就退回
NumPy(演算法相同,方便小規模驗證)。能量慣例與 step03 完全一致,**不必手動改符號**。

DGX 上執行:

```bash
# 1) 裝 PyTorch(依 DGX 的 CUDA 版本挑對應 wheel;B200 需較新的 CUDA runtime)
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"  # 應印 True 與 B200

# 2) 求解(step03 產出的 ising.npz -> states.txt)
python solve_gpu.py --ising outputs/ising.npz --out outputs/states.txt \
    --n-temp 16 --chains 256 --sweeps 2000 --device cuda

# 3) 把解交回 pipeline
python step05_postprocess.py ../config.yaml --states outputs/states.txt
```

參數:`--n-temp × --chains = replica 總數`(B200 記憶體大,可開到數千);
`--sweeps` 是每條 replica 的掃描次數;`--tmin/--tmax` 是溫度階梯範圍;
`--swap-interval` 是幾個 sweep 做一次相鄰溫度交換。求解器會輸出每條 replica
造訪過的最低能量組態(spin `+1` = 該基因被選),step05 再依能量取低能量群、
統計每個基因的被選頻率。

### 想用「別的」CUDA-Ising 求解器?

若你之後裝了特定的 GPU Ising/QUBO 套件(OpenJij、sqaod、D-Wave 生態、或自寫
CUDA 二進位),step03 已同時輸出 `ising_coupling.txt`(純文字 COO)與
`ising_bqm.json`(dimod BQM)可餵給它;只要把它的輸出整成「每列一個 ±1 解」的
狀態檔,一樣用 `step05 --states` 接回。此時才需要注意下面的能量慣例對齊。

### ⚠ 慣例對齊(只有用外部求解器才需要)

本骨架輸出採 **dimod / 最小化** 慣例:`E = Σ h s + Σ J s s`,求能量最小。
很多物理版 CUDA-Ising 用 **`H = -Σ J s s - Σ h s`** 並找基態。
若你的求解器是後者,把 `h`、`J` **各自取負號** 再餵入,結果才一致。
求解後把每個自旋的 `+1` 視為「選中該基因」(若你用相反 spin 意義,對調即可)。

## step06:通路富集 + 已知/新穎標註

```bash
python step06_enrich_and_novelty.py ../config.yaml
```

做兩件事:

1. **新穎性(Open Targets)**:查 `novelty.efo_id`(預設 `EFO_0003882` = osteoporosis,
   已核對 EBI OLS / Open Targets)的關聯 target 分數,分數 ≥ `ot_score_threshold`
   視為「已知」,其餘標「新穎」。**連不到 Open Targets 時自動退回**
   `data/known_osteoporosis_genes.txt`(策展自 GEFOS / Morris 2019 / Richards 2008 等
   的核心 BMD 基因,約 56 個)。OT 查詢只用標準庫 `urllib`,無需額外套件。
2. **通路富集**:對 `select_freq` 前 `top_candidates` 個候選做富集。
   `method: auto` 先試線上 Enrichr(需 `gseapy` + 網路),失敗改用本地超幾何檢定
   (需在 `enrichment.gmt_path` 給一份 GMT,可從 Enrichr/MSigDB 下載;離線也能算,
   背景宇宙取 `features.npz` 的所有基因)。也可強制 `enrichr` / `local` / `off`。

輸出:
- `candidates_annotated.csv`:`gene, select_freq, R, ot_score, status(known/novel), known_source`
- `enrichment.csv`:`term, library, p_value, fdr, overlap, genes`(FDR 為 BH 校正)

> 這步是 pipeline 之外的下游詮釋:把候選排序、標出哪些已知/富集在哪些通路,
> 幫你把驗證資源集中在高 `select_freq` 的「新穎」候選。真正的新標的仍需濕實驗驗證。

## step07:經典基線對照(最能說服人的一步)

```bash
python step07_baselines.py ../config.yaml
```

在同一份 `X, y` 上,用同樣的基因數 `k`,跑五種經典/對照方法並與 Ising 比:
LASSO(L1-logistic)、mRMR(自實作,免額外套件)、RandomForest importance、
Univariate(step02 的 R 排序)、Random(隨機 k 個,下限對照)。

三個看點:

- **預測力**:對每個方法選出的基因組做 5-fold 交叉驗證分類 AUC —— 同樣 k 個基因,
  誰組成的 panel 更能區分 high/low BMD。informed 方法都應大幅贏過 Random。
- **重疊度**:各方法與 Ising 的 Jaccard 與共享基因數 —— 看一致性。
- **Ising 獨有**:只有 Ising 選到、其他方法都沒選到的基因 —— 潛在的新發現,
  正是「Ising 多找到了什麼」的直接證據。

輸出 `baseline_comparison.csv`(每方法:n、cv_auc、jaccard_vs_ising、n_shared)與
`method_gene_sets.json`(各方法基因 + `_Ising_unique`)。需要 `scikit-learn`。

> ⚠ 評估方式:各方法(含 Ising)都在全資料上選基因,再用 CV 評估「該基因組的預測力」,
> 回答的是「給定這組 panel 有多強」。若要無偏估計「選擇流程」本身的泛化力,應把選擇
> 也包進 CV 折內(nested CV)——Ising 因求解較貴通常另外處理,可先用本步的固定 panel
> 比較,再對最終候選做獨立世代(external cohort)複製。

## step08:外部世代複製(把結果從「有趣」推到「可信」)

```bash
python step08_replicate.py ../config.yaml
```

拿 discovery(GSE56815 / GPL96)的候選 panel,到一份**獨立、不同平台**的世代
GSE56814(GPL5175 Affymetrix Exon 陣列,73 例:42 high + 31 low BMD,同一實驗室
同樣 high/low BMD 單核球表型)上檢驗。換平台複製更能排除技術性假訊號。
兩份資料的 gene symbol 對齊方式不同(GPL96 用 `Gene Symbol` 欄,GPL5175 解析
`gene_assignment` 欄),step01 的 `load_series` 已寫死兩者;候選以 **gene symbol** 配對。

三個檢驗:

- **(A) 跨世代轉移 AUC**:只用候選基因,在 discovery 訓練 logistic、到 external 測
  AUC(並做反向)。能跨平台轉移 = 訊號穩健、非過度擬合。
- **(B) 每基因方向一致性**:兩世代各算 low vs high BMD 的帶方向效應(`2*(AUC-0.5)`),
  看方向是否相同(concordance)、效應量是否相關(Spearman rho)。
- **(C) external 自身 5-fold CV AUC**:panel 在全新資料裡還準不準。

輸出 `replication_panel.csv`(每個候選兩世代的效應與方向是否一致)與
`replication_summary.txt`(三項檢驗的 JSON 摘要)。需要網路 + GEOparse 下載 GSE56814。

> 讀法:方向一致性高(如 >80%)、Spearman 顯著正相關、跨世代 AUC 仍明顯高於 0.5,
> 才算「複製成功」。方向一致性接近 50%、轉移 AUC 掉到 ~0.5,代表候選多半是
> discovery 世代的過度擬合,應剔除。

## 再下一步(尚未腳本化,建議你接手)

比對 MGI 小鼠骨表型與 GTEx eQTL 共定位、查可成藥性(DGIdb/ChEMBL);
若要更嚴謹,可把上述複製擴充到第三、四份 BMD series 做 meta 分析。

## 注意

計算結果為假說,需濕實驗驗證。`lam` 權重是最大誤差來源,請務必用 `step04` 掃描。
