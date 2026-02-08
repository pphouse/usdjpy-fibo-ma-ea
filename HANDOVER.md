# プロジェクト引き継ぎドキュメント (v2)

## プロジェクト概要
MT5用EA開発 - フィボナッチ+MA戦略 & ゴゴジャン向け高勝率戦略

---

## 完了済みタスク

### 1. フィボナッチ+MA戦略（メイン）
- **戦略**: フィボナッチリトレースメント + 移動平均線 + ナンピン
- **バックテスト完了**: H4, H1, M30, M15, M5
- **ベスト結果**:
  - H4: +32,225 pips (2015-2025)
  - M5: +26,385 pips
- **データ**: Axiory実データ（USDJPY M1 2015-2025）

### 2. ゴゴジャン向け高勝率戦略リサーチ
- **目的**: ゴゴジャンで売れるEA開発
- **売れ筋要素**: 勝率80%+, 連勝記録, M5時間足, AI/ビジュアル
- **テスト済み戦略**:
  1. RSI Mean Reversion - 連勝156回記録
  2. Bollinger Bounce
  3. **Tokyo Range Breakout** - 勝率96.2%, PF 4.91 ★最有力
  4. High WR Scalping

### 3. 多通貨ペアデータ取得・展開
- **ダウンロード・展開完了** (Axiory M1 2015-2025):
  - USDJPY ✓ (`usd_jpy_M1/extracted/`)
  - EURJPY ✓ (`eur_jpy_M1/extracted/`)
  - GBPJPY ✓ (`gbp_jpy_M1/extracted/`)
  - AUDJPY ✓ (`aud_jpy_M1/extracted/`)
  - EURUSD ✓ (`eur_usd_M1/extracted/`)
  - GBPUSD ✓ (`gbp_usd_M1/extracted/`)
- 各通貨 134 CSVファイル (月別)

### 4. ベスト戦略の詳細分析・可視化 ★NEW
- **対象**: Tokyo Range Breakout (USDJPY TP=10/SL=50)
- **結果**:
  - 勝率: 96.2% (687W / 27L)
  - 合計pips: +5,305.8
  - PF: 4.91
  - 最大DD: -90.9 pips
  - 最大連勝: 56、最大連敗: 1
  - リカバリーファクター: 58.4
- **可視化**: `gogojungle_strategy/best_strategy_detailed.png`
  - 累積損益カーブ、ドローダウン、年別/月別パフォーマンス
  - エントリー時間分布、BUY/SELL別分析、東京レンジ幅分布、連勝分布

### 5. Tokyo Range Breakout EA (MQL5) 作成 ★NEW
- **ファイル**: `TokyoRangeBreakout_EA.mq5`
- **機能**:
  - 東京レンジ自動収集 (0-6 UTC)
  - ロンドン時間ブレイクアウト判定 (6-14 UTC)
  - レンジフィルター (min/max pips)
  - スプレッドフィルター
  - 自動ロット計算 (リスク%ベース)
  - チャート上にレンジライン/エントリー矢印描画
  - 1日1トレード制限
- **パラメータ最適化・フォワードテストは未実施**

### 6. アルゴリズム仕様書・検証用スクリプト ★NEW
- **仕様書**: `gogojungle_strategy/TOKYO_RANGE_BREAKOUT_ALGORITHM.md`
  - アルゴリズムの完全な説明、フローチャート、パラメータ一覧
  - 損益分岐勝率の計算 (83.8%)
- **スタンドアロン検証スクリプト**: `gogojungle_strategy/tokyo_range_breakout_standalone.py`
  - 他の人が独立して検証可能
  - データ読み込み→バックテスト→統計→可視化の一括実行

---

## 進行中タスク

### 東京レンジブレイクアウト - 6通貨ペア検証
- **状態**: Azure VMで62分実行後、マシン変更のため中断
- **スクリプト**: `gogojungle_strategy/multi_currency_tokyo_v2.py`
  - パスは Azure VM用に更新済み (`/home/azureuser/usdjpy-fibo-ma-ea/...`)
- **データ**: 6通貨全てダウンロード・展開済み（上記「完了済み#3」参照）
- **次のアクション**: 上位マシンで以下を実行

```bash
# 環境セットアップ
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib

# 6通貨ペアバックテスト実行 (CPU集約: 60分+見込み)
python3 gogojungle_strategy/multi_currency_tokyo_v2.py
```

- **処理内容**: 6通貨ペア × 4パラメータセット × 10年M1→M5データ
  - TP/SL組み合わせ: (10,50), (15,50), (10,30), (15,30)
  - 出力: `gogojungle_strategy/multi_currency_tokyo_v2.csv` + `.png`

---

## 未着手タスク

### 1. パラメータ最適化 (通貨ペア別)
- 多通貨検証結果をもとに各通貨ペアの最適パラメータを決定
- レンジ幅フィルターの調整 (通貨ペアのボラティリティに応じて)

### 2. MQL5 EA のフォワードテスト
- `TokyoRangeBreakout_EA.mq5` をMT5でコンパイル
- デモ口座でフォワードテスト
- 多通貨対応版の作成（パラメータを通貨ペア別にプリセット）

### 3. ゴゴジャン向けパッケージング
- 高勝率アピール（96%+）
- 連勝記録アピール（56連勝）
- ビジュアルダッシュボード追加
- 説明文・スクリーンショット作成
- バックテスト結果のレポート

---

## ファイル構成

```
usdjpy-fibo-ma-ea/
├── HANDOVER.md                              # ★この引き継ぎドキュメント
├── ALGORITHM_SPEC.md                        # Fibo+MA戦略仕様
│
├── FiboMA_EA.mq5                            # Fibo+MA EA
├── FiboMA_EA_Nanpin.mq5                     # ナンピン版EA
├── TokyoRangeBreakout_EA.mq5                # ★東京ブレイクアウトEA (NEW)
│
├── gogojungle_strategy/
│   ├── TOKYO_RANGE_BREAKOUT_ALGORITHM.md    # ★アルゴリズム仕様書 (NEW)
│   ├── tokyo_range_breakout_standalone.py   # ★スタンドアロン検証 (NEW)
│   ├── visualize_best_strategy.py           # ★ベスト戦略可視化 (NEW)
│   ├── best_strategy_detailed.png           # ★詳細分析9パネル (NEW)
│   ├── best_strategy_trades.csv             # ★714トレード履歴 (NEW)
│   ├── multi_currency_tokyo_v2.py           # 6通貨ペアテスト (パス更新済)
│   ├── high_winrate_research.py             # 高勝率戦略リサーチ
│   └── research_results.csv                 # 62パラメータ組み合わせ結果
│
├── usd_jpy_M1/extracted/                    # USDJPY M1データ (展開済)
├── eur_jpy_M1/extracted/                    # EURJPY M1データ (展開済)
├── gbp_jpy_M1/extracted/                    # GBPJPY M1データ (展開済)
├── aud_jpy_M1/extracted/                    # AUDJPY M1データ (展開済)
├── eur_usd_M1/extracted/                    # EURUSD M1データ (展開済)
├── gbp_usd_M1/extracted/                    # GBPUSD M1データ (展開済)
│
├── h4_real_backtest/                        # H4バックテスト
├── m5_real_backtest/                        # M5バックテスト
└── .venv/                                   # Python仮想環境
```

---

## 環境情報

- **現在のVM**: Azure (Ubuntu, Python 3.12.3)
- **venv**: `.venv/` に pandas, numpy, matplotlib インストール済
- **M1データ**: 全6通貨ペア展開済み (`.gitignore`で除外、サイズ大)
- **データ再取得**: Axiory (`https://www.axiory.com/jp/assets/download/historical/mt4_standard/{year}/{symbol}.zip`)

---

## Tokyo Range Breakout ベスト結果 (USDJPY)

| TP | SL | 勝率 | 連勝 | Net Pips | PF | MaxDD |
|----|----|----|------|----------|-----|-------|
| 10 | 50 | 96.2% | 56 | +5,306 | 4.91 | -90.9 |
| 15 | 50 | 94.2% | 41 | +5,894 | 4.78 | -106.8 |
| 10 | 30 | 93.4% | 52 | +3,874 | 4.35 | - |
| 20 | 50 | 91.1% | 27 | +5,419 | 3.99 | -142.1 |

損益分岐勝率: 83.8% (TP10/SL50の場合)

---

## 連絡先・リポジトリ
- GitHub: https://github.com/pphouse/usdjpy-fibo-ma-ea
