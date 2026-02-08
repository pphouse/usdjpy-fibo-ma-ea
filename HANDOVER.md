# プロジェクト引き継ぎドキュメント

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

### 3. 多通貨ペアデータ取得
- **ダウンロード完了** (Axiory M1 2015-2025):
  - USDJPY ✓
  - EURJPY ✓
  - GBPJPY ✓
  - AUDJPY ✓
  - EURUSD ✓
  - GBPUSD ✓

---

## 進行中タスク

### 東京レンジブレイクアウト - 6通貨ペア検証
- **状態**: MacBook Airでは処理時間が長すぎるため中断
- **スクリプト**: `gogojungle_strategy/multi_currency_tokyo_v2.py`
- **次のアクション**: Azure VMで実行

---

## 未着手タスク

### 1. 多通貨ペア検証完了
Azure VMで `multi_currency_tokyo_v2.py` を実行

### 2. Tokyo Range Breakout EA作成
- MQL5コード作成
- パラメータ最適化
- フォワードテスト

### 3. ゴゴジャン向けパッケージング
- 高勝率アピール（96%+）
- 連勝記録アピール
- ビジュアルダッシュボード追加
- 説明文・スクリーンショット作成

---

## ファイル構成

```
usdjpy-fibo-ma-ea/
├── gogojungle_strategy/
│   ├── high_winrate_research.py      # 高勝率戦略リサーチ
│   ├── high_winrate_research.png     # 結果可視化
│   ├── multi_currency_tokyo_v2.py    # 6通貨ペアテスト★実行待ち
│   ├── multi_currency_tokyo_breakout.py
│   └── research_results.csv
├── h4_real_backtest/                 # H4バックテスト
├── m5_real_backtest/                 # M5バックテスト
├── FiboMA_EA.mq5                     # メインEA
├── FiboMA_EA_Nanpin.mq5              # ナンピン版EA
└── ...
```

---

## Azure VMセットアップ

### 推奨スペック
- **VM**: F4s_v2 (4 vCPU, 8GB RAM)
- **OS**: Ubuntu 22.04

### セットアップコマンド
```bash
# 1. システム更新
sudo apt update && sudo apt install -y python3-pip unzip wget

# 2. Python依存関係
pip3 install pandas numpy matplotlib

# 3. リポジトリクローン
git clone https://github.com/pphouse/usdjpy-fibo-ma-ea.git
cd usdjpy-fibo-ma-ea

# 4. 通貨データダウンロード
mkdir -p eur_jpy_M1/extracted gbp_jpy_M1/extracted aud_jpy_M1/extracted eur_usd_M1/extracted gbp_usd_M1/extracted

# Axioryからダウンロード (2015-2025)
for year in {2015..2025}; do
  for symbol in EURJPY GBPJPY AUDJPY EURUSD GBPUSD; do
    wget -q "https://www.axiory.com/jp/assets/download/historical/mt4_standard/${year}/${symbol}.zip" -O ${symbol}_${year}.zip
    # 各フォルダに展開
  done
done

# 5. テスト実行
python3 gogojungle_strategy/multi_currency_tokyo_v2.py
```

---

## Tokyo Range Breakout 戦略詳細

### ロジック
1. **東京時間（9-15時 JST / 0-6時 UTC）**: レンジ形成
2. **ロンドン時間（15-23時 JST / 6-14時 UTC）**: ブレイクアウトでエントリー
3. **条件**:
   - レンジ幅: 10-50 pips
   - ブレイクアウトバッファ: 5 pips
   - 1日1トレード

### USDJPYでの結果
| TP | SL | 勝率 | 連勝 | Net Pips | PF |
|----|----|----|------|----------|-----|
| 10 | 50 | 96.2% | 56 | +5,306 | 4.91 |
| 15 | 50 | 94.2% | 41 | +5,894 | 4.78 |
| 10 | 30 | 93.4% | 52 | +3,874 | 4.35 |

---

## 注意事項

1. **M1データ**: .gitignoreでリポジトリから除外（サイズ大）
2. **データ取得**: Axioryから直接ダウンロード必要
3. **処理時間**: 6通貨ペア x 10年データ = Azure推奨

---

## 連絡先・リポジトリ
- GitHub: https://github.com/pphouse/usdjpy-fibo-ma-ea
