"""
Tokyo Range Breakout 戦略 - スタンドアロン検証スクリプト

使い方:
  1. Axiory から USDJPY M1 データをダウンロード
     https://www.axiory.com/jp/trading-tools/historical-data
  2. ZIP を展開して extracted/ フォルダに配置
  3. DATA_PATH を環境に合わせて変更
  4. 実行: python tokyo_range_breakout_standalone.py

必要パッケージ:
  pip install pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False

# ====================================================================
# 設定 - 環境に合わせて変更してください
# ====================================================================
DATA_PATH = Path("/home/azureuser/usdjpy-fibo-ma-ea/usd_jpy_M1/extracted")
SYMBOL = "USDJPY"
OUTPUT_DIR = Path("/home/azureuser/usdjpy-fibo-ma-ea/gogojungle_strategy")

# ====================================================================
# 戦略パラメータ
# ====================================================================
PARAMS = {
    'tokyo_start_hour': 0,    # UTC (= JST 9時)
    'tokyo_end_hour': 6,      # UTC (= JST 15時)
    'london_start_hour': 6,   # UTC (= JST 15時)
    'london_end_hour': 14,    # UTC (= JST 23時)
    'min_range_pips': 10,     # 最小レンジ幅
    'max_range_pips': 50,     # 最大レンジ幅
    'buffer_pips': 5,         # ブレイクアウトバッファ
    'tp_pips': 10,            # テイクプロフィット
    'sl_pips': 50,            # ストップロス
    'spread_pips': 0.3,       # スプレッド
    'pip_value': 0.01,        # USDJPY の 1pip = 0.01円
}


def load_m1_data(data_path: Path, symbol: str) -> pd.DataFrame:
    """
    M1データを読み込む

    Axiory の CSV 形式:
      ヘッダーなし、カラム: date, time, open, high, low, close, volume
      date 形式: 2015.01.02
      time 形式: 00:00
    """
    all_files = []

    # サブフォルダ内のファイル (例: extracted/USDJPY/USDJPY_2015_01.csv)
    symbol_folder = data_path / symbol
    if symbol_folder.exists():
        for f in symbol_folder.glob(f"{symbol}_20*_*.csv"):
            if "_all" not in f.name:
                all_files.append(f)

    # 直下のファイル (例: extracted/USDJPY_2019_01.csv)
    for f in data_path.glob(f"{symbol}_20*_*.csv"):
        if "_all" not in f.name:
            all_files.append(f)

    if not all_files:
        raise FileNotFoundError(
            f"データが見つかりません: {data_path}\n"
            f"Axiory からダウンロードして {data_path} に展開してください"
        )

    print(f"  CSV ファイル数: {len(all_files)}")

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(
                f, header=None,
                names=['date', 'time', 'open', 'high', 'low', 'close', 'volume']
            )
            dfs.append(df)
        except Exception as e:
            print(f"  警告: {f.name} の読み込みに失敗: {e}")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['datetime'] = pd.to_datetime(
        df_all['date'] + ' ' + df_all['time'],
        format='%Y.%m.%d %H:%M'
    )
    df_all = df_all.sort_values('datetime').reset_index(drop=True)
    df_all = df_all.set_index('datetime')
    df_all = df_all[~df_all.index.duplicated(keep='first')]

    return df_all[['open', 'high', 'low', 'close', 'volume']]


def resample_to_m5(df_m1: pd.DataFrame) -> pd.DataFrame:
    """M1 データを M5 (5分足) にリサンプリング"""
    df_m5 = df_m1.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return df_m5


def run_backtest(df: pd.DataFrame, params: dict) -> list:
    """
    東京レンジブレイクアウト バックテスト

    アルゴリズム:
    1. 東京時間 (0-6 UTC) の High/Low からレンジを定義
    2. レンジ幅が min_range ~ max_range の日のみ対象
    3. ロンドン時間 (6-14 UTC) にレンジ + バッファをブレイクしたらエントリー
    4. TP/SL で決済、ロンドン時間内に決済しない場合はノーカウント
    5. 1日1トレードのみ

    Returns:
        list of dict: 各トレードの詳細情報
    """
    pip = params['pip_value']
    min_range = params['min_range_pips'] * pip
    max_range = params['max_range_pips'] * pip
    buffer = params['buffer_pips'] * pip
    tp_pips = params['tp_pips']
    sl_pips = params['sl_pips']
    spread = params['spread_pips'] * pip
    tp_dist = tp_pips * pip
    sl_dist = sl_pips * pip

    tokyo_start = params['tokyo_start_hour']
    tokyo_end = params['tokyo_end_hour']
    london_start = params['london_start_hour']
    london_end = params['london_end_hour']

    df['hour'] = df.index.hour
    trades = []

    dates = df.index.date
    unique_dates = pd.unique(dates)

    # 最初100日と最後10日を除外 (ウォームアップ / 不完全データ)
    for date in unique_dates[100:-10]:
        day_data = df[df.index.date == date]

        # ---- ステップ1: 東京レンジの定義 ----
        tokyo_data = day_data[
            (day_data['hour'] >= tokyo_start) & (day_data['hour'] < tokyo_end)
        ]

        if len(tokyo_data) < 10:
            continue

        tokyo_high = tokyo_data['high'].max()
        tokyo_low = tokyo_data['low'].min()
        tokyo_range = tokyo_high - tokyo_low

        # ---- レンジフィルター ----
        if tokyo_range < min_range or tokyo_range > max_range:
            continue

        # ---- ステップ2: ロンドン時間でブレイクアウト判定 ----
        london_data = day_data[
            (day_data['hour'] >= london_start) & (day_data['hour'] < london_end)
        ]

        entry_done = False
        for idx in range(len(london_data)):
            if entry_done:
                break

            row = london_data.iloc[idx]
            close = row['close']

            # ブレイクアウト判定
            direction = None
            if close > tokyo_high + buffer:
                direction = 'BUY'
                entry_price = close
            elif close < tokyo_low - buffer:
                direction = 'SELL'
                entry_price = close

            if direction is None:
                continue

            # ---- ステップ3: 決済チェック ----
            remaining = (
                london_data.iloc[idx + 1:]
                if idx + 1 < len(london_data)
                else pd.DataFrame()
            )
            entry_time = london_data.index[idx]

            for i in range(len(remaining)):
                bar_high = remaining.iloc[i]['high']
                bar_low = remaining.iloc[i]['low']
                exit_time = remaining.index[i]

                if direction == 'BUY':
                    # TP チェック (High が TP 到達)
                    if bar_high >= entry_price + tp_dist:
                        trades.append({
                            'date': str(date),
                            'entry_time': str(entry_time),
                            'exit_time': str(exit_time),
                            'direction': direction,
                            'entry_price': round(entry_price, 5),
                            'tokyo_high': round(tokyo_high, 5),
                            'tokyo_low': round(tokyo_low, 5),
                            'tokyo_range_pips': round(tokyo_range / pip, 1),
                            'result': 'WIN',
                            'pips': round(tp_pips - spread / pip, 1),
                            'entry_hour': entry_time.hour,
                        })
                        entry_done = True
                        break

                    # SL チェック (Low が SL 到達)
                    if bar_low <= entry_price - sl_dist:
                        trades.append({
                            'date': str(date),
                            'entry_time': str(entry_time),
                            'exit_time': str(exit_time),
                            'direction': direction,
                            'entry_price': round(entry_price, 5),
                            'tokyo_high': round(tokyo_high, 5),
                            'tokyo_low': round(tokyo_low, 5),
                            'tokyo_range_pips': round(tokyo_range / pip, 1),
                            'result': 'LOSS',
                            'pips': round(-(sl_pips + spread / pip), 1),
                            'entry_hour': entry_time.hour,
                        })
                        entry_done = True
                        break

                else:  # SELL
                    # TP チェック (Low が TP 到達)
                    if bar_low <= entry_price - tp_dist:
                        trades.append({
                            'date': str(date),
                            'entry_time': str(entry_time),
                            'exit_time': str(exit_time),
                            'direction': direction,
                            'entry_price': round(entry_price, 5),
                            'tokyo_high': round(tokyo_high, 5),
                            'tokyo_low': round(tokyo_low, 5),
                            'tokyo_range_pips': round(tokyo_range / pip, 1),
                            'result': 'WIN',
                            'pips': round(tp_pips - spread / pip, 1),
                            'entry_hour': entry_time.hour,
                        })
                        entry_done = True
                        break

                    # SL チェック (High が SL 到達)
                    if bar_high >= entry_price + sl_dist:
                        trades.append({
                            'date': str(date),
                            'entry_time': str(entry_time),
                            'exit_time': str(exit_time),
                            'direction': direction,
                            'entry_price': round(entry_price, 5),
                            'tokyo_high': round(tokyo_high, 5),
                            'tokyo_low': round(tokyo_low, 5),
                            'tokyo_range_pips': round(tokyo_range / pip, 1),
                            'result': 'LOSS',
                            'pips': round(-(sl_pips + spread / pip), 1),
                            'entry_hour': entry_time.hour,
                        })
                        entry_done = True
                        break

            # ロンドン時間内に決済されなかった場合はノーカウント
            if not entry_done and direction:
                break

    return trades


def calculate_stats(trades: list) -> dict:
    """トレード統計を計算"""
    df = pd.DataFrame(trades)
    total = len(df)
    wins = (df['result'] == 'WIN').sum()
    losses = (df['result'] == 'LOSS').sum()
    win_rate = wins / total * 100 if total > 0 else 0

    win_pips = df[df['pips'] > 0]['pips'].sum()
    loss_pips = abs(df[df['pips'] < 0]['pips'].sum())
    pf = win_pips / loss_pips if loss_pips > 0 else float('inf')

    avg_win = df[df['pips'] > 0]['pips'].mean() if wins > 0 else 0
    avg_loss = df[df['pips'] < 0]['pips'].mean() if losses > 0 else 0

    cumulative = df['pips'].cumsum()
    max_dd = (cumulative - cumulative.expanding().max()).min()

    # 連勝・連敗
    max_consec_wins = 0
    max_consec_losses = 0
    cur = 0
    for r in df['result']:
        if r == 'WIN':
            cur = cur + 1 if cur > 0 else 1
            max_consec_wins = max(max_consec_wins, cur)
        else:
            cur = cur - 1 if cur < 0 else -1
            max_consec_losses = max(max_consec_losses, abs(cur))

    total_pips = df['pips'].sum()
    recovery_factor = abs(total_pips / max_dd) if max_dd != 0 else float('inf')

    return {
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 1),
        'total_pips': round(total_pips, 1),
        'profit_factor': round(pf, 2),
        'avg_win_pips': round(avg_win, 1),
        'avg_loss_pips': round(avg_loss, 1),
        'max_drawdown_pips': round(max_dd, 1),
        'recovery_factor': round(recovery_factor, 1),
        'max_consecutive_wins': max_consec_wins,
        'max_consecutive_losses': max_consec_losses,
    }


def print_results(stats: dict, params: dict):
    """結果を表示"""
    print()
    print("=" * 60)
    print("  Tokyo Range Breakout - バックテスト結果")
    print("=" * 60)
    print()
    print(f"  パラメータ:")
    print(f"    TP: {params['tp_pips']} pips")
    print(f"    SL: {params['sl_pips']} pips")
    print(f"    レンジ: {params['min_range_pips']}-{params['max_range_pips']} pips")
    print(f"    バッファ: {params['buffer_pips']} pips")
    print(f"    スプレッド: {params['spread_pips']} pips")
    print()
    print(f"  結果:")
    print(f"    トレード数:       {stats['total_trades']}")
    print(f"    勝率:             {stats['win_rate']}%  ({stats['wins']}W / {stats['losses']}L)")
    print(f"    合計pips:         {stats['total_pips']:+.1f}")
    print(f"    PF:               {stats['profit_factor']}")
    print(f"    平均利益:         {stats['avg_win_pips']:+.1f} pips")
    print(f"    平均損失:         {stats['avg_loss_pips']:+.1f} pips")
    print(f"    最大DD:           {stats['max_drawdown_pips']:.1f} pips")
    print(f"    リカバリーファクタ: {stats['recovery_factor']}")
    print(f"    最大連勝:         {stats['max_consecutive_wins']}")
    print(f"    最大連敗:         {stats['max_consecutive_losses']}")
    print()

    # 損益分岐勝率
    be_wr = abs(stats['avg_loss_pips']) / (stats['avg_win_pips'] + abs(stats['avg_loss_pips'])) * 100
    print(f"  損益分岐勝率:       {be_wr:.1f}%")
    print(f"  マージン:           {stats['win_rate'] - be_wr:+.1f}%")
    print("=" * 60)


def create_visualization(trades: list, output_path: Path):
    """可視化"""
    df = pd.DataFrame(trades)
    df['date_dt'] = pd.to_datetime(df['date'])
    cumulative = df['pips'].cumsum()
    running_max = cumulative.expanding().max()
    drawdowns = cumulative - running_max

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tokyo Range Breakout - Backtest Results\n'
                 f'USDJPY TP={PARAMS["tp_pips"]} / SL={PARAMS["sl_pips"]} (M5)',
                 fontsize=14, fontweight='bold')

    # 1. 累積損益
    ax = axes[0, 0]
    ax.plot(df['date_dt'], cumulative.values, color='blue', linewidth=1.5)
    ax.fill_between(df['date_dt'], cumulative.values, 0,
                    where=cumulative.values >= 0, alpha=0.3, color='green')
    ax.fill_between(df['date_dt'], cumulative.values, 0,
                    where=cumulative.values < 0, alpha=0.3, color='red')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Cumulative P/L (pips)')
    ax.set_ylabel('Pips')
    ax.grid(True, alpha=0.3)

    # 2. ドローダウン
    ax = axes[0, 1]
    ax.fill_between(df['date_dt'], drawdowns.values, 0, color='red', alpha=0.5)
    ax.set_title(f'Drawdown (Max: {drawdowns.min():.1f} pips)')
    ax.set_ylabel('Drawdown (pips)')
    ax.grid(True, alpha=0.3)

    # 3. 年別パフォーマンス
    ax = axes[1, 0]
    df['year'] = df['date_dt'].dt.year
    yearly = df.groupby('year')['pips'].agg(['sum', 'count'])
    yearly_wr = df.groupby('year')['result'].apply(lambda x: (x == 'WIN').mean() * 100)
    colors = ['green' if x >= 0 else 'red' for x in yearly['sum']]
    bars = ax.bar(yearly.index.astype(str), yearly['sum'], color=colors, alpha=0.7)
    for bar, (yr, row), wr in zip(bars, yearly.iterrows(), yearly_wr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{wr:.0f}%\n({int(row["count"])})',
                ha='center', va='bottom', fontsize=7)
    ax.set_title('Yearly Performance')
    ax.set_ylabel('Net Pips')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    # 4. エントリー時間分布
    ax = axes[1, 1]
    hour_counts = df.groupby(['entry_hour', 'result']).size().unstack(fill_value=0)
    if 'WIN' in hour_counts.columns and 'LOSS' in hour_counts.columns:
        ax.bar(hour_counts.index, hour_counts['WIN'], color='green', alpha=0.7, label='WIN')
        ax.bar(hour_counts.index, hour_counts['LOSS'], bottom=hour_counts['WIN'],
               color='red', alpha=0.7, label='LOSS')
    ax.set_title('Entry Hour Distribution (UTC)')
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Trades')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可視化保存: {output_path}")


def main():
    print("=" * 60)
    print("  Tokyo Range Breakout - スタンドアロン検証")
    print("=" * 60)

    # データ読み込み
    print(f"\nデータパス: {DATA_PATH}")
    print(f"通貨ペア: {SYMBOL}")
    print("\nM1 データ読み込み中...")
    df_m1 = load_m1_data(DATA_PATH, SYMBOL)
    print(f"  M1 レコード数: {len(df_m1):,}")
    print(f"  期間: {df_m1.index[0].date()} ~ {df_m1.index[-1].date()}")

    # M5 リサンプリング
    print("\nM5 にリサンプリング中...")
    df_m5 = resample_to_m5(df_m1)
    print(f"  M5 レコード数: {len(df_m5):,}")

    # バックテスト実行
    print("\nバックテスト実行中...")
    trades = run_backtest(df_m5, PARAMS)
    print(f"  完了: {len(trades)} トレード")

    if not trades:
        print("\n  トレードが発生しませんでした")
        return

    # 統計計算・表示
    stats = calculate_stats(trades)
    print_results(stats, PARAMS)

    # CSV 保存
    csv_path = OUTPUT_DIR / 'standalone_trades.csv'
    pd.DataFrame(trades).to_csv(csv_path, index=False)
    print(f"\nトレード履歴保存: {csv_path}")

    # 可視化
    png_path = OUTPUT_DIR / 'standalone_results.png'
    create_visualization(trades, png_path)

    print("\n完了!")


if __name__ == "__main__":
    main()
