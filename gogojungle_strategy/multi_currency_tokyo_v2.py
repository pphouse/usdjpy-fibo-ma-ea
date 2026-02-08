"""
東京レンジブレイクアウト - 6通貨ペアテスト (正確版)
元のhigh_winrate_research.pyと同じロジック
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


CURRENCY_PAIRS = {
    'USDJPY': {
        'path': Path("/Users/naoto/ドル円/usd_jpy_M1/extracted"),
        'folder': 'USDJPY',
        'pip_value': 0.01,
        'type': 'JPY',
        'min_range': 0.1,  # 10 pips
        'max_range': 0.5,  # 50 pips
        'buffer': 0.05,    # 5 pips
    },
    'EURJPY': {
        'path': Path("/Users/naoto/ドル円/eur_jpy_M1/extracted"),
        'folder': 'EURJPY',
        'pip_value': 0.01,
        'type': 'JPY',
        'min_range': 0.1,
        'max_range': 0.6,  # EURJPYは少しボラ高め
        'buffer': 0.05,
    },
    'GBPJPY': {
        'path': Path("/Users/naoto/ドル円/gbp_jpy_M1/extracted"),
        'folder': 'GBPJPY',
        'pip_value': 0.01,
        'type': 'JPY',
        'min_range': 0.15,  # GBPJPYはボラ高い
        'max_range': 0.8,
        'buffer': 0.08,
    },
    'AUDJPY': {
        'path': Path("/Users/naoto/ドル円/aud_jpy_M1/extracted"),
        'folder': 'AUDJPY',
        'pip_value': 0.01,
        'type': 'JPY',
        'min_range': 0.08,
        'max_range': 0.4,
        'buffer': 0.04,
    },
    'EURUSD': {
        'path': Path("/Users/naoto/ドル円/eur_usd_M1/extracted"),
        'folder': 'EURUSD',
        'pip_value': 0.0001,
        'type': 'USD',
        'min_range': 0.0010,  # 10 pips
        'max_range': 0.0050,  # 50 pips
        'buffer': 0.0005,     # 5 pips
    },
    'GBPUSD': {
        'path': Path("/Users/naoto/ドル円/gbp_usd_M1/extracted"),
        'folder': 'GBPUSD',
        'pip_value': 0.0001,
        'type': 'USD',
        'min_range': 0.0015,
        'max_range': 0.0080,
        'buffer': 0.0008,
    }
}


def load_m1_data(pair_info):
    """M1データを読み込む"""
    base_path = pair_info['path']
    folder_name = pair_info['folder']
    all_files = []

    symbol_folder = base_path / folder_name
    if symbol_folder.exists():
        for f in symbol_folder.glob(f"{folder_name}_20*_*.csv"):
            if "_all" not in f.name:
                all_files.append(f)

    for f in base_path.glob(f"{folder_name}_20*_*.csv"):
        if "_all" not in f.name:
            all_files.append(f)

    if not all_files:
        return None

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f, header=None,
                           names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
            dfs.append(df)
        except:
            pass

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['datetime'] = pd.to_datetime(df_all['date'] + ' ' + df_all['time'],
                                        format='%Y.%m.%d %H:%M')
    df_all = df_all.sort_values('datetime').reset_index(drop=True)
    df_all = df_all.set_index('datetime')
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    return df_all[['open', 'high', 'low', 'close', 'volume']]


def tokyo_range_breakout(df, pair_info, tp_pips, sl_pips):
    """
    東京レンジブレイクアウト戦略
    元のhigh_winrate_research.pyと同じロジック
    """
    pip_value = pair_info['pip_value']
    min_range = pair_info['min_range']
    max_range = pair_info['max_range']
    buffer = pair_info['buffer']
    spread = 0.3 * pip_value

    df['hour'] = df.index.hour

    trades = []
    dates = df.index.date
    unique_dates = pd.unique(dates)

    for date in unique_dates[100:-10]:
        day_data = df[df.index.date == date]

        # 東京時間（0-6 UTC = 9-15 JST）
        tokyo_data = day_data[(day_data['hour'] >= 0) & (day_data['hour'] < 6)]

        if len(tokyo_data) < 10:
            continue

        tokyo_high = tokyo_data['high'].max()
        tokyo_low = tokyo_data['low'].min()
        tokyo_range = tokyo_high - tokyo_low

        # レンジフィルター
        if tokyo_range < min_range or tokyo_range > max_range:
            continue

        # ロンドン時間（6-14 UTC）でブレイクを待つ
        london_data = day_data[(day_data['hour'] >= 6) & (day_data['hour'] < 14)]

        entry_done = False
        for idx in range(len(london_data)):
            if entry_done:
                break

            row = london_data.iloc[idx]
            close = row['close']

            direction = None
            if close > tokyo_high + buffer:
                direction = 'BUY'
                entry_price = close
            elif close < tokyo_low - buffer:
                direction = 'SELL'
                entry_price = close

            if direction is None:
                continue

            # 決済チェック（ロンドン時間の残り）
            remaining = london_data.iloc[idx+1:] if idx+1 < len(london_data) else pd.DataFrame()

            for i in range(len(remaining)):
                current_high = remaining.iloc[i]['high']
                current_low = remaining.iloc[i]['low']

                tp_dist = tp_pips * pip_value
                sl_dist = sl_pips * pip_value

                if direction == 'BUY':
                    if current_high >= entry_price + tp_dist:
                        trades.append({
                            'date': date,
                            'direction': direction,
                            'result': 'WIN',
                            'pips': tp_pips - spread / pip_value
                        })
                        entry_done = True
                        break
                    elif current_low <= entry_price - sl_dist:
                        trades.append({
                            'date': date,
                            'direction': direction,
                            'result': 'LOSS',
                            'pips': -(sl_pips + spread / pip_value)
                        })
                        entry_done = True
                        break
                else:
                    if current_low <= entry_price - tp_dist:
                        trades.append({
                            'date': date,
                            'direction': direction,
                            'result': 'WIN',
                            'pips': tp_pips - spread / pip_value
                        })
                        entry_done = True
                        break
                    elif current_high >= entry_price + sl_dist:
                        trades.append({
                            'date': date,
                            'direction': direction,
                            'result': 'LOSS',
                            'pips': -(sl_pips + spread / pip_value)
                        })
                        entry_done = True
                        break

            # ロンドン時間内で決済されなかった場合はトレードなし扱い
            if not entry_done and direction:
                break  # 次の日へ

    return trades


def analyze_trades(trades, symbol):
    """トレード分析"""
    if not trades:
        return None

    df_trades = pd.DataFrame(trades)
    total = len(df_trades)
    wins = (df_trades['result'] == 'WIN').sum()
    win_rate = wins / total * 100 if total > 0 else 0
    total_pips = df_trades['pips'].sum()

    # 連勝記録
    max_consecutive_wins = 0
    current_streak = 0
    for result in df_trades['result']:
        if result == 'WIN':
            current_streak += 1
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        else:
            current_streak = 0

    # PF
    win_pips = df_trades[df_trades['pips'] > 0]['pips'].sum()
    loss_pips = abs(df_trades[df_trades['pips'] < 0]['pips'].sum())
    pf = win_pips / loss_pips if loss_pips > 0 else 999

    # MaxDD
    cumulative = df_trades['pips'].cumsum()
    max_dd = (cumulative - cumulative.expanding().max()).min()

    return {
        'symbol': symbol,
        'trades': total,
        'win_rate': round(win_rate, 1),
        'total_pips': round(total_pips, 1),
        'max_consecutive_wins': max_consecutive_wins,
        'pf': round(pf, 2),
        'max_dd': round(max_dd, 1)
    }


def main():
    print("=" * 70)
    print("  東京レンジブレイクアウト - 6通貨ペアテスト (正確版)")
    print("  元のhigh_winrate_research.pyと同じロジック")
    print("=" * 70)

    # テストパラメータ
    test_params = [
        {'tp': 10, 'sl': 50},
        {'tp': 15, 'sl': 50},
        {'tp': 10, 'sl': 30},
        {'tp': 15, 'sl': 30},
    ]

    all_results = []

    for symbol, pair_info in CURRENCY_PAIRS.items():
        print(f"\n{'='*50}")
        print(f"  {symbol}")
        print(f"{'='*50}")

        df_m1 = load_m1_data(pair_info)
        if df_m1 is None:
            print(f"  データなし")
            continue

        # M5に変換
        df_m5 = df_m1.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        print(f"  M5データ: {len(df_m5):,}本")
        print(f"  期間: {df_m5.index[0].date()} ~ {df_m5.index[-1].date()}")

        for params in test_params:
            tp, sl = params['tp'], params['sl']

            trades = tokyo_range_breakout(df_m5, pair_info, tp, sl)
            stats = analyze_trades(trades, symbol)

            if stats:
                stats['tp'] = tp
                stats['sl'] = sl
                all_results.append(stats)

                status = "★★★" if stats['win_rate'] >= 90 else "★★" if stats['win_rate'] >= 80 else "★" if stats['win_rate'] >= 70 else ""
                print(f"  TP{tp}/SL{sl}: WR={stats['win_rate']:.1f}%, 連勝={stats['max_consecutive_wins']}, "
                      f"Pips={stats['total_pips']:+,.0f}, PF={stats['pf']:.2f} {status}")

    # 結果をDataFrameに変換
    df_results = pd.DataFrame(all_results)

    # 可視化
    print("\n" + "=" * 70)
    print("  結果可視化")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Tokyo Range Breakout - 6 Currency Pairs (Accurate Implementation)\n'
                 'Same Logic as Original High Win Rate Research',
                 fontsize=14, fontweight='bold')

    # 1. 通貨ペア × パラメータ別勝率
    ax1 = fig.add_subplot(2, 2, 1)
    pivot_wr = df_results.pivot_table(values='win_rate', index='symbol',
                                       columns=['tp', 'sl'], aggfunc='first')
    if len(pivot_wr) > 0:
        im = ax1.imshow(pivot_wr.values, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)
        ax1.set_yticks(range(len(pivot_wr.index)))
        ax1.set_yticklabels(pivot_wr.index)
        ax1.set_xticks(range(len(pivot_wr.columns)))
        ax1.set_xticklabels([f'TP{c[0]}/SL{c[1]}' for c in pivot_wr.columns], rotation=45, ha='right')
        for i in range(len(pivot_wr.index)):
            for j in range(len(pivot_wr.columns)):
                val = pivot_wr.values[i, j]
                if not pd.isna(val):
                    ax1.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=9,
                            color='white' if val < 60 else 'black')
        plt.colorbar(im, ax=ax1, label='Win Rate (%)')
    ax1.set_title('Win Rate Heatmap', fontsize=12, fontweight='bold')

    # 2. 通貨ペア別ベスト結果
    ax2 = fig.add_subplot(2, 2, 2)
    best_by_symbol = df_results.loc[df_results.groupby('symbol')['win_rate'].idxmax()]
    colors = ['green' if x >= 0 else 'red' for x in best_by_symbol['total_pips'].values]
    bars = ax2.bar(best_by_symbol['symbol'], best_by_symbol['total_pips'], color=colors, alpha=0.7)
    for i, (bar, row) in enumerate(zip(bars, best_by_symbol.itertuples())):
        label = f"WR:{row.win_rate:.0f}%\nTP{row.tp}/SL{row.sl}"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                label, ha='center', va='bottom', fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Currency Pair')
    ax2.set_ylabel('Best Net Pips')
    ax2.set_title('Best Result by Currency Pair', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 連勝記録比較
    ax3 = fig.add_subplot(2, 2, 3)
    best_consec = df_results.loc[df_results.groupby('symbol')['max_consecutive_wins'].idxmax()]
    ax3.bar(best_consec['symbol'], best_consec['max_consecutive_wins'], color='blue', alpha=0.7)
    for i, (sym, val, wr) in enumerate(zip(best_consec['symbol'],
                                            best_consec['max_consecutive_wins'],
                                            best_consec['win_rate'])):
        ax3.text(i, val, f'{int(val)}\n({wr:.0f}%)', ha='center', va='bottom', fontsize=9)
    ax3.set_xlabel('Currency Pair')
    ax3.set_ylabel('Max Consecutive Wins')
    ax3.set_title('Maximum Consecutive Wins by Currency Pair', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. サマリー
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # 勝率80%以上のペア
    high_wr = df_results[df_results['win_rate'] >= 80].groupby('symbol').first().reset_index()
    # 利益プラスのペア
    profitable = df_results[df_results['total_pips'] > 0].groupby('symbol').first().reset_index()

    summary_text = f"""
═══════════════════════════════════════════════════════════════
     TOKYO RANGE BREAKOUT - MULTI-CURRENCY VALIDATION
═══════════════════════════════════════════════════════════════

  Total Results: {len(df_results)}
  Currency Pairs Tested: {df_results['symbol'].nunique()}

  Win Rate Analysis:
    Pairs with WR >= 90%: {(df_results['win_rate'] >= 90).sum()}
    Pairs with WR >= 80%: {(df_results['win_rate'] >= 80).sum()}
    Pairs with WR >= 70%: {(df_results['win_rate'] >= 70).sum()}

  Profitability:
    Profitable Results: {(df_results['total_pips'] > 0).sum()} / {len(df_results)}

  Best Results by Win Rate:
"""
    top3 = df_results.nlargest(5, 'win_rate')
    for _, row in top3.iterrows():
        summary_text += f"    {row['symbol']} TP{row['tp']}/SL{row['sl']}: WR={row['win_rate']:.1f}%, Pips={row['total_pips']:+,.0f}\n"

    summary_text += """
═══════════════════════════════════════════════════════════════
"""

    ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = '/Users/naoto/ドル円/gogojungle_strategy/multi_currency_tokyo_v2.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n保存: {save_path}")

    # CSV保存
    csv_path = '/Users/naoto/ドル円/gogojungle_strategy/multi_currency_tokyo_v2.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"保存: {csv_path}")

    # 最終サマリー
    print("\n" + "=" * 70)
    print("  【検証結果サマリー】")
    print("=" * 70)

    print("\n【勝率ランキング Top 10】")
    print("-" * 70)
    top10 = df_results.nlargest(10, 'win_rate')
    for _, row in top10.iterrows():
        status = "★★★" if row['win_rate'] >= 90 else "★★" if row['win_rate'] >= 80 else "★"
        print(f"  {status} {row['symbol']} TP{row['tp']}/SL{row['sl']}: "
              f"WR={row['win_rate']:.1f}%, 連勝={row['max_consecutive_wins']}, "
              f"Pips={row['total_pips']:+,.0f}, PF={row['pf']:.2f}")

    print("\n【通貨ペア別ベスト】")
    print("-" * 70)
    for symbol in df_results['symbol'].unique():
        sym_data = df_results[df_results['symbol'] == symbol]
        best = sym_data.loc[sym_data['win_rate'].idxmax()]
        status = "★★★" if best['win_rate'] >= 90 else "★★" if best['win_rate'] >= 80 else "★" if best['win_rate'] >= 70 else "△"
        print(f"  {status} {symbol}: TP{best['tp']}/SL{best['sl']} - "
              f"WR={best['win_rate']:.1f}%, Pips={best['total_pips']:+,.0f}")

    print("\n" + "=" * 70)
    print("  完了!")
    print("=" * 70)


if __name__ == "__main__":
    main()
