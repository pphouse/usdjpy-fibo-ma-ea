"""
東京レンジブレイクアウト - 6通貨ペアテスト
USDJPY, EURJPY, GBPJPY, AUDJPY, EURUSD, GBPUSD
勝率96%戦略の通貨ペア別検証
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
        'type': 'JPY'
    },
    'EURJPY': {
        'path': Path("/Users/naoto/ドル円/eur_jpy_M1/extracted"),
        'folder': 'EURJPY',
        'pip_value': 0.01,
        'type': 'JPY'
    },
    'GBPJPY': {
        'path': Path("/Users/naoto/ドル円/gbp_jpy_M1/extracted"),
        'folder': 'GBPJPY',
        'pip_value': 0.01,
        'type': 'JPY'
    },
    'AUDJPY': {
        'path': Path("/Users/naoto/ドル円/aud_jpy_M1/extracted"),
        'folder': 'AUDJPY',
        'pip_value': 0.01,
        'type': 'JPY'
    },
    'EURUSD': {
        'path': Path("/Users/naoto/ドル円/eur_usd_M1/extracted"),
        'folder': 'EURUSD',
        'pip_value': 0.0001,
        'type': 'USD'
    },
    'GBPUSD': {
        'path': Path("/Users/naoto/ドル円/gbp_usd_M1/extracted"),
        'folder': 'GBPUSD',
        'pip_value': 0.0001,
        'type': 'USD'
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


def convert_to_m5(df_m1):
    """M1をM5に変換"""
    return df_m1.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


class TokyoRangeBreakout:
    def __init__(self, symbol, pip_value, spread_pips=0.3):
        self.symbol = symbol
        self.pip_value = pip_value
        self.spread_pips = spread_pips
        self.df = None
        self.trades = []

    def set_data(self, df):
        self.df = df
        self.df['hour'] = self.df.index.hour
        self.df['date'] = self.df.index.date

    def run_strategy(self, tp_pips, sl_pips):
        """東京レンジブレイクアウト戦略を実行"""
        self.trades = []
        spread = self.spread_pips * self.pip_value

        dates = self.df['date'].unique()

        for date in dates[1:-1]:
            day_data = self.df[self.df['date'] == date]

            # 東京時間（0-6 UTC = 9-15 JST）でレンジを形成
            tokyo_data = day_data[(day_data['hour'] >= 0) & (day_data['hour'] < 6)]
            if len(tokyo_data) < 10:
                continue

            tokyo_high = tokyo_data['high'].max()
            tokyo_low = tokyo_data['low'].min()
            tokyo_range = tokyo_high - tokyo_low

            # レンジが狭すぎる場合はスキップ
            min_range = 10 * self.pip_value
            max_range = 100 * self.pip_value
            if tokyo_range < min_range or tokyo_range > max_range:
                continue

            # ロンドン時間（6-14 UTC）でブレイクアウトを探す
            london_data = day_data[(day_data['hour'] >= 6) & (day_data['hour'] < 14)]

            entry_done = False
            for idx in range(len(london_data)):
                if entry_done:
                    break

                row = london_data.iloc[idx]
                close = row['close']

                # ブレイクアウト判定（5pipsのバッファ）
                buffer = 5 * self.pip_value
                direction = None

                if close > tokyo_high + buffer:
                    direction = 'BUY'
                    entry_price = close + spread
                elif close < tokyo_low - buffer:
                    direction = 'SELL'
                    entry_price = close

                if direction is None:
                    continue

                # エントリー
                entry_time = london_data.index[idx]

                tp_dist = tp_pips * self.pip_value
                sl_dist = sl_pips * self.pip_value

                if direction == 'BUY':
                    tp_price = entry_price + tp_dist
                    sl_price = entry_price - sl_dist
                else:
                    tp_price = entry_price - tp_dist
                    sl_price = entry_price + sl_dist

                # 決済を探す（当日の残りと翌日まで）
                remaining_idx = london_data.index.get_loc(entry_time)
                check_data = london_data.iloc[remaining_idx+1:]

                result = None
                exit_time = None

                for check_idx in range(len(check_data)):
                    check_row = check_data.iloc[check_idx]

                    if direction == 'BUY':
                        if check_row['high'] >= tp_price:
                            result = 'WIN'
                            exit_time = check_data.index[check_idx]
                            break
                        elif check_row['low'] <= sl_price:
                            result = 'LOSS'
                            exit_time = check_data.index[check_idx]
                            break
                    else:
                        if check_row['low'] <= tp_price:
                            result = 'WIN'
                            exit_time = check_data.index[check_idx]
                            break
                        elif check_row['high'] >= sl_price:
                            result = 'LOSS'
                            exit_time = check_data.index[check_idx]
                            break

                # 決済されなかった場合は終値で決済
                if result is None and len(check_data) > 0:
                    last_close = check_data.iloc[-1]['close']
                    if direction == 'BUY':
                        pnl = (last_close - entry_price) / self.pip_value
                    else:
                        pnl = (entry_price - last_close) / self.pip_value
                    result = 'WIN' if pnl > 0 else 'LOSS'
                    exit_time = check_data.index[-1]

                if result:
                    pips = tp_pips if result == 'WIN' else -sl_pips
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'direction': direction,
                        'result': result,
                        'pips': pips - self.spread_pips
                    })
                    entry_done = True

        return self.trades

    def get_stats(self):
        """統計情報を取得"""
        if not self.trades:
            return None

        df_trades = pd.DataFrame(self.trades)

        wins = (df_trades['result'] == 'WIN').sum()
        total = len(df_trades)
        win_rate = wins / total * 100 if total > 0 else 0

        net_pips = df_trades['pips'].sum()

        # 連勝記録
        max_consec_wins = 0
        current_wins = 0
        for result in df_trades['result']:
            if result == 'WIN':
                current_wins += 1
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_wins = 0

        # 最大ドローダウン
        cumsum = df_trades['pips'].cumsum()
        cummax = cumsum.expanding().max()
        drawdown = cumsum - cummax
        max_dd = drawdown.min()

        # プロフィットファクター
        gross_profit = df_trades[df_trades['pips'] > 0]['pips'].sum()
        gross_loss = abs(df_trades[df_trades['pips'] < 0]['pips'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'symbol': self.symbol,
            'trades': total,
            'win_rate': win_rate,
            'net_pips': net_pips,
            'max_consec_wins': max_consec_wins,
            'max_dd': max_dd,
            'pf': pf,
            'trades_df': df_trades
        }


def main():
    print("=" * 70)
    print("  東京レンジブレイクアウト - 6通貨ペアテスト")
    print("  勝率96%戦略の多通貨ペア検証")
    print("=" * 70)

    # テストパラメータ（USDJPYで最高成績だった設定）
    test_params = [
        {'tp': 10, 'sl': 50},  # 96.2% win rate on USDJPY
        {'tp': 15, 'sl': 50},  # 94.2% win rate on USDJPY
        {'tp': 20, 'sl': 50},  # 91.1% win rate on USDJPY
    ]

    all_results = []
    pair_trades = {}

    for symbol, pair_info in CURRENCY_PAIRS.items():
        print(f"\n{'='*50}")
        print(f"  {symbol} 読み込み中...")
        print(f"{'='*50}")

        df_m1 = load_m1_data(pair_info)
        if df_m1 is None:
            print(f"  {symbol}: データなし")
            continue

        print(f"  M1データ: {len(df_m1):,}本")

        df_m5 = convert_to_m5(df_m1)
        print(f"  M5データ: {len(df_m5):,}本")
        print(f"  期間: {df_m5.index[0]} ~ {df_m5.index[-1]}")

        strategy = TokyoRangeBreakout(
            symbol=symbol,
            pip_value=pair_info['pip_value'],
            spread_pips=0.3 if pair_info['type'] == 'JPY' else 0.0003
        )
        strategy.set_data(df_m5)

        for params in test_params:
            tp, sl = params['tp'], params['sl']
            print(f"\n  テスト: TP{tp}/SL{sl}")

            strategy.run_strategy(tp, sl)
            stats = strategy.get_stats()

            if stats:
                print(f"    トレード数: {stats['trades']}")
                print(f"    勝率: {stats['win_rate']:.1f}%")
                print(f"    連勝記録: {stats['max_consec_wins']}")
                print(f"    Net Pips: {stats['net_pips']:+,.1f}")
                print(f"    PF: {stats['pf']:.2f}")

                all_results.append({
                    'symbol': symbol,
                    'tp': tp,
                    'sl': sl,
                    'trades': stats['trades'],
                    'win_rate': stats['win_rate'],
                    'consec_wins': stats['max_consec_wins'],
                    'net_pips': stats['net_pips'],
                    'max_dd': stats['max_dd'],
                    'pf': stats['pf']
                })

                key = f"{symbol}_TP{tp}_SL{sl}"
                pair_trades[key] = stats['trades_df']

    # 結果をDataFrameに変換
    df_results = pd.DataFrame(all_results)

    # 可視化
    print("\n" + "=" * 70)
    print("  結果可視化")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Tokyo Range Breakout - 6 Currency Pairs Validation\n'
                 'Target: 96%+ Win Rate Strategy for GogoJungle',
                 fontsize=16, fontweight='bold')

    # 1. 通貨ペア別勝率比較
    ax1 = fig.add_subplot(2, 2, 1)
    for tp_sl in [(10, 50), (15, 50), (20, 50)]:
        subset = df_results[(df_results['tp'] == tp_sl[0]) & (df_results['sl'] == tp_sl[1])]
        if len(subset) > 0:
            ax1.bar([f"{row['symbol']}\n({row['trades']})" for _, row in subset.iterrows()],
                   subset['win_rate'].values,
                   label=f'TP{tp_sl[0]}/SL{tp_sl[1]}', alpha=0.7)
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% Target')
    ax1.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% Target')
    ax1.set_xlabel('Currency Pair (Trades)')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate by Currency Pair', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. 通貨ペア別Net Pips
    ax2 = fig.add_subplot(2, 2, 2)
    best_params = df_results[df_results['tp'] == 10]
    if len(best_params) > 0:
        colors = ['green' if x >= 0 else 'red' for x in best_params['net_pips'].values]
        bars = ax2.bar(best_params['symbol'], best_params['net_pips'], color=colors, alpha=0.7)
        for bar, val in zip(bars, best_params['net_pips']):
            ax2.text(bar.get_x() + bar.get_width()/2, val,
                    f'{val:+,.0f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Currency Pair')
    ax2.set_ylabel('Net Pips')
    ax2.set_title('Net Pips by Currency Pair (TP10/SL50)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 連勝記録比較
    ax3 = fig.add_subplot(2, 2, 3)
    if len(best_params) > 0:
        ax3.bar(best_params['symbol'], best_params['consec_wins'], color='blue', alpha=0.7)
        for i, (sym, val) in enumerate(zip(best_params['symbol'], best_params['consec_wins'])):
            ax3.text(i, val, str(int(val)), ha='center', va='bottom', fontsize=10)
    ax3.set_xlabel('Currency Pair')
    ax3.set_ylabel('Consecutive Wins')
    ax3.set_title('Maximum Consecutive Wins (TP10/SL50)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. サマリーテーブル
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # TP10/SL50の結果でサマリー
    summary_df = df_results[df_results['tp'] == 10].copy()
    if len(summary_df) > 0:
        total_trades = summary_df['trades'].sum()
        avg_wr = summary_df['win_rate'].mean()
        total_pips = summary_df['net_pips'].sum()
        avg_consec = summary_df['consec_wins'].mean()

        # 勝率80%以上のペア数
        high_wr_pairs = (summary_df['win_rate'] >= 80).sum()

        summary_text = f"""
═══════════════════════════════════════════════════════
     TOKYO RANGE BREAKOUT - MULTI-CURRENCY SUMMARY
               TP10/SL50 Configuration
═══════════════════════════════════════════════════════

  Currency Pairs Tested: {len(summary_df)}

  Win Rate Analysis:
    Average Win Rate: {avg_wr:.1f}%
    Pairs with WR >= 80%: {high_wr_pairs}/{len(summary_df)}
    Pairs with WR >= 90%: {(summary_df['win_rate'] >= 90).sum()}/{len(summary_df)}

  Trading Statistics:
    Total Trades: {total_trades:,}
    Combined Net Pips: {total_pips:+,.0f}
    Average Consec Wins: {avg_consec:.0f}

  Individual Results:
"""
        for _, row in summary_df.iterrows():
            summary_text += f"    {row['symbol']}: WR={row['win_rate']:.1f}%, Pips={row['net_pips']:+,.0f}\n"

        summary_text += """
═══════════════════════════════════════════════════════
"""

        ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = '/Users/naoto/ドル円/gogojungle_strategy/multi_currency_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n保存: {save_path}")

    # CSV保存
    csv_path = '/Users/naoto/ドル円/gogojungle_strategy/multi_currency_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"保存: {csv_path}")

    # 最終サマリー
    print("\n" + "=" * 70)
    print("  【6通貨ペア検証結果】")
    print("=" * 70)

    print("\n【TP10/SL50 - 最高勝率設定】")
    print("-" * 60)
    tp10_results = df_results[df_results['tp'] == 10].sort_values('win_rate', ascending=False)
    for _, row in tp10_results.iterrows():
        status = "★" if row['win_rate'] >= 90 else "○" if row['win_rate'] >= 80 else "△"
        print(f"  {status} {row['symbol']}: WR={row['win_rate']:.1f}%, "
              f"連勝={int(row['consec_wins'])}, Pips={row['net_pips']:+,.0f}, PF={row['pf']:.2f}")

    print("\n" + "=" * 70)
    print("  完了!")
    print("=" * 70)


if __name__ == "__main__":
    main()
