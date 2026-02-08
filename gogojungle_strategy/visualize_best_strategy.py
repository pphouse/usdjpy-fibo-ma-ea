"""
Tokyo Range Breakout ベスト戦略の詳細可視化
TP=10/SL=50 (勝率96.2%, PF 4.91) の詳細分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path("/home/azureuser/usdjpy-fibo-ma-ea")


def load_m1_data():
    """USDJPY M1データを読み込み"""
    base_path = BASE_DIR / "usd_jpy_M1/extracted"
    all_files = []

    usdjpy_folder = base_path / "USDJPY"
    if usdjpy_folder.exists():
        for f in usdjpy_folder.glob("USDJPY_20*_*.csv"):
            if "_all" not in f.name:
                all_files.append(f)

    for f in base_path.glob("USDJPY_20*_*.csv"):
        if "_all" not in f.name:
            all_files.append(f)

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f, header=None,
                           names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
            dfs.append(df)
        except:
            pass

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['datetime'] = pd.to_datetime(df_all['date'] + ' ' + df_all['time'],
                                        format='%Y.%m.%d %H:%M')
    df_all = df_all.sort_values('datetime').reset_index(drop=True)
    df_all = df_all.set_index('datetime')
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    return df_all[['open', 'high', 'low', 'close', 'volume']]


def tokyo_range_breakout_detailed(df, tp_pips=10, sl_pips=50):
    """トレード履歴付きの東京レンジブレイクアウト"""
    pip_value = 0.01
    min_range = 0.1
    max_range = 0.5
    buffer = 0.05
    spread = 0.3 * pip_value

    df['hour'] = df.index.hour
    trades = []
    dates = df.index.date
    unique_dates = pd.unique(dates)

    for date in unique_dates[100:-10]:
        day_data = df[df.index.date == date]
        tokyo_data = day_data[(day_data['hour'] >= 0) & (day_data['hour'] < 6)]

        if len(tokyo_data) < 10:
            continue

        tokyo_high = tokyo_data['high'].max()
        tokyo_low = tokyo_data['low'].min()
        tokyo_range = tokyo_high - tokyo_low

        if tokyo_range < min_range or tokyo_range > max_range:
            continue

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

            remaining = london_data.iloc[idx+1:] if idx+1 < len(london_data) else pd.DataFrame()
            entry_time = london_data.index[idx]

            for i in range(len(remaining)):
                current_high = remaining.iloc[i]['high']
                current_low = remaining.iloc[i]['low']
                exit_time = remaining.index[i]

                tp_dist = tp_pips * pip_value
                sl_dist = sl_pips * pip_value

                if direction == 'BUY':
                    if current_high >= entry_price + tp_dist:
                        trades.append({
                            'date': date,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': entry_price + tp_dist,
                            'result': 'WIN',
                            'pips': tp_pips - spread / pip_value,
                            'tokyo_high': tokyo_high,
                            'tokyo_low': tokyo_low,
                            'tokyo_range_pips': tokyo_range / pip_value,
                            'entry_hour': entry_time.hour,
                        })
                        entry_done = True
                        break
                    elif current_low <= entry_price - sl_dist:
                        trades.append({
                            'date': date,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': entry_price - sl_dist,
                            'result': 'LOSS',
                            'pips': -(sl_pips + spread / pip_value),
                            'tokyo_high': tokyo_high,
                            'tokyo_low': tokyo_low,
                            'tokyo_range_pips': tokyo_range / pip_value,
                            'entry_hour': entry_time.hour,
                        })
                        entry_done = True
                        break
                else:
                    if current_low <= entry_price - tp_dist:
                        trades.append({
                            'date': date,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': entry_price - tp_dist,
                            'result': 'WIN',
                            'pips': tp_pips - spread / pip_value,
                            'tokyo_high': tokyo_high,
                            'tokyo_low': tokyo_low,
                            'tokyo_range_pips': tokyo_range / pip_value,
                            'entry_hour': entry_time.hour,
                        })
                        entry_done = True
                        break
                    elif current_high >= entry_price + sl_dist:
                        trades.append({
                            'date': date,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': entry_price + sl_dist,
                            'result': 'LOSS',
                            'pips': -(sl_pips + spread / pip_value),
                            'tokyo_high': tokyo_high,
                            'tokyo_low': tokyo_low,
                            'tokyo_range_pips': tokyo_range / pip_value,
                            'entry_hour': entry_time.hour,
                        })
                        entry_done = True
                        break

            if not entry_done and direction:
                break

    return trades


def main():
    print("=" * 70)
    print("  Tokyo Range Breakout - ベスト戦略 詳細分析")
    print("  USDJPY TP=10 / SL=50")
    print("=" * 70)

    # データ読み込み
    print("\nM1データ読み込み中...")
    df_m1 = load_m1_data()
    print(f"  M1データ: {len(df_m1):,}本")

    # M5に変換
    df_m5 = df_m1.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    print(f"  M5データ: {len(df_m5):,}本")
    print(f"  期間: {df_m5.index[0].date()} ~ {df_m5.index[-1].date()}")

    # バックテスト実行
    print("\nバックテスト実行中...")
    trades = tokyo_range_breakout_detailed(df_m5, tp_pips=10, sl_pips=50)
    df_trades = pd.DataFrame(trades)

    print(f"  トレード数: {len(df_trades)}")

    # 基本統計
    total = len(df_trades)
    wins = (df_trades['result'] == 'WIN').sum()
    losses = (df_trades['result'] == 'LOSS').sum()
    win_rate = wins / total * 100
    total_pips = df_trades['pips'].sum()

    win_pips = df_trades[df_trades['pips'] > 0]['pips'].sum()
    loss_pips = abs(df_trades[df_trades['pips'] < 0]['pips'].sum())
    pf = win_pips / loss_pips if loss_pips > 0 else 999
    avg_win = df_trades[df_trades['pips'] > 0]['pips'].mean()
    avg_loss = df_trades[df_trades['pips'] < 0]['pips'].mean()

    cumulative = df_trades['pips'].cumsum()
    max_dd = (cumulative - cumulative.expanding().max()).min()
    running_max = cumulative.expanding().max()
    drawdowns = cumulative - running_max

    # 連勝・連敗
    max_consec_wins = 0
    max_consec_losses = 0
    cur_wins = 0
    cur_losses = 0
    consec_wins_list = []
    consec_losses_list = []
    for r in df_trades['result']:
        if r == 'WIN':
            cur_wins += 1
            if cur_losses > 0:
                consec_losses_list.append(cur_losses)
            cur_losses = 0
            max_consec_wins = max(max_consec_wins, cur_wins)
        else:
            cur_losses += 1
            if cur_wins > 0:
                consec_wins_list.append(cur_wins)
            cur_wins = 0
            max_consec_losses = max(max_consec_losses, cur_losses)
    if cur_wins > 0:
        consec_wins_list.append(cur_wins)
    if cur_losses > 0:
        consec_losses_list.append(cur_losses)

    print(f"\n{'='*60}")
    print(f"  勝率: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  合計pips: {total_pips:+,.1f}")
    print(f"  PF: {pf:.2f}")
    print(f"  平均利益: {avg_win:.1f} pips")
    print(f"  平均損失: {avg_loss:.1f} pips")
    print(f"  最大DD: {max_dd:.1f} pips")
    print(f"  最大連勝: {max_consec_wins}")
    print(f"  最大連敗: {max_consec_losses}")
    print(f"{'='*60}")

    # ===== 可視化 =====
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('Tokyo Range Breakout - Best Strategy Detailed Analysis\n'
                 'USDJPY TP=10 / SL=50 (2015-2025)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. 累積損益カーブ
    ax1 = fig.add_subplot(3, 3, 1)
    dates_plot = pd.to_datetime(df_trades['date'])
    ax1.plot(dates_plot, cumulative.values, color='blue', linewidth=1.5, label='Cumulative P/L')
    ax1.fill_between(dates_plot, cumulative.values, 0,
                     where=cumulative.values >= 0, alpha=0.3, color='green')
    ax1.fill_between(dates_plot, cumulative.values, 0,
                     where=cumulative.values < 0, alpha=0.3, color='red')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_title('Cumulative P/L (pips)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pips')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # 2. ドローダウンカーブ
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.fill_between(dates_plot, drawdowns.values, 0, color='red', alpha=0.5)
    ax2.plot(dates_plot, drawdowns.values, color='darkred', linewidth=0.8)
    ax2.set_title(f'Drawdown (Max: {max_dd:.1f} pips)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (pips)')
    ax2.grid(True, alpha=0.3)

    # 3. 年別パフォーマンス
    ax3 = fig.add_subplot(3, 3, 3)
    df_trades['year'] = pd.to_datetime(df_trades['date']).dt.year
    yearly = df_trades.groupby('year').agg(
        total_pips=('pips', 'sum'),
        trades=('pips', 'count'),
        win_rate=('result', lambda x: (x == 'WIN').mean() * 100)
    )
    colors_yearly = ['green' if x >= 0 else 'red' for x in yearly['total_pips']]
    bars = ax3.bar(yearly.index.astype(str), yearly['total_pips'], color=colors_yearly, alpha=0.7)
    for bar, (_, row) in zip(bars, yearly.iterrows()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{row["win_rate"]:.0f}%\n({int(row["trades"])})',
                ha='center', va='bottom', fontsize=7)
    ax3.set_title('Yearly Performance', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Net Pips')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)

    # 4. 月別パフォーマンスヒートマップ
    ax4 = fig.add_subplot(3, 3, 4)
    df_trades['month'] = pd.to_datetime(df_trades['date']).dt.month
    monthly_pivot = df_trades.pivot_table(values='pips', index='year', columns='month', aggfunc='sum')
    monthly_pivot = monthly_pivot.reindex(columns=range(1, 13))
    im = ax4.imshow(monthly_pivot.values, cmap='RdYlGn', aspect='auto')
    ax4.set_yticks(range(len(monthly_pivot.index)))
    ax4.set_yticklabels(monthly_pivot.index.astype(int))
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
                        fontsize=7)
    for i in range(monthly_pivot.shape[0]):
        for j in range(monthly_pivot.shape[1]):
            val = monthly_pivot.values[i, j]
            if not np.isnan(val):
                ax4.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6,
                        color='white' if abs(val) > 100 else 'black')
    plt.colorbar(im, ax=ax4, label='Pips', shrink=0.8)
    ax4.set_title('Monthly P/L Heatmap', fontsize=12, fontweight='bold')

    # 5. エントリー時間分布
    ax5 = fig.add_subplot(3, 3, 5)
    hour_counts = df_trades.groupby(['entry_hour', 'result']).size().unstack(fill_value=0)
    if 'WIN' in hour_counts.columns and 'LOSS' in hour_counts.columns:
        ax5.bar(hour_counts.index, hour_counts['WIN'], color='green', alpha=0.7, label='WIN')
        ax5.bar(hour_counts.index, hour_counts['LOSS'], bottom=hour_counts['WIN'],
                color='red', alpha=0.7, label='LOSS')
    ax5.set_title('Entry Hour Distribution (UTC)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Hour (UTC)')
    ax5.set_ylabel('Trade Count')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. BUY/SELL方向別分析
    ax6 = fig.add_subplot(3, 3, 6)
    dir_stats = df_trades.groupby('direction').agg(
        total=('pips', 'count'),
        wins=('result', lambda x: (x == 'WIN').sum()),
        pips=('pips', 'sum')
    )
    dir_stats['win_rate'] = dir_stats['wins'] / dir_stats['total'] * 100
    x_pos = range(len(dir_stats))
    bars = ax6.bar(dir_stats.index, dir_stats['pips'],
                   color=['green' if d == 'BUY' else 'red' for d in dir_stats.index], alpha=0.7)
    for bar, (_, row) in zip(bars, dir_stats.iterrows()):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'WR:{row["win_rate"]:.1f}%\n{int(row["total"])} trades',
                ha='center', va='bottom', fontsize=9)
    ax6.set_title('BUY vs SELL Performance', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Net Pips')
    ax6.axhline(y=0, color='black', linewidth=0.5)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. 東京レンジ幅とトレード結果の関係
    ax7 = fig.add_subplot(3, 3, 7)
    win_trades = df_trades[df_trades['result'] == 'WIN']
    loss_trades = df_trades[df_trades['result'] == 'LOSS']
    ax7.hist(win_trades['tokyo_range_pips'], bins=20, alpha=0.6, color='green', label=f'WIN ({len(win_trades)})')
    ax7.hist(loss_trades['tokyo_range_pips'], bins=20, alpha=0.6, color='red', label=f'LOSS ({len(loss_trades)})')
    ax7.set_title('Tokyo Range Width Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Tokyo Range (pips)')
    ax7.set_ylabel('Frequency')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 8. 連勝分布
    ax8 = fig.add_subplot(3, 3, 8)
    if consec_wins_list:
        ax8.hist(consec_wins_list, bins=range(1, max(consec_wins_list)+2), color='green',
                alpha=0.7, edgecolor='black', align='left')
        ax8.axvline(x=np.mean(consec_wins_list), color='blue', linestyle='--',
                   label=f'Avg: {np.mean(consec_wins_list):.1f}')
        ax8.axvline(x=max_consec_wins, color='red', linestyle='--',
                   label=f'Max: {max_consec_wins}')
    ax8.set_title('Consecutive Wins Distribution', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Consecutive Wins')
    ax8.set_ylabel('Frequency')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # 9. サマリーテーブル
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')

    # Recovery Factor計算
    recovery_factor = abs(total_pips / max_dd) if max_dd != 0 else 0

    summary = f"""
 ══════════════════════════════════════════
   TOKYO RANGE BREAKOUT - BEST STRATEGY
   USDJPY  TP=10 / SL=50  (M5, 2015-2025)
 ══════════════════════════════════════════

   Total Trades:     {total:>8}
   Win Rate:         {win_rate:>7.1f}%
   Wins / Losses:    {wins:>4} / {losses}

   Total Pips:       {total_pips:>+8.1f}
   Profit Factor:    {pf:>8.2f}

   Avg Win:          {avg_win:>+8.1f} pips
   Avg Loss:         {avg_loss:>+8.1f} pips
   Payoff Ratio:     {abs(avg_win/avg_loss):>8.2f}

   Max Drawdown:     {max_dd:>8.1f} pips
   Recovery Factor:  {recovery_factor:>8.1f}

   Max Consec Wins:  {max_consec_wins:>8}
   Max Consec Loss:  {max_consec_losses:>8}
   Avg Consec Wins:  {np.mean(consec_wins_list):>8.1f}

   Best Year:        {yearly['total_pips'].idxmax()} ({yearly['total_pips'].max():+.0f})
   Worst Year:       {yearly['total_pips'].idxmin()} ({yearly['total_pips'].min():+.0f})
 ══════════════════════════════════════════
"""
    ax9.text(0.05, 0.5, summary, transform=ax9.transAxes,
            fontsize=9, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = str(BASE_DIR / 'gogojungle_strategy/best_strategy_detailed.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n保存: {save_path}")

    # トレード履歴CSV保存
    csv_path = str(BASE_DIR / 'gogojungle_strategy/best_strategy_trades.csv')
    df_trades.to_csv(csv_path, index=False)
    print(f"保存: {csv_path}")

    print("\n完了!")


if __name__ == "__main__":
    main()
