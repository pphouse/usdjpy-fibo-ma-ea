"""
ゴゴジャン向け 高勝率戦略リサーチ
目標：勝率80%以上、連勝記録重視
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


def load_m1_data():
    """M1データを読み込み"""
    base_path = Path("/Users/naoto/ドル円/usd_jpy_M1/extracted")
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


class HighWinRateResearch:
    """高勝率戦略リサーチ"""

    def __init__(self):
        self.df = None
        self.results = []

    def load_data(self, timeframe='5min'):
        """データ読み込み"""
        print(f"M1データを読み込み中...")
        df_m1 = load_m1_data()

        print(f"{timeframe}に変換中...")
        self.df = df_m1.resample(timeframe).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

        # インジケーター計算
        for period in [10, 20, 50, 100, 200]:
            self.df[f'ma_{period}'] = self.df['close'].rolling(window=period).mean()

        # ATR計算
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(window=14).mean()

        # RSI計算
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))

        # ボリンジャーバンド
        self.df['bb_mid'] = self.df['close'].rolling(window=20).mean()
        self.df['bb_std'] = self.df['close'].rolling(window=20).std()
        self.df['bb_upper'] = self.df['bb_mid'] + 2 * self.df['bb_std']
        self.df['bb_lower'] = self.df['bb_mid'] - 2 * self.df['bb_std']

        # 時間帯情報
        self.df['hour'] = self.df.index.hour

        print(f"データ: {len(self.df):,}本")
        return self.df

    def strategy_mean_reversion(self, tp_pips, sl_pips, rsi_oversold=30, rsi_overbought=70):
        """
        戦略1: RSI逆張り（高勝率型）
        RSIが売られすぎ/買われすぎで逆張り
        """
        pip_value = 0.01
        spread = 0.3

        trades = []
        in_trade = False
        last_exit_idx = 0

        for idx in range(200, len(self.df) - 500, 12):  # 1時間ごとにチェック
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            rsi = self.df['rsi'].iloc[idx]
            close = self.df['close'].iloc[idx]

            if pd.isna(rsi):
                continue

            direction = None
            if rsi < rsi_oversold:
                direction = 'BUY'
            elif rsi > rsi_overbought:
                direction = 'SELL'

            if direction is None:
                continue

            entry_price = close

            for i in range(idx + 1, min(idx + 2000, len(self.df))):
                current_high = self.df['high'].iloc[i]
                current_low = self.df['low'].iloc[i]

                tp_dist = tp_pips * pip_value
                sl_dist = sl_pips * pip_value

                if direction == 'BUY':
                    tp_price = entry_price + tp_dist
                    sl_price = entry_price - sl_dist
                    if current_high >= tp_price:
                        trades.append({'result': 'WIN', 'pips': tp_pips - spread})
                        last_exit_idx = i
                        in_trade = True
                        break
                    elif current_low <= sl_price:
                        trades.append({'result': 'LOSS', 'pips': -(sl_pips + spread)})
                        last_exit_idx = i
                        in_trade = True
                        break
                else:
                    tp_price = entry_price - tp_dist
                    sl_price = entry_price + sl_dist
                    if current_low <= tp_price:
                        trades.append({'result': 'WIN', 'pips': tp_pips - spread})
                        last_exit_idx = i
                        in_trade = True
                        break
                    elif current_high >= sl_price:
                        trades.append({'result': 'LOSS', 'pips': -(sl_pips + spread)})
                        last_exit_idx = i
                        in_trade = True
                        break

        return self._analyze_trades(trades, 'RSI Mean Reversion')

    def strategy_bollinger_bounce(self, tp_pips, sl_pips):
        """
        戦略2: ボリンジャーバンド反発（高勝率型）
        バンドタッチで逆張り
        """
        pip_value = 0.01
        spread = 0.3

        trades = []
        in_trade = False
        last_exit_idx = 0

        for idx in range(200, len(self.df) - 500, 12):
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            close = self.df['close'].iloc[idx]
            bb_upper = self.df['bb_upper'].iloc[idx]
            bb_lower = self.df['bb_lower'].iloc[idx]

            if pd.isna(bb_upper) or pd.isna(bb_lower):
                continue

            direction = None
            if close <= bb_lower:
                direction = 'BUY'
            elif close >= bb_upper:
                direction = 'SELL'

            if direction is None:
                continue

            entry_price = close

            for i in range(idx + 1, min(idx + 2000, len(self.df))):
                current_high = self.df['high'].iloc[i]
                current_low = self.df['low'].iloc[i]

                tp_dist = tp_pips * pip_value
                sl_dist = sl_pips * pip_value

                if direction == 'BUY':
                    tp_price = entry_price + tp_dist
                    sl_price = entry_price - sl_dist
                    if current_high >= tp_price:
                        trades.append({'result': 'WIN', 'pips': tp_pips - spread})
                        last_exit_idx = i
                        in_trade = True
                        break
                    elif current_low <= sl_price:
                        trades.append({'result': 'LOSS', 'pips': -(sl_pips + spread)})
                        last_exit_idx = i
                        in_trade = True
                        break
                else:
                    tp_price = entry_price - tp_dist
                    sl_price = entry_price + sl_dist
                    if current_low <= tp_price:
                        trades.append({'result': 'WIN', 'pips': tp_pips - spread})
                        last_exit_idx = i
                        in_trade = True
                        break
                    elif current_high >= sl_price:
                        trades.append({'result': 'LOSS', 'pips': -(sl_pips + spread)})
                        last_exit_idx = i
                        in_trade = True
                        break

        return self._analyze_trades(trades, 'Bollinger Bounce')

    def strategy_tokyo_range(self, tp_pips, sl_pips):
        """
        戦略3: 東京時間レンジブレイク
        東京時間(9-15時JST)のレンジを形成し、ブレイクで順張り
        """
        pip_value = 0.01
        spread = 0.3

        trades = []

        # 日付ごとに処理
        dates = self.df.index.date
        unique_dates = pd.unique(dates)

        for date in unique_dates[100:-10]:
            day_data = self.df[self.df.index.date == date]

            # 東京時間（0-6 UTC = 9-15 JST）
            tokyo_data = day_data[(day_data['hour'] >= 0) & (day_data['hour'] < 6)]

            if len(tokyo_data) < 10:
                continue

            tokyo_high = tokyo_data['high'].max()
            tokyo_low = tokyo_data['low'].min()
            tokyo_range = tokyo_high - tokyo_low

            if tokyo_range < 0.1 or tokyo_range > 0.5:  # レンジが小さすぎ/大きすぎは除外
                continue

            # ロンドン時間（6-14 UTC）でブレイクを待つ
            london_data = day_data[(day_data['hour'] >= 6) & (day_data['hour'] < 14)]

            for idx in range(len(london_data)):
                row = london_data.iloc[idx]
                close = row['close']

                direction = None
                if close > tokyo_high + 0.05:
                    direction = 'BUY'
                    entry_price = close
                elif close < tokyo_low - 0.05:
                    direction = 'SELL'
                    entry_price = close

                if direction is None:
                    continue

                # 決済チェック
                remaining = london_data.iloc[idx+1:] if idx+1 < len(london_data) else pd.DataFrame()

                for i in range(len(remaining)):
                    current_high = remaining.iloc[i]['high']
                    current_low = remaining.iloc[i]['low']

                    tp_dist = tp_pips * pip_value
                    sl_dist = sl_pips * pip_value

                    if direction == 'BUY':
                        if current_high >= entry_price + tp_dist:
                            trades.append({'result': 'WIN', 'pips': tp_pips - spread})
                            break
                        elif current_low <= entry_price - sl_dist:
                            trades.append({'result': 'LOSS', 'pips': -(sl_pips + spread)})
                            break
                    else:
                        if current_low <= entry_price - tp_dist:
                            trades.append({'result': 'WIN', 'pips': tp_pips - spread})
                            break
                        elif current_high >= entry_price + sl_dist:
                            trades.append({'result': 'LOSS', 'pips': -(sl_pips + spread)})
                            break

                break  # 1日1トレード

        return self._analyze_trades(trades, 'Tokyo Range Breakout')

    def strategy_scalping_high_winrate(self, tp_pips, sl_pips):
        """
        戦略4: 高勝率スキャルピング
        小さいTP、大きいSL（コツコツドカン型だが勝率は高い）
        複合条件エントリー
        """
        pip_value = 0.01
        spread = 0.3

        trades = []
        in_trade = False
        last_exit_idx = 0

        for idx in range(200, len(self.df) - 500, 6):  # 30分ごとにチェック
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            close = self.df['close'].iloc[idx]
            ma20 = self.df['ma_20'].iloc[idx]
            ma50 = self.df['ma_50'].iloc[idx]
            rsi = self.df['rsi'].iloc[idx]
            bb_upper = self.df['bb_upper'].iloc[idx]
            bb_lower = self.df['bb_lower'].iloc[idx]

            if pd.isna(ma20) or pd.isna(ma50) or pd.isna(rsi):
                continue

            # 複合条件
            direction = None

            # BUY条件: RSI < 40 AND close < MA20 AND close近くbb_lower
            if rsi < 40 and close < ma20 and close < bb_lower + (bb_upper - bb_lower) * 0.2:
                direction = 'BUY'
            # SELL条件: RSI > 60 AND close > MA20 AND close近くbb_upper
            elif rsi > 60 and close > ma20 and close > bb_upper - (bb_upper - bb_lower) * 0.2:
                direction = 'SELL'

            if direction is None:
                continue

            entry_price = close

            for i in range(idx + 1, min(idx + 1000, len(self.df))):
                current_high = self.df['high'].iloc[i]
                current_low = self.df['low'].iloc[i]

                tp_dist = tp_pips * pip_value
                sl_dist = sl_pips * pip_value

                if direction == 'BUY':
                    tp_price = entry_price + tp_dist
                    sl_price = entry_price - sl_dist
                    if current_high >= tp_price:
                        trades.append({'result': 'WIN', 'pips': tp_pips - spread})
                        last_exit_idx = i
                        in_trade = True
                        break
                    elif current_low <= sl_price:
                        trades.append({'result': 'LOSS', 'pips': -(sl_pips + spread)})
                        last_exit_idx = i
                        in_trade = True
                        break
                else:
                    tp_price = entry_price - tp_dist
                    sl_price = entry_price + sl_dist
                    if current_low <= tp_price:
                        trades.append({'result': 'WIN', 'pips': tp_pips - spread})
                        last_exit_idx = i
                        in_trade = True
                        break
                    elif current_high >= sl_price:
                        trades.append({'result': 'LOSS', 'pips': -(sl_pips + spread)})
                        last_exit_idx = i
                        in_trade = True
                        break

        return self._analyze_trades(trades, 'High WR Scalping')

    def _analyze_trades(self, trades, strategy_name):
        """トレード分析"""
        if not trades:
            return None

        df_trades = pd.DataFrame(trades)
        total = len(df_trades)
        wins = (df_trades['result'] == 'WIN').sum()
        win_rate = wins / total * 100
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

        # 最大連敗
        max_consecutive_losses = 0
        current_streak = 0
        for result in df_trades['result']:
            if result == 'LOSS':
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
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
            'strategy': strategy_name,
            'trades': total,
            'win_rate': round(win_rate, 1),
            'total_pips': round(total_pips, 1),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'pf': round(pf, 2),
            'max_dd': round(max_dd, 1)
        }

    def run_research(self):
        """戦略リサーチ実行"""
        if self.df is None:
            self.load_data()

        print("\n高勝率戦略リサーチ開始...")
        print("=" * 80)

        self.results = []

        # 戦略1: RSI逆張り（様々なTP/SL組み合わせ）
        print("\n【戦略1】RSI Mean Reversion")
        for tp in [5, 10, 15, 20]:
            for sl in [30, 50, 70, 100]:
                result = self.strategy_mean_reversion(tp, sl)
                if result:
                    result['tp'] = tp
                    result['sl'] = sl
                    self.results.append(result)
                    if result['win_rate'] >= 70:
                        print(f"  TP{tp}/SL{sl}: WR={result['win_rate']}%, Trades={result['trades']}, "
                              f"ConsecWins={result['max_consecutive_wins']}, Pips={result['total_pips']}")

        # 戦略2: ボリンジャー反発
        print("\n【戦略2】Bollinger Bounce")
        for tp in [5, 10, 15, 20]:
            for sl in [30, 50, 70, 100]:
                result = self.strategy_bollinger_bounce(tp, sl)
                if result:
                    result['tp'] = tp
                    result['sl'] = sl
                    self.results.append(result)
                    if result['win_rate'] >= 70:
                        print(f"  TP{tp}/SL{sl}: WR={result['win_rate']}%, Trades={result['trades']}, "
                              f"ConsecWins={result['max_consecutive_wins']}, Pips={result['total_pips']}")

        # 戦略3: 東京レンジブレイク
        print("\n【戦略3】Tokyo Range Breakout")
        for tp in [10, 15, 20, 30]:
            for sl in [20, 30, 50]:
                result = self.strategy_tokyo_range(tp, sl)
                if result:
                    result['tp'] = tp
                    result['sl'] = sl
                    self.results.append(result)
                    if result['win_rate'] >= 50:
                        print(f"  TP{tp}/SL{sl}: WR={result['win_rate']}%, Trades={result['trades']}, "
                              f"ConsecWins={result['max_consecutive_wins']}, Pips={result['total_pips']}")

        # 戦略4: 高勝率スキャルピング
        print("\n【戦略4】High WR Scalping")
        for tp in [5, 8, 10, 15]:
            for sl in [40, 60, 80, 100]:
                result = self.strategy_scalping_high_winrate(tp, sl)
                if result:
                    result['tp'] = tp
                    result['sl'] = sl
                    self.results.append(result)
                    if result['win_rate'] >= 70:
                        print(f"  TP{tp}/SL{sl}: WR={result['win_rate']}%, Trades={result['trades']}, "
                              f"ConsecWins={result['max_consecutive_wins']}, Pips={result['total_pips']}")

        return self.results

    def print_best_results(self):
        """最良結果表示"""
        if not self.results:
            print("結果がありません")
            return

        df = pd.DataFrame(self.results)

        print("\n" + "=" * 100)
        print("  高勝率戦略リサーチ結果")
        print("=" * 100)

        # 勝率順
        print("\n【勝率 Top 10】")
        top_wr = df.nlargest(10, 'win_rate')
        print(f"{'Strategy':<25}{'TP':<5}{'SL':<5}{'WR%':<8}{'Trades':<8}{'ConsecW':<10}{'Pips':<12}{'PF':<8}")
        print("-" * 90)
        for _, row in top_wr.iterrows():
            print(f"{row['strategy']:<25}{row['tp']:<5}{row['sl']:<5}{row['win_rate']:<8}"
                  f"{row['trades']:<8}{row['max_consecutive_wins']:<10}{row['total_pips']:<12.0f}{row['pf']:<8.2f}")

        # 連勝記録順
        print("\n【連勝記録 Top 10】")
        top_consec = df.nlargest(10, 'max_consecutive_wins')
        print(f"{'Strategy':<25}{'TP':<5}{'SL':<5}{'WR%':<8}{'Trades':<8}{'ConsecW':<10}{'Pips':<12}{'PF':<8}")
        print("-" * 90)
        for _, row in top_consec.iterrows():
            print(f"{row['strategy']:<25}{row['tp']:<5}{row['sl']:<5}{row['win_rate']:<8}"
                  f"{row['trades']:<8}{row['max_consecutive_wins']:<10}{row['total_pips']:<12.0f}{row['pf']:<8.2f}")

        # 総合（勝率 >= 70% かつ Pips > 0）
        print("\n【総合評価 (勝率>=70% AND 利益>0)】")
        good = df[(df['win_rate'] >= 70) & (df['total_pips'] > 0)]
        if len(good) > 0:
            good = good.sort_values(['win_rate', 'total_pips'], ascending=[False, False])
            print(f"{'Strategy':<25}{'TP':<5}{'SL':<5}{'WR%':<8}{'Trades':<8}{'ConsecW':<10}{'Pips':<12}{'PF':<8}")
            print("-" * 90)
            for _, row in good.head(15).iterrows():
                print(f"{row['strategy']:<25}{row['tp']:<5}{row['sl']:<5}{row['win_rate']:<8}"
                      f"{row['trades']:<8}{row['max_consecutive_wins']:<10}{row['total_pips']:<12.0f}{row['pf']:<8.2f}")
        else:
            print("  該当なし")

        return df

    def visualize_results(self, save_path):
        """結果可視化"""
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('High Win Rate Strategy Research - USDJPY', fontsize=14, fontweight='bold')

        # 1. 勝率 vs 総利益
        ax1 = axes[0, 0]
        for strategy in df['strategy'].unique():
            subset = df[df['strategy'] == strategy]
            ax1.scatter(subset['win_rate'], subset['total_pips'],
                       label=strategy, alpha=0.7, s=50)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(x=70, color='red', linestyle='--', alpha=0.5, label='70% WR Line')
        ax1.set_xlabel('Win Rate (%)')
        ax1.set_ylabel('Total Pips')
        ax1.set_title('Win Rate vs Total Pips', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 勝率 vs 連勝記録
        ax2 = axes[0, 1]
        for strategy in df['strategy'].unique():
            subset = df[df['strategy'] == strategy]
            ax2.scatter(subset['win_rate'], subset['max_consecutive_wins'],
                       label=strategy, alpha=0.7, s=50)
        ax2.set_xlabel('Win Rate (%)')
        ax2.set_ylabel('Max Consecutive Wins')
        ax2.set_title('Win Rate vs Consecutive Wins', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 戦略別勝率分布
        ax3 = axes[1, 0]
        strategy_wr = df.groupby('strategy')['win_rate'].mean()
        bars = ax3.bar(range(len(strategy_wr)), strategy_wr.values, alpha=0.7)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.set_xticks(range(len(strategy_wr)))
        ax3.set_xticklabels(strategy_wr.index, rotation=15, ha='right')
        ax3.set_ylabel('Average Win Rate (%)')
        ax3.set_title('Average Win Rate by Strategy', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. 最良結果サマリー
        ax4 = axes[1, 1]
        ax4.axis('off')

        best_wr = df.loc[df['win_rate'].idxmax()]
        best_consec = df.loc[df['max_consecutive_wins'].idxmax()]
        best_pips = df.loc[df['total_pips'].idxmax()]

        summary = f"""
╔════════════════════════════════════════════════════════════════╗
║                    BEST RESULTS SUMMARY                        ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  【最高勝率】                                                   ║
║    {best_wr['strategy']} (TP{best_wr['tp']}/SL{best_wr['sl']})
║    勝率: {best_wr['win_rate']}%, 連勝: {best_wr['max_consecutive_wins']}, Pips: {best_wr['total_pips']:+.0f}
║                                                                ║
║  【最長連勝】                                                   ║
║    {best_consec['strategy']} (TP{best_consec['tp']}/SL{best_consec['sl']})
║    連勝: {best_consec['max_consecutive_wins']}, 勝率: {best_consec['win_rate']}%, Pips: {best_consec['total_pips']:+.0f}
║                                                                ║
║  【最高利益】                                                   ║
║    {best_pips['strategy']} (TP{best_pips['tp']}/SL{best_pips['sl']})
║    Pips: {best_pips['total_pips']:+.0f}, 勝率: {best_pips['win_rate']}%
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
        ax4.text(0.05, 0.5, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n保存: {save_path}")


def main():
    print("=" * 70)
    print("  ゴゴジャン向け 高勝率戦略リサーチ")
    print("  USDJPY M5 (Axiory 2015-2025)")
    print("=" * 70)

    research = HighWinRateResearch()
    research.load_data('5min')
    research.run_research()
    df = research.print_best_results()
    research.visualize_results('/Users/naoto/ドル円/gogojungle_strategy/high_winrate_research.png')

    # CSV保存
    df.to_csv('/Users/naoto/ドル円/gogojungle_strategy/research_results.csv', index=False)
    print("\n保存: /Users/naoto/ドル円/gogojungle_strategy/research_results.csv")

    print("\n完了!")


if __name__ == "__main__":
    main()
