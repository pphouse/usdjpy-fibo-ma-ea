"""
バックテスト実行スクリプト（サンプルデータ版）
実際のMT5がなくても動作します
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # GUIなしバックエンド
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# フォント設定（日本語なし）
plt.rcParams['axes.unicode_minus'] = False


class FiboMABacktester:
    """フィボナッチ + MA バックテスター"""

    def __init__(self, symbol="USDJPY"):
        self.symbol = symbol
        self.trades = []

        # パラメータ設定（ここで手動変更可能）
        self.params = {
            'fibo_lookback': 50,
            'fibo_tolerance': 0.002,
            'ma_periods': [20, 50, 100],
            'ma_tolerance': 0.001,
            'take_profit_pips': 50,   # ★ TP設定（変更可能）
            'stop_loss_pips': 50,     # ★ SL設定（変更可能）
            'use_trend_filter': True,
        }

    def set_tp_sl(self, tp_pips, sl_pips):
        """TP/SLを設定"""
        self.params['take_profit_pips'] = tp_pips
        self.params['stop_loss_pips'] = sl_pips
        print(f"設定: TP={tp_pips}pips, SL={sl_pips}pips")

    def generate_realistic_data(self, days=365):
        """リアルなUSDJPYデータを生成"""
        print("過去1年のデータを生成中...")

        np.random.seed(42)

        # 4時間足のデータ生成（1日6本 x 365日）
        n_bars = days * 6
        dates = pd.date_range(
            end=datetime.now(),
            periods=n_bars,
            freq='4h'
        )

        # トレンドを含むランダムウォーク
        # USDJPYの典型的なボラティリティを再現
        base_price = 145.0
        volatility = 0.003  # 4時間あたりの変動率

        # トレンド成分を追加
        trend = np.sin(np.linspace(0, 4*np.pi, n_bars)) * 5  # ±5円の波
        noise = np.cumsum(np.random.normal(0, volatility * base_price, n_bars))

        prices = base_price + trend + noise

        # OHLC生成
        df = pd.DataFrame(index=dates)
        df['close'] = prices

        # High/Lowはclose周辺に
        df['high'] = df['close'] + np.abs(np.random.normal(0, 0.2, n_bars))
        df['low'] = df['close'] - np.abs(np.random.normal(0, 0.2, n_bars))
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])

        # high >= max(open, close), low <= min(open, close) を保証
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)

        print(f"データ生成完了: {len(df)}本のバー")
        print(f"期間: {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"価格レンジ: {df['low'].min():.2f} ~ {df['high'].max():.2f}")

        return df

    def calculate_ma(self, df, period):
        return df['close'].rolling(window=period).mean()

    def calculate_fibonacci(self, df, idx, lookback):
        if idx < lookback:
            return None, None, 0, None, None

        window = df.iloc[idx-lookback:idx]
        high = window['high'].max()
        low = window['low'].min()
        high_idx = window['high'].idxmax()
        low_idx = window['low'].idxmin()

        range_val = high - low
        if range_val < 0.5:  # 0.5円未満はスキップ
            return None, None, 0, None, None

        if high_idx > low_idx:  # 上昇トレンド
            trend = 1
            fibo_382 = high - range_val * 0.382
            fibo_618 = high - range_val * 0.618
        else:  # 下降トレンド
            trend = -1
            fibo_382 = low + range_val * 0.382
            fibo_618 = low + range_val * 0.618

        return fibo_382, fibo_618, trend, high, low

    def check_entry(self, df, idx, fibo_382, fibo_618, trend):
        close = df['close'].iloc[idx]
        fibo_tol = close * self.params['fibo_tolerance']

        # フィボナッチ条件
        fibo_hit = None
        if abs(close - fibo_382) < fibo_tol:
            fibo_hit = "38.2%"
        elif abs(close - fibo_618) < fibo_tol:
            fibo_hit = "61.8%"

        if fibo_hit is None:
            return None

        # MA条件
        ma_tol = close * self.params['ma_tolerance']
        ma_hit = None

        for period in self.params['ma_periods']:
            ma_col = f'ma_{period}'
            if ma_col in df.columns:
                ma_val = df[ma_col].iloc[idx]
                if pd.notna(ma_val) and abs(close - ma_val) < ma_tol:
                    ma_hit = period
                    break

        if ma_hit is None:
            return None

        # トレンドフィルター
        if self.params['use_trend_filter'] and 'ma_200' in df.columns:
            trend_ma = df['ma_200'].iloc[idx]
            if pd.notna(trend_ma):
                if trend == 1 and close < trend_ma:
                    return None
                if trend == -1 and close > trend_ma:
                    return None

        return {
            'direction': 'BUY' if trend == 1 else 'SELL',
            'fibo_level': fibo_hit,
            'ma_period': ma_hit,
            'entry_price': close
        }

    def simulate_trade(self, df, entry_idx, entry_info):
        entry_price = entry_info['entry_price']
        direction = entry_info['direction']
        pip_value = 0.01  # USDJPYのpip値

        tp_dist = self.params['take_profit_pips'] * pip_value
        sl_dist = self.params['stop_loss_pips'] * pip_value

        if direction == 'BUY':
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
        else:
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist

        for i in range(entry_idx + 1, len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            if direction == 'BUY':
                if high >= tp_price:
                    return {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': tp_price,
                        'pips': self.params['take_profit_pips'],
                        'result': 'WIN',
                        'fibo_level': entry_info['fibo_level'],
                        'ma_period': entry_info['ma_period']
                    }
                elif low <= sl_price:
                    return {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': sl_price,
                        'pips': -self.params['stop_loss_pips'],
                        'result': 'LOSS',
                        'fibo_level': entry_info['fibo_level'],
                        'ma_period': entry_info['ma_period']
                    }
            else:
                if low <= tp_price:
                    return {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': tp_price,
                        'pips': self.params['take_profit_pips'],
                        'result': 'WIN',
                        'fibo_level': entry_info['fibo_level'],
                        'ma_period': entry_info['ma_period']
                    }
                elif high >= sl_price:
                    return {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': sl_price,
                        'pips': -self.params['stop_loss_pips'],
                        'result': 'LOSS',
                        'fibo_level': entry_info['fibo_level'],
                        'ma_period': entry_info['ma_period']
                    }

        return None

    def run_backtest(self, df=None):
        """バックテスト実行"""
        if df is None:
            df = self.generate_realistic_data()

        print("\nバックテスト実行中...")

        # MA計算
        for period in self.params['ma_periods']:
            df[f'ma_{period}'] = self.calculate_ma(df, period)
        df['ma_200'] = self.calculate_ma(df, 200)

        self.trades = []
        self.df = df
        lookback = self.params['fibo_lookback']

        in_trade = False
        last_exit_idx = 0

        for idx in range(lookback + 200, len(df) - 50):
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            fibo_382, fibo_618, trend, _, _ = self.calculate_fibonacci(df, idx, lookback)
            if trend == 0:
                continue

            entry = self.check_entry(df, idx, fibo_382, fibo_618, trend)
            if entry is None:
                continue

            result = self.simulate_trade(df, idx, entry)
            if result:
                self.trades.append(result)
                in_trade = True
                last_exit_idx = df.index.get_loc(result['exit_time'])

        print(f"バックテスト完了: {len(self.trades)}トレード")
        return self.trades

    def get_statistics(self):
        """統計を計算"""
        if not self.trades:
            return None

        df = pd.DataFrame(self.trades)

        total = len(df)
        wins = len(df[df['result'] == 'WIN'])
        losses = len(df[df['result'] == 'LOSS'])
        win_rate = wins / total * 100 if total > 0 else 0

        total_pips = df['pips'].sum()
        avg_pips = df['pips'].mean()

        cumulative = df['pips'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_dd = drawdown.min()

        win_pips = df[df['pips'] > 0]['pips'].sum()
        loss_pips = abs(df[df['pips'] < 0]['pips'].sum())
        pf = win_pips / loss_pips if loss_pips > 0 else float('inf')

        # 連勝・連敗
        results = df['result'].values
        max_wins = max_losses = current_wins = current_losses = 0
        for r in results:
            if r == 'WIN':
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'total_pips': round(total_pips, 1),
            'avg_pips': round(avg_pips, 2),
            'max_drawdown': round(max_dd, 1),
            'profit_factor': round(pf, 2),
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses
        }

    def print_results(self):
        """結果を表示"""
        stats = self.get_statistics()
        if stats is None:
            print("トレードがありません")
            return

        print("\n" + "="*60)
        print("          バックテスト結果サマリー")
        print("="*60)
        print(f"  通貨ペア: {self.symbol}")
        print(f"  時間足: 4時間足 (H4)")
        print(f"  TP: {self.params['take_profit_pips']} pips")
        print(f"  SL: {self.params['stop_loss_pips']} pips")
        print("-"*60)
        print(f"  総トレード数: {stats['total_trades']}")
        print(f"  勝ちトレード: {stats['wins']}")
        print(f"  負けトレード: {stats['losses']}")
        print(f"  勝率: {stats['win_rate']}%")
        print("-"*60)
        print(f"  合計損益: {stats['total_pips']} pips")
        print(f"  平均損益: {stats['avg_pips']} pips/トレード")
        print(f"  最大ドローダウン: {stats['max_drawdown']} pips")
        print(f"  プロフィットファクター: {stats['profit_factor']}")
        print("-"*60)
        print(f"  最大連勝: {stats['max_consecutive_wins']}")
        print(f"  最大連敗: {stats['max_consecutive_losses']}")
        print("="*60)

    def visualize(self, save_path=None):
        """結果をビジュアル化"""
        if not self.trades:
            print("トレードがありません")
            return

        df_trades = pd.DataFrame(self.trades)

        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(f'Fibonacci + MA EA Backtest Results\n'
                     f'TP={self.params["take_profit_pips"]}pips, '
                     f'SL={self.params["stop_loss_pips"]}pips',
                     fontsize=16, fontweight='bold')

        # 1. 累積損益
        ax1 = fig.add_subplot(3, 2, 1)
        cumulative = df_trades['pips'].cumsum()
        ax1.plot(df_trades['entry_time'], cumulative, 'b-', linewidth=2)
        ax1.fill_between(df_trades['entry_time'], cumulative, 0,
                        where=(cumulative >= 0), color='green', alpha=0.3)
        ax1.fill_between(df_trades['entry_time'], cumulative, 0,
                        where=(cumulative < 0), color='red', alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_title('Cumulative Pips', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Pips')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. 勝率
        ax2 = fig.add_subplot(3, 2, 2)
        wins = len(df_trades[df_trades['result'] == 'WIN'])
        losses = len(df_trades[df_trades['result'] == 'LOSS'])
        colors = ['#2ecc71', '#e74c3c']
        ax2.pie([wins, losses], labels=['Win', 'Loss'], colors=colors,
                autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05))
        ax2.set_title(f'Win Rate ({wins}W / {losses}L)', fontsize=12, fontweight='bold')

        # 3. 月別パフォーマンス
        ax3 = fig.add_subplot(3, 2, 3)
        df_trades['month'] = pd.to_datetime(df_trades['entry_time']).dt.to_period('M')
        monthly = df_trades.groupby('month')['pips'].sum()
        colors = ['green' if x > 0 else 'red' for x in monthly.values]
        ax3.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(monthly)))
        ax3.set_xticklabels([str(m) for m in monthly.index], rotation=45)
        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.set_title('Monthly Performance (Pips)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Pips')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. フィボナッチレベル別
        ax4 = fig.add_subplot(3, 2, 4)
        fibo_stats = df_trades.groupby('fibo_level')['pips'].agg(['sum', 'count'])
        x = range(len(fibo_stats))
        bars = ax4.bar(x, fibo_stats['sum'], color=['#f39c12', '#e67e22'])
        ax4.set_xticks(x)
        ax4.set_xticklabels(fibo_stats.index)
        ax4.set_title('Performance by Fibonacci Level', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Pips')
        for bar, count in zip(bars, fibo_stats['count']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'n={int(count)}', ha='center', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. MA期間別
        ax5 = fig.add_subplot(3, 2, 5)
        ma_stats = df_trades.groupby('ma_period')['pips'].agg(['sum', 'count'])
        x = range(len(ma_stats))
        colors = ['#3498db', '#9b59b6', '#1abc9c'][:len(ma_stats)]
        bars = ax5.bar(x, ma_stats['sum'], color=colors)
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'MA{int(p)}' for p in ma_stats.index])
        ax5.set_title('Performance by MA Period', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Pips')
        for bar, count in zip(bars, ma_stats['count']):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'n={int(count)}', ha='center', fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. ドローダウン
        ax6 = fig.add_subplot(3, 2, 6)
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        ax6.fill_between(df_trades['entry_time'], drawdown, 0, color='red', alpha=0.4)
        ax6.plot(df_trades['entry_time'], drawdown, 'r-', linewidth=1)
        ax6.set_title('Drawdown (Pips)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Pips')
        ax6.grid(True, alpha=0.3)
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nグラフを保存しました: {save_path}")

        plt.close()

    def export_trades(self, filepath):
        """トレード履歴をエクスポート"""
        if not self.trades:
            return
        df = pd.DataFrame(self.trades)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"トレード履歴を保存しました: {filepath}")


def main():
    """メイン関数"""
    print("="*60)
    print("    Fibonacci + MA EA Backtest")
    print("    USDJPY 4H - Past 1 Year")
    print("="*60)

    #===========================================
    # パラメータ設定（ここで変更可能）
    #===========================================
    TP_PIPS = 50   # 利確 pips
    SL_PIPS = 50   # 損切 pips
    #===========================================

    bt = FiboMABacktester("USDJPY")
    bt.set_tp_sl(TP_PIPS, SL_PIPS)

    # バックテスト実行
    bt.run_backtest()

    # 結果表示
    bt.print_results()

    # ビジュアル化
    bt.visualize(save_path='/Users/naoto/ドル円/backtest_results.png')

    # トレード履歴エクスポート
    bt.export_trades('/Users/naoto/ドル円/trade_history.csv')

    print("\n完了!")


if __name__ == "__main__":
    main()
