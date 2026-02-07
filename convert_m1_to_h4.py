"""
Axiory M1データをH4に変換
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

def load_all_m1_data():
    """全M1データを読み込み"""
    base_path = Path("/Users/naoto/ドル円/usd_jpy_M1/extracted")

    all_files = []

    # USDJPYフォルダ内 (2015-2018)
    usdjpy_folder = base_path / "USDJPY"
    if usdjpy_folder.exists():
        for f in usdjpy_folder.glob("USDJPY_20*_*.csv"):
            if "_all" not in f.name:
                all_files.append(f)

    # ルートレベル (2019-2025)
    for f in base_path.glob("USDJPY_20*_*.csv"):
        if "_all" not in f.name:
            all_files.append(f)

    print(f"見つかったファイル: {len(all_files)}")

    # 全ファイル読み込み
    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f, header=None,
                           names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
            dfs.append(df)
            print(f"  {f.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  {f.name}: ERROR - {e}")

    # 結合
    df_all = pd.concat(dfs, ignore_index=True)

    # 日時変換
    df_all['datetime'] = pd.to_datetime(df_all['date'] + ' ' + df_all['time'],
                                        format='%Y.%m.%d %H:%M')
    df_all = df_all.sort_values('datetime').reset_index(drop=True)
    df_all = df_all.set_index('datetime')

    # 重複削除
    df_all = df_all[~df_all.index.duplicated(keep='first')]

    print(f"\n合計: {len(df_all):,} rows")
    print(f"期間: {df_all.index[0]} ~ {df_all.index[-1]}")
    print(f"価格レンジ: {df_all['low'].min():.3f} ~ {df_all['high'].max():.3f}")

    return df_all[['open', 'high', 'low', 'close', 'volume']]


def resample_to_h4(df_m1):
    """M1をH4に変換"""
    print("\nH4に変換中...")

    df_h4 = df_m1.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"H4データ: {len(df_h4):,} bars")
    print(f"期間: {df_h4.index[0]} ~ {df_h4.index[-1]}")

    return df_h4


def main():
    print("="*60)
    print("  Axiory M1 → H4 変換")
    print("="*60)

    # M1読み込み
    df_m1 = load_all_m1_data()

    # H4変換
    df_h4 = resample_to_h4(df_m1)

    # 保存
    output_path = "/Users/naoto/ドル円/USDJPY_H4_2015_2025.csv"
    df_h4.to_csv(output_path)
    print(f"\n保存しました: {output_path}")

    # 確認
    print("\n【データサンプル】")
    print(df_h4.head(10))
    print("\n...")
    print(df_h4.tail(5))

    return df_h4


if __name__ == "__main__":
    main()
