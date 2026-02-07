//+------------------------------------------------------------------+
//|                                            FiboMA_EA_Nanpin.mq5 |
//|        フィボナッチ + 移動平均線 + ナンピン エントリーEA          |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "FiboMA Nanpin EA"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>

//--- 入力パラメータ
input group "=== エントリー設定 ==="
input ENUM_TIMEFRAMES TimeFrame = PERIOD_H4;           // 時間足（H4 or D1）
input int    FiboLookbackBars = 50;                    // フィボナッチ計算用の過去バー数
input double FiboLevel1 = 38.2;                        // フィボナッチレベル1 (%)
input double FiboLevel2 = 61.8;                        // フィボナッチレベル2 (%)

input group "=== 条件許容設定 ==="
input double OverlapTolerance_Pips = 10.0;             // Fibo/MA重なり許容誤差 (Pips)

input group "=== 移動平均線設定 ==="
input int    MA_Period1 = 20;                          // MA期間1
input int    MA_Period2 = 50;                          // MA期間2
input int    MA_Period3 = 100;                         // MA期間3
input ENUM_MA_METHOD MA_Method = MODE_SMA;             // MA計算方法

input group "=== 決済設定 ==="
input double TakeProfit_Pips = 50.0;                   // 利確 (Pips) - 平均建値から
input double StopLoss_Pips = 50.0;                     // 損切り (Pips) - 平均建値から

input group "=== ナンピン設定 ==="
input double NanpinInterval_Pips = 30.0;               // ナンピン間隔 (Pips)
input double Lot1 = 0.1;                               // 初回ロット (比率1)
input double Lot2 = 0.3;                               // 2回目ロット (比率3)
input double Lot3 = 0.3;                               // 3回目ロット (比率3)
input double Lot4 = 0.5;                               // 4回目ロット (比率5)
input int    MaxNanpinCount = 3;                       // 最大ナンピン回数 (1-3)

input group "=== その他設定 ==="
input int    MagicNumber = 123456;                     // マジックナンバー
input bool   UseTrendFilter = true;                    // トレンドフィルター使用
input int    TrendMA_Period = 200;                     // トレンド判定用MA期間

//--- グローバル変数
CTrade trade;
int ma1_handle, ma2_handle, ma3_handle, trend_ma_handle;
double pip_value;

// ナンピン管理用
int current_nanpin_count = 0;
double average_entry_price = 0;
double total_lots = 0;
int current_direction = 0;  // 1=BUY, -1=SELL, 0=なし
double last_entry_price = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- PIP値の計算
    pip_value = _Point;
    if(_Digits == 3 || _Digits == 5)
        pip_value = _Point * 10;

    //--- MAハンドル作成
    ma1_handle = iMA(_Symbol, TimeFrame, MA_Period1, 0, MA_Method, PRICE_CLOSE);
    ma2_handle = iMA(_Symbol, TimeFrame, MA_Period2, 0, MA_Method, PRICE_CLOSE);
    ma3_handle = iMA(_Symbol, TimeFrame, MA_Period3, 0, MA_Method, PRICE_CLOSE);

    if(UseTrendFilter)
        trend_ma_handle = iMA(_Symbol, TimeFrame, TrendMA_Period, 0, MA_Method, PRICE_CLOSE);

    if(ma1_handle == INVALID_HANDLE || ma2_handle == INVALID_HANDLE || ma3_handle == INVALID_HANDLE)
    {
        Print("MAハンドル作成エラー");
        return(INIT_FAILED);
    }

    //--- トレード設定
    trade.SetExpertMagicNumber(MagicNumber);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    Print("=== FiboMA Nanpin EA 初期化完了 ===");
    Print("時間足: ", EnumToString(TimeFrame));
    Print("TP: ", TakeProfit_Pips, " pips, SL: ", StopLoss_Pips, " pips");
    Print("ナンピン間隔: ", NanpinInterval_Pips, " pips");
    Print("ロット比率: ", Lot1, ":", Lot2, ":", Lot3, ":", Lot4);
    Print("Fibo/MA許容誤差: ±", OverlapTolerance_Pips, " pips");

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    IndicatorRelease(ma1_handle);
    IndicatorRelease(ma2_handle);
    IndicatorRelease(ma3_handle);
    if(UseTrendFilter)
        IndicatorRelease(trend_ma_handle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- ポジション状態を更新
    UpdatePositionStatus();

    //--- 既存ポジションがある場合
    if(current_direction != 0)
    {
        //--- ナンピンチェック
        CheckNanpin();

        //--- TP/SL更新（平均建値ベース）
        UpdateTPSL();
        return;
    }

    //--- 新しいバーでのみ新規エントリーチェック
    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(_Symbol, TimeFrame, 0);

    if(last_bar_time == current_bar_time)
        return;
    last_bar_time = current_bar_time;

    //--- 新規エントリーチェック
    CheckNewEntry();
}

//+------------------------------------------------------------------+
//| ポジション状態を更新                                              |
//+------------------------------------------------------------------+
void UpdatePositionStatus()
{
    double total_volume = 0;
    double weighted_price = 0;
    int direction = 0;
    int position_count = 0;

    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
               PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                double volume = PositionGetDouble(POSITION_VOLUME);
                double price = PositionGetDouble(POSITION_PRICE_OPEN);
                int pos_type = (int)PositionGetInteger(POSITION_TYPE);

                total_volume += volume;
                weighted_price += price * volume;
                direction = (pos_type == POSITION_TYPE_BUY) ? 1 : -1;
                position_count++;
            }
        }
    }

    if(position_count > 0)
    {
        current_direction = direction;
        total_lots = total_volume;
        average_entry_price = weighted_price / total_volume;
        current_nanpin_count = position_count - 1;  // 初回を除く
    }
    else
    {
        // ポジションなし - リセット
        current_direction = 0;
        total_lots = 0;
        average_entry_price = 0;
        current_nanpin_count = 0;
        last_entry_price = 0;
    }
}

//+------------------------------------------------------------------+
//| ナンピンチェック                                                  |
//+------------------------------------------------------------------+
void CheckNanpin()
{
    if(current_nanpin_count >= MaxNanpinCount)
        return;

    double current_price = (current_direction == 1) ?
                           SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                           SymbolInfoDouble(_Symbol, SYMBOL_BID);

    double distance_pips;

    if(current_direction == 1)  // BUY
    {
        distance_pips = (last_entry_price - current_price) / pip_value;
    }
    else  // SELL
    {
        distance_pips = (current_price - last_entry_price) / pip_value;
    }

    // ナンピン間隔に達したらナンピン
    if(distance_pips >= NanpinInterval_Pips)
    {
        double lot = GetNanpinLot(current_nanpin_count + 1);

        if(current_direction == 1)
        {
            double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            if(trade.Buy(lot, _Symbol, ask, 0, 0, "Nanpin BUY #" + IntegerToString(current_nanpin_count + 2)))
            {
                last_entry_price = ask;
                Print("ナンピンBUY #", current_nanpin_count + 2, " ロット: ", lot, " 価格: ", ask);
                DrawEntryArrow(true, ask, current_nanpin_count + 2);
            }
        }
        else
        {
            double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            if(trade.Sell(lot, _Symbol, bid, 0, 0, "Nanpin SELL #" + IntegerToString(current_nanpin_count + 2)))
            {
                last_entry_price = bid;
                Print("ナンピンSELL #", current_nanpin_count + 2, " ロット: ", lot, " 価格: ", bid);
                DrawEntryArrow(false, bid, current_nanpin_count + 2);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| ナンピン回数に応じたロット取得                                    |
//+------------------------------------------------------------------+
double GetNanpinLot(int nanpin_number)
{
    switch(nanpin_number)
    {
        case 0: return Lot1;  // 初回
        case 1: return Lot2;  // 1回目ナンピン
        case 2: return Lot3;  // 2回目ナンピン
        case 3: return Lot4;  // 3回目ナンピン
        default: return Lot4;
    }
}

//+------------------------------------------------------------------+
//| TP/SL更新（平均建値ベース）                                       |
//+------------------------------------------------------------------+
void UpdateTPSL()
{
    if(average_entry_price == 0)
        return;

    double new_tp, new_sl;

    if(current_direction == 1)  // BUY
    {
        new_tp = average_entry_price + TakeProfit_Pips * pip_value;
        new_sl = average_entry_price - StopLoss_Pips * pip_value;
    }
    else  // SELL
    {
        new_tp = average_entry_price - TakeProfit_Pips * pip_value;
        new_sl = average_entry_price + StopLoss_Pips * pip_value;
    }

    // すべてのポジションのTP/SLを更新
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket))
        {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
               PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                double current_tp = PositionGetDouble(POSITION_TP);
                double current_sl = PositionGetDouble(POSITION_SL);

                // TPまたはSLが異なる場合のみ更新
                if(MathAbs(current_tp - new_tp) > _Point || MathAbs(current_sl - new_sl) > _Point)
                {
                    trade.PositionModify(ticket, new_sl, new_tp);
                }
            }
        }
    }

    // 平均建値ラインを描画
    DrawAverageLine();
}

//+------------------------------------------------------------------+
//| 平均建値ライン描画                                                |
//+------------------------------------------------------------------+
void DrawAverageLine()
{
    ObjectDelete(0, "AvgEntry");
    ObjectDelete(0, "TP_Line");
    ObjectDelete(0, "SL_Line");

    if(average_entry_price == 0)
        return;

    // 平均建値
    ObjectCreate(0, "AvgEntry", OBJ_HLINE, 0, 0, average_entry_price);
    ObjectSetInteger(0, "AvgEntry", OBJPROP_COLOR, clrYellow);
    ObjectSetInteger(0, "AvgEntry", OBJPROP_STYLE, STYLE_SOLID);
    ObjectSetInteger(0, "AvgEntry", OBJPROP_WIDTH, 2);
    ObjectSetString(0, "AvgEntry", OBJPROP_TEXT, "Avg: " + DoubleToString(average_entry_price, _Digits));

    // TP/SLライン
    double tp_price, sl_price;
    if(current_direction == 1)
    {
        tp_price = average_entry_price + TakeProfit_Pips * pip_value;
        sl_price = average_entry_price - StopLoss_Pips * pip_value;
    }
    else
    {
        tp_price = average_entry_price - TakeProfit_Pips * pip_value;
        sl_price = average_entry_price + StopLoss_Pips * pip_value;
    }

    ObjectCreate(0, "TP_Line", OBJ_HLINE, 0, 0, tp_price);
    ObjectSetInteger(0, "TP_Line", OBJPROP_COLOR, clrLime);
    ObjectSetInteger(0, "TP_Line", OBJPROP_STYLE, STYLE_DASH);

    ObjectCreate(0, "SL_Line", OBJ_HLINE, 0, 0, sl_price);
    ObjectSetInteger(0, "SL_Line", OBJPROP_COLOR, clrRed);
    ObjectSetInteger(0, "SL_Line", OBJPROP_STYLE, STYLE_DASH);
}

//+------------------------------------------------------------------+
//| 新規エントリーチェック                                            |
//+------------------------------------------------------------------+
void CheckNewEntry()
{
    //--- フィボナッチレベル計算
    double fibo_382, fibo_618;
    int trend_direction;

    if(!CalculateFibonacci(fibo_382, fibo_618, trend_direction))
        return;

    //--- MA値取得
    double ma1[], ma2[], ma3[];
    ArraySetAsSeries(ma1, true);
    ArraySetAsSeries(ma2, true);
    ArraySetAsSeries(ma3, true);

    if(CopyBuffer(ma1_handle, 0, 0, 3, ma1) < 3) return;
    if(CopyBuffer(ma2_handle, 0, 0, 3, ma2) < 3) return;
    if(CopyBuffer(ma3_handle, 0, 0, 3, ma3) < 3) return;

    //--- 現在価格
    double close = iClose(_Symbol, TimeFrame, 1);
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    //--- トレンドフィルター
    if(UseTrendFilter)
    {
        double trend_ma[];
        ArraySetAsSeries(trend_ma, true);
        if(CopyBuffer(trend_ma_handle, 0, 0, 2, trend_ma) < 2) return;

        if(trend_direction == 1 && current_price < trend_ma[0]) return;
        if(trend_direction == -1 && current_price > trend_ma[0]) return;
    }

    //--- 許容誤差（pips）
    double tolerance = OverlapTolerance_Pips * pip_value;

    //--- エントリー条件チェック
    bool fibo_condition = false;
    bool ma_condition = false;
    string entry_reason = "";

    //--- フィボナッチ条件（±10pips許容）
    if(MathAbs(close - fibo_382) < tolerance)
    {
        fibo_condition = true;
        entry_reason += "Fibo38.2% ";
    }
    else if(MathAbs(close - fibo_618) < tolerance)
    {
        fibo_condition = true;
        entry_reason += "Fibo61.8% ";
    }

    if(!fibo_condition) return;

    //--- MA条件（±10pips許容）
    if(MathAbs(close - ma1[1]) < tolerance)
    {
        ma_condition = true;
        entry_reason += "MA" + IntegerToString(MA_Period1) + " ";
    }
    if(MathAbs(close - ma2[1]) < tolerance)
    {
        ma_condition = true;
        entry_reason += "MA" + IntegerToString(MA_Period2) + " ";
    }
    if(MathAbs(close - ma3[1]) < tolerance)
    {
        ma_condition = true;
        entry_reason += "MA" + IntegerToString(MA_Period3) + " ";
    }

    if(!ma_condition) return;

    //--- 初回エントリー実行
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double lot = Lot1;

    if(trend_direction == 1)  // 上昇トレンド → 押し目買い
    {
        if(trade.Buy(lot, _Symbol, ask, 0, 0, "FiboMA Buy: " + entry_reason))
        {
            last_entry_price = ask;
            Print("初回BUY エントリー: ", entry_reason, " ロット: ", lot);
            DrawEntryArrow(true, ask, 1);
        }
    }
    else if(trend_direction == -1)  // 下降トレンド → 戻り売り
    {
        if(trade.Sell(lot, _Symbol, bid, 0, 0, "FiboMA Sell: " + entry_reason))
        {
            last_entry_price = bid;
            Print("初回SELL エントリー: ", entry_reason, " ロット: ", lot);
            DrawEntryArrow(false, bid, 1);
        }
    }
}

//+------------------------------------------------------------------+
//| フィボナッチレベル計算                                            |
//+------------------------------------------------------------------+
bool CalculateFibonacci(double &fibo_382, double &fibo_618, int &trend)
{
    double high_array[], low_array[];
    ArraySetAsSeries(high_array, true);
    ArraySetAsSeries(low_array, true);

    if(CopyHigh(_Symbol, TimeFrame, 1, FiboLookbackBars, high_array) < FiboLookbackBars) return false;
    if(CopyLow(_Symbol, TimeFrame, 1, FiboLookbackBars, low_array) < FiboLookbackBars) return false;

    int highest_idx = ArrayMaximum(high_array);
    int lowest_idx = ArrayMinimum(low_array);

    double highest = high_array[highest_idx];
    double lowest = low_array[lowest_idx];
    double range = highest - lowest;

    if(range < _Point * 100) return false;

    if(highest_idx < lowest_idx)
    {
        trend = 1;
        fibo_382 = highest - range * 0.382;
        fibo_618 = highest - range * 0.618;
    }
    else
    {
        trend = -1;
        fibo_382 = lowest + range * 0.382;
        fibo_618 = lowest + range * 0.618;
    }

    DrawFibonacciLevels(highest, lowest, fibo_382, fibo_618, trend);
    return true;
}

//+------------------------------------------------------------------+
//| フィボナッチレベル描画                                            |
//+------------------------------------------------------------------+
void DrawFibonacciLevels(double high, double low, double fibo_382, double fibo_618, int trend)
{
    ObjectDelete(0, "Fibo_High");
    ObjectDelete(0, "Fibo_Low");
    ObjectDelete(0, "Fibo_382");
    ObjectDelete(0, "Fibo_618");

    ObjectCreate(0, "Fibo_High", OBJ_HLINE, 0, 0, high);
    ObjectSetInteger(0, "Fibo_High", OBJPROP_COLOR, clrGreen);
    ObjectSetInteger(0, "Fibo_High", OBJPROP_STYLE, STYLE_SOLID);

    ObjectCreate(0, "Fibo_Low", OBJ_HLINE, 0, 0, low);
    ObjectSetInteger(0, "Fibo_Low", OBJPROP_COLOR, clrRed);
    ObjectSetInteger(0, "Fibo_Low", OBJPROP_STYLE, STYLE_SOLID);

    ObjectCreate(0, "Fibo_382", OBJ_HLINE, 0, 0, fibo_382);
    ObjectSetInteger(0, "Fibo_382", OBJPROP_COLOR, clrGold);
    ObjectSetInteger(0, "Fibo_382", OBJPROP_STYLE, STYLE_DASH);

    ObjectCreate(0, "Fibo_618", OBJ_HLINE, 0, 0, fibo_618);
    ObjectSetInteger(0, "Fibo_618", OBJPROP_COLOR, clrOrange);
    ObjectSetInteger(0, "Fibo_618", OBJPROP_STYLE, STYLE_DASH);
}

//+------------------------------------------------------------------+
//| エントリー矢印描画                                                |
//+------------------------------------------------------------------+
void DrawEntryArrow(bool is_buy, double price, int entry_num)
{
    string name = "Entry_" + IntegerToString(entry_num) + "_" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);

    ObjectCreate(0, name, OBJ_ARROW, 0, TimeCurrent(), price);
    ObjectSetInteger(0, name, OBJPROP_ARROWCODE, is_buy ? 233 : 234);
    ObjectSetInteger(0, name, OBJPROP_COLOR, is_buy ? clrDodgerBlue : clrOrangeRed);
    ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
}
//+------------------------------------------------------------------+
