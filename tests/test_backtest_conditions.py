import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from back_test.technical_conditions import BuyConditions, conditions_pass, technical_features
from test_backtest_engine import StaticPriceBacktestEngine, write_strategy


def history(n=100):
    close = pd.Series([100 + i * .15 for i in range(n)])
    return pd.DataFrame({'Date':pd.bdate_range('2025-01-01',periods=n),'Open':close,
                         'High':close+1,'Low':close-1,'Close':close,'Volume':100.0})


def run(tmp_path, data, history_data=None, **overrides):
    return StaticPriceBacktestEngine('TEST', str(data.Date.iloc[0].date()), str(data.Date.iloc[-1].date()),
        strategies_dir=tmp_path, initial_capital=1000, prices=data,
        volume_history=history_data, **overrides).run()


def test_plan_waits_until_later_day_and_fills_without_new_llm(tmp_path):
    data = history()
    data.loc[97,'Volume'] = 300  # confirmed at this close; fill on session 98
    first = str(data.Date.iloc[94].date())
    write_strategy(tmp_path,'TEST',first,valid_until=str(data.Date.iloc[-1].date()),
        entry={'price':200,'size_pct':50}, take_profit={'price':None,'size_pct':0}, stop_loss={'price':80},
        execution_conditions={'entry':{'min_volume_ratio':1.5,'macd':'bullish','kdj':'bullish','boll':'above_mid'}},
        signal_ttl_trading_days={'entry':5})
    result = run(tmp_path,data.iloc[94:],data)
    buys = [e for e in result.executions if e['side']=='BUY']
    assert len(buys) == 1
    assert buys[0]['fill_date'] == str(data.Date.iloc[98].date())
    assert result.report['buy_deferred_conditions'] == 3
    assert result.report['indicator_timing'] == 'prior_completed_bar'
    # Fill day's closing information must not change its entry eligibility.
    changed = data.copy()
    changed.loc[98:,'Volume'] = 0
    same = run(tmp_path,changed.iloc[94:],changed)
    assert [e['fill_date'] for e in same.executions if e['side']=='BUY'] == [buys[0]['fill_date']]


def test_indicator_features_are_prefix_invariant_and_warmed():
    data = history()
    all_features = technical_features(data)
    prefix = technical_features(data.iloc[:80])
    pd.testing.assert_frame_equal(all_features.iloc[:80],prefix)
    assert all_features.macd_hist.iloc[:59].isna().all()
    assert all_features.iloc[-1].macd_hist > 0
    assert all_features.iloc[-1].kdj_k > all_features.iloc[-1].kdj_d
    assert all_features.iloc[-1].kdj_j == pytest.approx(3*all_features.iloc[-1].kdj_k-2*all_features.iloc[-1].kdj_d)
    assert all_features.iloc[-1].boll_mid == pytest.approx(data.Close.iloc[-20:].mean())
    assert all_features.iloc[-1].boll_upper == pytest.approx(data.Close.iloc[-20:].mean()+2*data.Close.iloc[-20:].std(ddof=0))
    assert all_features.iloc[-1].volume_ratio == 1


def test_enabled_conditions_fail_closed_on_short_history(tmp_path):
    data = history(6)
    write_strategy(tmp_path,'TEST',str(data.Date.iloc[0].date()),valid_until=str(data.Date.iloc[-1].date()),
        entry={'price':200,'size_pct':100},stop_loss={'price':80},
        execution_conditions={'entry':{'macd':'bullish'}})
    result = run(tmp_path,data,data)
    assert not any(e['side']=='BUY' for e in result.executions)
    assert result.report['buy_condition_data_missing'] == 5


def test_enabled_volume_gate_fails_closed_when_data_missing(tmp_path):
    data = history(6).drop(columns='Volume')
    write_strategy(tmp_path,'TEST',str(data.Date.iloc[0].date()),valid_until=str(data.Date.iloc[-1].date()),
        entry={'price':200,'size_pct':100},stop_loss={'price':80})
    result = run(tmp_path,data,data,entry_min_volume_ratio=1)
    assert result.report['volume_gate_data_missing'] == 5
    assert not result.executions


def test_each_configured_condition_is_required():
    values = technical_features(history()).iloc[-1].to_dict()
    rules={'min_volume_ratio':1,'macd':'bullish','kdj':'bullish','boll':'above_mid'}
    assert conditions_pass(rules,values)[0]
    for name,value in [('volume_ratio',.5),('macd_hist',-1),('kdj_k',0),('close',0)]:
        assert not conditions_pass(rules,{**values,name:value})[0]
    assert not conditions_pass({'kdj':'cross_up'},values)[0]
    assert conditions_pass({},None)[0]


def test_bad_conditions_do_not_silently_turn_into_price_only(tmp_path):
    data=history(3)
    write_strategy(tmp_path,'TEST',str(data.Date.iloc[0].date()),valid_until=str(data.Date.iloc[-1].date()),
                   execution_conditions={'entry':{'macdd':'bullish'}})
    with pytest.raises(ValueError):
        run(tmp_path,data,data)


def test_explicit_price_only_override_is_available_for_ablation(tmp_path):
    data=history(6)
    write_strategy(tmp_path,'TEST',str(data.Date.iloc[0].date()),valid_until=str(data.Date.iloc[-1].date()),
        entry={'price':200,'size_pct':100},stop_loss={'price':80},
        execution_conditions={'entry':{'macd':'bullish'}})
    result=run(tmp_path,data,data,confirmation_profile='price_only')
    assert any(e['side']=='BUY' for e in result.executions)


def test_first_entry_arms_profit_order_without_next_review(tmp_path):
    data=history(4)
    # Parent fills session 1. Profit is first touched on session 3.
    data.loc[3,['Open','High','Low','Close']] = [110,112,109,111]
    write_strategy(tmp_path,'TEST',str(data.Date.iloc[0].date()),valid_until=str(data.Date.iloc[-1].date()),
        entry={'price':102,'size_pct':100},take_profit={'price':110,'size_pct':100},stop_loss={'price':80})
    result=run(tmp_path,data,data)
    assert result.trades[0]['reason']=='take_profit'
    assert result.trades[0]['exit_date']==str(data.Date.iloc[3].date())


def test_review_can_raise_stop_without_an_add(tmp_path):
    data=history(4)
    data.loc[3,['Open','High','Low','Close']] = [100,101,95,99]
    dates=[str(d.date()) for d in data.Date]
    write_strategy(tmp_path,'TEST',dates[0],valid_until=dates[-1],entry={'price':102,'size_pct':100},
                   take_profit={'price':None,'size_pct':0},stop_loss={'price':80})
    write_strategy(tmp_path,'TEST',dates[2],valid_until=dates[-1],action='HOLD',
                   entry={'price':None,'size_pct':0},take_profit={'price':None,'size_pct':0},stop_loss={'price':98})
    result=run(tmp_path,data,data)
    assert result.trades[0]['reason']=='stop_loss'
    assert result.trades[0]['raw_exit_price']==98


def test_market_anchors_enforce_cutoff_even_if_loader_returns_future(monkeypatch):
    from tradingagents.agents.managers import portfolio_state_manager as pm
    data=history()
    monkeypatch.setattr(pm,'load_ohlcv',lambda *a: data)
    cutoff=str(data.Date.iloc[79].date())
    a=pm._compute_short_term_market_anchors('TEST',cutoff)
    assert a['as_of_close_date']==cutoff
    assert a['history_bars']==80
    assert a['current_price']==data.Close.iloc[79]
    assert a['technical_confirmation']['macd_hist']==pytest.approx(technical_features(data.iloc[:80]).iloc[-1].macd_hist)


def test_open_position_valuation_is_not_realized_feedback():
    from cli.main import _summarize_trading_history
    from back_test.metrics import summarize
    trades=[{'exit_date':'2025-01-02','pnl':100,'reason':'end_of_backtest'},
            {'exit_date':'2025-01-02','pnl':-10,'reason':'stop_loss'}]
    feedback=_summarize_trading_history(trades,'2025-01-02',7)
    assert feedback['n_trades']==1 and feedback['total_pnl']==-10
    metrics=summarize(pd.Series([99,100]),trades,initial_capital=100)
    assert metrics['total_return']==0
    assert metrics['max_drawdown']==pytest.approx(-.01)
    assert metrics['n_trades']==1 and metrics['win_rate']==0


def test_daily_add_confirmation_can_pass_after_plan_day():
    from tradingagents.agents.managers.portfolio_state_manager import _attach_add_execution_condition
    from back_test.policy_config import PortfolioStatePolicyConfig
    plan={'add_position':{'price':100,'size_pct':10}}
    plan=_attach_add_execution_condition(plan,{'nearest_support':110},PortfolioStatePolicyConfig())
    rule=plan['execution_conditions']['add']
    assert not conditions_pass(rule,{'close_min_2':100})[0]
    assert conditions_pass(rule,{'close_min_2':111})[0]


def test_saved_strategy_records_daily_conditions_and_review_ttl(tmp_path,monkeypatch):
    # Exercise the real serializer without any model/provider calls.
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    import tradingagents.graph.trading_graph as graph_module
    graph=object.__new__(TradingAgentsGraph)
    graph.config={'review_cadence_trading_days':7,'backtest_confirmation_profile':'volume_macd_kdj_boll'}
    monkeypatch.setattr(graph_module,'__file__',str(tmp_path/'tradingagents'/'graph'/'trading_graph.py'))
    strategy=json.loads(write_strategy(tmp_path,'TEST','2025-01-01').read_text())
    strategy['execution_conditions']={'add':{'hold_above_price':9.5,'close_hold_days':2}}
    path=graph._save_backtest_strategy('TEST','2025-01-01',{'structured_strategy':strategy})
    saved=json.loads(path.read_text())
    assert saved['execution_conditions']['entry']['kdj']=='bullish'
    assert saved['execution_conditions']['add']['hold_above_price']==9.5
    assert saved['signal_ttl_trading_days']=={'entry':7,'add':7}
