import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tradingagents.live_portfolio import PortfolioSnapshot, allocation_context, load_live_portfolio
from tradingagents.agents.schemas import LiveDecisionRow, LivePortfolioDecision, render_live_portfolio_decision
from tradingagents.agents.managers.live_allocation import enforce_live_allocation
from tradingagents.agents.utils.agent_utils import build_capital_context
from tradingagents.agents.utils.decision_context import risk_handoff


def snapshot(**kwargs):
    return {"total_portfolio": 10000, "holdings": [
        {"ticker": " aapl ", "quantity": 10, "avg_buy_price": 50},
        {"ticker": "CNC.TO", "quantity": 20, "avg_buy_price": 200},
    ], **kwargs}


def test_minimal_snapshot_is_not_cash_or_market_value(tmp_path):
    path = tmp_path / 'account.json'
    path.write_text(json.dumps(snapshot()))
    h = load_live_portfolio('AAPL', tmp_path)
    assert h['quantity'] == 10 and h['avg_buy_price'] == 50
    assert h['nav'] == h['equity'] == 10000
    assert h['cash'] is None
    a = allocation_context(h, 100)
    assert a['target_weight_pct'] == 10
    assert a['other_unpriced_cost_basis'] == 4000
    assert a['other_known_market_value'] == 0
    assert a['buy_budget'] == 0
    context = build_capital_context(h)
    assert 'all NAV is allocatable' not in context
    assert 'NOT current market value' in context
    # File is re-read on each call, no stale account cache.
    path.write_text(json.dumps(snapshot(total_portfolio=15000)))
    assert load_live_portfolio('AAPL', tmp_path)['nav'] == 15000
    assert load_live_portfolio('MSFT', tmp_path)['quantity'] == 0
    assert load_live_portfolio('cnc.to', tmp_path)['quantity'] == 20


def test_cash_inferred_only_from_complete_market_prices():
    data = snapshot()
    for p in data['holdings']:
        p['market_price'] = 100
    h = PortfolioSnapshot.model_validate(data).for_ticker('AAPL')
    assert h['cash'] == 7000
    assert allocation_context(h)['other_known_weight_pct'] == 20
    assert PortfolioSnapshot(total_portfolio=100, holdings=[]).for_ticker('MSFT')['cash'] == 100


@pytest.mark.parametrize('patch', [
    {'total_portfolio': 0}, {'total_portfolio': -1}, {'total_portfolio': True},
    {'total_portfolio': float('nan')}, {'total_portfolio': float('inf')},
    {'cash': -1}, {'cash': 10001}, {'max_position_pct': 101}, {'currency': 'bad'},
    {'unexpected_field': 1},
    {'holdings': [{'ticker': '', 'quantity': 1, 'avg_buy_price': 1}]},
    {'holdings': [{'ticker': 'AAPL', 'quantity': -1, 'avg_buy_price': 1}]},
    {'holdings': [{'ticker': 'AAPL', 'quantity': 1, 'avg_buy_price': 0}]},
    {'holdings': [{'ticker': 'AAPL', 'quantity': True, 'avg_buy_price': 1}]},
    {'holdings': [{'ticker': 'AAPL', 'quantity': 1, 'avg_buy_price': 1, 'market_price': float('inf')}]},
    {'holdings': [{'ticker': 'AAPL', 'quantity': 1, 'avg_buy_price': 1}, {'ticker': 'aapl', 'quantity': 1, 'avg_buy_price': 1}]},
    {'cash': 100, 'holdings': []},
])
def test_reject_bad_snapshots(patch):
    with pytest.raises(ValidationError):
        PortfolioSnapshot.model_validate(snapshot(**patch))


def test_discovery_and_errors(tmp_path, monkeypatch):
    monkeypatch.setenv('TRADINGAGENTS_PORTFOLIO_DIR', str(tmp_path))
    assert load_live_portfolio('AAPL') is None
    (tmp_path / 'one.json').write_text('{')
    with pytest.raises(ValueError, match='Invalid portfolio snapshot'):
        load_live_portfolio('AAPL')
    (tmp_path / 'two.JSON').write_text('{}')
    with pytest.raises(ValueError, match='exactly one'):
        load_live_portfolio('AAPL')


def test_cli_skips_questions_and_manual_fallback(tmp_path, monkeypatch):
    import cli.main as cli
    monkeypatch.setenv('TRADINGAGENTS_PORTFOLIO_DIR', str(tmp_path))
    nav = MagicMock(return_value=10000)
    manual = MagicMock(return_value={'quantity': 2})
    monkeypatch.setattr(cli, 'get_portfolio_nav', nav)
    monkeypatch.setattr(cli, 'get_holdings_info', manual)
    cash = MagicMock(return_value=5000)
    monkeypatch.setattr(cli, 'get_available_cash', cash)
    path = tmp_path / 'account.json'
    path.write_text(json.dumps(snapshot()))
    assert cli.get_live_holdings_info('AAPL')['quantity'] == 10
    nav.assert_not_called()
    manual.assert_not_called()
    cash.assert_not_called()
    path.write_text('{')
    with pytest.raises(cli.typer.Exit):
        cli.get_live_holdings_info('AAPL')
    nav.assert_not_called()
    path.unlink()
    assert cli.get_live_holdings_info('AAPL') == {'quantity': 2, 'nav': 10000, 'equity': 10000, 'cash': 5000}


def constraints(**kwargs):
    return {'allowed_actions': ['BUY', 'HOLD', 'SELL'], 'entry_mode': 'normal',
            'max_entry_size_pct': 30, 'max_add_position_size_pct': 20, **kwargs}


def decision():
    return LivePortfolioDecision(ticker='AAPL', as_of_date='2026-09-04', decision='BUY', current_price=100,
        holding_period='days', rationale='test', plan_rows=[
            LiveDecisionRow(item='buy1', operation='BUY', shares=1000, reference_price=100, trigger_description='buy'),
            LiveDecisionRow(item='buy2', operation='BUY', shares=1000, reference_price=100, trigger_description='add'),
            LiveDecisionRow(item='stop', operation='CLEAR', scenario='defensive', price_low=90, trigger_description='stop'),
            LiveDecisionRow(item='profit', operation='SELL', scenario='profit', shares=1000, reference_price=110, trigger_description='profit'),
        ])


def test_cumulative_cash_and_allocation_caps():
    h = PortfolioSnapshot.model_validate(snapshot(cash=5000, max_position_pct=25)).for_ticker('AAPL')
    d = enforce_live_allocation(decision(), h, constraints())
    assert [r.shares for r in d.plan_rows[:2]] == [15, 0]
    assert d.plan_rows[0].target_shares_after == 25
    # A defensive stop does not consume the alternative profit branch.
    assert [r.shares for r in d.plan_rows[2:]] == [10, 10]
    assert d.plan_rows[2].target_shares_after == d.plan_rows[3].target_shares_after == 0
    assert decision().plan_rows[0].shares == 1000  # copy, not mutation
    rendered = render_live_portfolio_decision(d, h)
    assert '15.0%' in rendered and '10.0%' in rendered
    assert '其他持仓保持不变' in rendered


def test_add_cap_is_shared_across_all_add_rows():
    d = enforce_live_allocation(decision(), {'nav': 10000, 'quantity': 0, 'cash': 10000}, constraints())
    assert [r.shares for r in d.plan_rows[:2]] == [30, 20]
    h = {'nav': 10000, 'quantity': 10, 'cash': 10000}
    d = enforce_live_allocation(decision(), h, constraints())
    assert [r.shares for r in d.plan_rows[:2]] == [20, 0]


@pytest.mark.parametrize('cash,entry_mode,stop', [(None,'normal',True),(0,'normal',True),(5000,'no_new_or_add',True),(5000,'normal',False)])
def test_no_cash_or_blocked_rule_or_missing_stop_prevents_buys(cash, entry_mode, stop):
    d = decision()
    if not stop:
        d.plan_rows = [r for r in d.plan_rows if r.operation != 'CLEAR']
    result = enforce_live_allocation(d, {'nav':10000, 'cash':cash, 'quantity':10}, constraints(entry_mode=entry_mode))
    assert not any(r.shares for r in result.plan_rows if r.operation == 'BUY')
    assert result.decision == 'HOLD'


def test_price_range_reserves_upper_bound():
    d = decision()
    d.plan_rows[0].price_high = 200
    result = enforce_live_allocation(d, {'nav':10000, 'cash':500, 'quantity':10}, constraints())
    assert result.plan_rows[0].shares == 2
    assert result.plan_rows[1].shares == 1


def test_risk_context_is_bounded_and_preserves_each_role():
    h = risk_handoff({'history': 'old transcript' * 10000, **{
        f'current_{role}_response': role + 'x' * 20000 + 'invalidation'
        for role in ('aggressive','conservative','neutral')}})
    assert len(h) < 4300
    assert h.count('invalidation') == 3
    assert 'old transcript' not in h


def test_pm_reuses_anchors_and_never_fabricates_missing_price(monkeypatch):
    import tradingagents.agents.managers.portfolio_manager as pm
    from tradingagents.graph.propagation import Propagator
    monkeypatch.setattr(pm, 'bind_structured', lambda *a: None)
    state = Propagator().create_initial_state('AAPL', '2026-09-04', {'nav': 10000})
    state['decision_anchors'] = None
    llm = MagicMock()
    memory = MagicMock()
    result = pm.create_portfolio_manager(llm, memory)(state)
    assert result['live_portfolio_decision']['current_price'] is None
    assert result['live_portfolio_decision']['decision'] == 'HOLD'
    llm.invoke.assert_not_called()
    memory.get_memories.assert_not_called()


def test_risk_agents_get_evidence_without_repeating_transcripts():
    from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
    from tradingagents.graph.propagation import Propagator
    state = Propagator().create_initial_state('AAPL', '2026-09-04')
    state['risk_debate_state']['history'] = 'old transcript' * 1000
    state['conflict_report'] = {'signals':[{'factor':'trend', 'direction':'bullish', 'source_report':'market', 'excerpt':'2026-09-04 close=100'}]}
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content='risk assessment')
    result = create_aggressive_debator(llm)(state)
    prompt = llm.invoke.call_args.args[0]
    assert '2026-09-04 close=100' in prompt
    assert 'old transcript' not in prompt
    assert result['risk_debate_state']['history'].startswith(state['risk_debate_state']['history'])


def test_pm_full_handoff_is_bounded_and_cash_is_enforced(monkeypatch):
    import tradingagents.agents.managers.portfolio_manager as pm
    import tradingagents.agents.managers.portfolio_state_manager as state_pm
    from tradingagents.graph.propagation import Propagator
    structured = MagicMock()
    structured.invoke.return_value = decision()
    monkeypatch.setattr(pm, 'bind_structured', lambda *a: structured)
    anchors = {'current_price':100, 'atr14':2, 'ema10':99, 'ema20':98}
    monkeypatch.setattr(state_pm, '_format_short_term_market_anchors', lambda a: json.dumps(a))
    monkeypatch.setattr(state_pm, '_derive_short_term_rule_constraints', lambda *a: constraints())
    monkeypatch.setattr(state_pm, '_format_short_term_rule_constraints', lambda c: json.dumps(c))
    compute = MagicMock(side_effect=AssertionError('should reuse anchors'))
    monkeypatch.setattr(state_pm, '_compute_short_term_market_anchors', compute)
    state = Propagator().create_initial_state('AAPL', '2026-09-04', {'nav':10000, 'cash':1000, 'quantity':10})
    state['decision_anchors'] = anchors
    state['investment_plan'] = 'Research thesis ' * 10000
    state['trader_investment_plan'] = 'Trader proposal ' * 10000
    state['risk_debate_state']['history'] = 'old transcript ' * 10000
    for role in ('aggressive','conservative','neutral'):
        state['risk_debate_state'][f'current_{role}_response'] = role + 'x' * 10000
    memory = MagicMock()
    memory.get_memories.return_value = [{'recommendation':'lesson ' * 10000}] * 2
    result = pm.create_portfolio_manager(MagicMock(), memory)(state)
    assert len(structured.invoke.call_args.args[0]) < 15000
    assert 'old transcript' not in structured.invoke.call_args.args[0]
    assert result['live_portfolio_decision']['plan_rows'][0]['shares'] == 10
    assert result['risk_debate_state']['history'] == state['risk_debate_state']['history']
    compute.assert_not_called()


def test_known_other_exposure_reserves_nav():
    h = PortfolioSnapshot.model_validate({
        'total_portfolio':10000, 'cash':1000,
        'holdings':[
            {'ticker':'AAPL','quantity':10,'avg_buy_price':50},
            {'ticker':'MSFT','quantity':85,'avg_buy_price':80,'market_price':100},
        ],
    }).for_ticker('AAPL')
    result = enforce_live_allocation(decision(), h, constraints())
    assert sum(r.shares for r in result.plan_rows if r.operation == 'BUY') == 5


def test_multiple_profit_rows_cannot_oversell():
    d = decision()
    d.plan_rows[-1].shares = 7
    d.plan_rows.append(LiveDecisionRow(item='profit2', operation='SELL', scenario='profit', shares=7, trigger_description='profit'))
    result = enforce_live_allocation(d, {'nav':10000, 'cash':0, 'quantity':10}, constraints())
    assert [r.shares for r in result.plan_rows[-2:]] == [7, 3]


def test_invalid_numeric_live_levels_rejected():
    with pytest.raises(ValidationError):
        LiveDecisionRow(item='bad', operation='BUY', price_high=float('nan'), trigger_description='bad')


@pytest.mark.parametrize('analyst', ['market','social','news','fundamentals'])
def test_every_bound_analyst_tool_has_an_execution_handler(analyst):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.graph.propagation import Propagator
    from tradingagents.agents.analysts.market_analyst import create_market_analyst
    from tradingagents.agents.analysts.social_media_analyst import create_social_media_analyst
    from tradingagents.agents.analysts.news_analyst import create_news_analyst
    from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst

    class CapturingLLM:
        def bind_tools(self, tools):
            self.names = {tool.name for tool in tools}
            return RunnableLambda(lambda prompt: AIMessage(content='test report'))

    llm = CapturingLLM()
    factories = {'market':create_market_analyst, 'social':create_social_media_analyst,
                 'news':create_news_analyst, 'fundamentals':create_fundamentals_analyst}
    factories[analyst](llm)(Propagator().create_initial_state('AAPL','2026-09-04'))
    node = TradingAgentsGraph._create_tool_nodes(None)[analyst]
    assert llm.names <= set(node.tools_by_name)
