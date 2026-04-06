from __future__ import annotations
import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Optional, Sequence, Dict, Tuple

import polars as pl
from app.helpers.polars_hyper_plugin import *

from app.logs.logging import log
from app.helpers.common import PACT_USERNAMES
from app.services.payload.payloadV4 import Delta, Publish, RoomContext, INDEX_COL_NAME, User
from app.helpers.type_helpers import ensure_list
from app.services.redux.grid_system_v4 import rule, RuleDef, RuleDependency, Priority, EmitMode, DepMode, RulesEngine, RuleContext
from app.helpers.date_helpers import isonow
from app.services.rules.desigMatchV2 import desigNameFuzzyMatchRule
from app.helpers.regex_helpers import hyper_match
from app.services.loaders.kdb_queries_dev_v2 import *

# --------------------------------- Helper Functions --------------------------------

# --------------------------------- Constants ------------------------------------------

NEW_LEVEL_COLS = ("newLevelPx", "newLevelSpd", "newLevelYld", "newLevelMmy", "newLevelDm")
SKEW_COLS = ("relativeSkewValue", "skewType")

STATE_TO_ISREAL = {
    "TEST": 0, "ERROR": 0, "CANCELLED": 0, "DNT": 0, "INDICATIVE": 0, "TRANSFER": 0
}
DEFAULT_ISREAL = 1

AUDIT_USER_KEYS = ("user", "userName", "username", "displayName")
AUDIT_FP_KEYS = ("fingerprint", "fp", "client_fp")
AUDIT_TS_KEYS = ("client_ts", "ts", "timestamp_ms")

MARKET_COL_REGEX = re.compile(r"^([a-z]+)(Bid|Mid|Ask)(Px|Spd|Yld|Mmy|Dm|Ytm|Ytw|Ytc)$")

QT_TO_NEWLEVEL: Dict[str, str] = {
    "PX": "newLevelPx",
    "SPD": "newLevelSpd",
    "YLD": "newLevelYld",
    "MMY": "newLevelMmy",
    "DM": "newLevelDm",
    "YTM": "newLevelYtm",
    "YTW": "newLevelYtw",
    "YTC": "newLevelYtc",
}

CLEAR_RESETS_SKEW = True

# ---- ref_level_expand constants ------------------------------------------------

ACTIVE_QT_TO_SUFFIX: Dict[str, str] = {
    "price": "Px",
    "spread": "Spd",
    "mmy": "Mmy",
    "dm": "Dm",
}

CLIENT_QT_TO_SUFFIX: Dict[str, str] = {
    "PX": "Px",
    "SPD": "Spd",
    "MMY": "Mmy",
    "DM": "Dm",
}

REF_MARKET_WATERFALL = ("macp", "cbbt", "bval", "house")
MANUAL_QT_SUFFIXES = ("Px", "Spd", "Mmy", "Dm")

ALL_MANUAL_COLS = tuple(
    f"manual{side}{qt}"
    for qt in MANUAL_QT_SUFFIXES
    for side in ("Bid", "Mid", "Ask")
)

# ---- wavg_levels constants ------------------------------------------------

WAVG_SKEW_MARKETS = (
    ("Bval", "bval"),
    ("Macp", "macp"),
    ("Markit", "markit"),
    ("Idc", "idc"),
)

_skew_ref_cols: list[str] = []
for _mkt_cap, _mkt_lower in WAVG_SKEW_MARKETS:
    for _side in ("Bid", "Mid", "Ask"):
        for _qt in ("Spd", "Px"):
            _skew_ref_cols.append(f"{_mkt_lower}{_side}{_qt}")
for _side in ("Bid", "Mid", "Ask"):
    _skew_ref_cols.append(f"traceAdj{_side}Px")
WAVG_SKEW_REF_COLS: Tuple[str, ...] = tuple(_skew_ref_cols)

_wavg_base = ("newLevelSpd", "newLevelPx", "newLevelYld", "newLevelMmy",
              "newLevelPxWidth", "newLevelSpdWidth")
_wavg_bwic = tuple(f"bwic{c[0].upper()}{c[1:]}" for c in _wavg_base)
_wavg_owic = tuple(f"owic{c[0].upper()}{c[1:]}" for c in _wavg_base)

_wavg_skew_cols: list[str] = []
for _mkt_cap, _ in WAVG_SKEW_MARKETS:
    for _side in ("Bid", "Mid", "Ask"):
        _wavg_skew_cols.append(f"skew{_mkt_cap}{_side}Spd")
for _mkt_cap, _ in WAVG_SKEW_MARKETS:
    for _side in ("Bid", "Mid", "Ask"):
        _wavg_skew_cols.append(f"skew{_mkt_cap}{_side}Px")
for _side in ("Bid", "Mid", "Ask"):
    _wavg_skew_cols.append(f"skewTraceAdj{_side}Px")

WAVG_ALL_OUTPUT_COLS: Tuple[str, ...] = _wavg_base + _wavg_bwic + _wavg_owic + tuple(_wavg_skew_cols)

WAVG_TRIGGER_COLS: Tuple[str, ...] = (
    "newLevelSpd", "newLevelPx", "newLevelYld", "newLevelMmy",
    "grossDv01", "grossSize", "isReal", "side",
) + WAVG_SKEW_REF_COLS

# --------------------------------- Tiny helpers ---------------------------------------

def build_streaming_s3_rule(
        fields: Optional[Sequence[str]] = None,
        timeout: float = 31,
        max_retries: int = 3,
        retry_delay: float = 0.25,
        retry_jitter: float = 0.05,
        backoff: float = 1.0,
        stream=False,
) -> "RuleDef":
    @rule(
        name="s3_enrichment_stream",
        column_triggers_any=list(NEW_LEVEL_COLS) + [
            'tradeToConvention',
            'benchmarkIsin',
            "benchmarkBidYld",
            "benchmarkBidPx",
        ],
        depends_on_all=(
            RuleDependency("new_level_expand", DepMode.FINISHED),
        ),
        priority=Priority.HIGH,
        emit_mode=EmitMode.IMMEDIATE,
        declared_column_outputs=("newLevel",) + NEW_LEVEL_COLS
    )
    async def _s3_stream(ctx):
        await log.rules("[s3_enrichment] START")
        s3_cols = [
            "isin", "quoteType", "conventionQuoteType",
            "tradeToConvention", "settleDate", "tnum",
            "benchmarkIsin", "benchmarkBidYld", "benchmarkBidPx",
        ]
        df_delta = await ctx.running_delta_slice(columns=s3_cols)
        if df_delta is None or df_delta.hyper.is_empty():
            await log.rules("[s3_enrichment] EXIT: delta is empty")
            return

        portfolio_key = df_delta.hyper.peek("portfolioKey")
        market_cols = df_delta.hyper.cols_like("newLevel(Px|Spd|Yld|Mmy|Dm)$")
        await log.rules(f"[s3_enrichment] market_cols found: {market_cols}")

        if df_delta.filter([pl.col(tm).is_not_null() for tm in market_cols]).hyper.is_empty():
            await log.rules("[s3_enrichment] EXIT: all market cols are null after filter")
            return

        await log.rules(f"[s3_enrichment] calling S3 for portfolio={portfolio_key}, cols={market_cols}")
        from app.server import get_s3
        s3 = get_s3()

        all_payloads = []
        payloads = await s3.pt_to_s3_payloads(df_delta, market_cols=market_cols, contextFields=["tnum"])
        if not payloads:
            await log.rules("[s3_enrichment] EXIT: no payloads")
            return

        s3_chunk = await s3.stream_query_retry_failed(
            payloads,
            timeout=timeout,
            max_retries=max_retries,
            delay=retry_delay,
            jitter=retry_jitter,
            backoff=backoff,
            raw=False,
            stream=False,
        )
        if s3_chunk is None or s3_chunk.hyper.is_empty():
            await log.rules("[s3_enrichment] EXIT: s3_chunk is empty/null")
            return

        out_df = s3_chunk.with_columns([
            pl.col("s3Context").cast(pl.String, strict=False).alias("tnum"),
            pl.lit(portfolio_key, pl.String).alias("portfolioKey"),
        ])
        oc = out_df.hyper.schema()
        of = oc.keys()
        out_df = out_df.rename({x: x.replace("Bid", "").replace("Ask", "") for x in of if "newLevel" in x})

        if all([x in oc for x in ['newLevelBenchYld', 'newLevelYld', 'newLevelSpd']]):
            out_df = out_df.join(df_delta.select(["tnum", "quoteType"]), on="tnum", how="left")
            out_df = out_df.with_columns([
                pl.when(pl.col("newLevelSpd").is_null() & pl.col("newLevelBenchYld").is_not_null() & pl.col("newLevelYld").is_not_null())
                .then(
                    (
                            pl.col("newLevelYld").cast(pl.Float64, strict=False) -
                            pl.col("newLevelBenchYld").cast(pl.Float64, strict=False)
                    ) * 100
                ).otherwise(pl.col("newLevelSpd"))
                .alias("newLevelSpd")
            ])

        # Populate newLevel from the matching quoteType column in S3 output.
        # This handles both paths: matching edits (new_level_expand already set it,
        # S3 overwrites with converted value) and non-matching edits (nobody set
        # newLevel yet, so we derive it from S3's converted matching column).
        if "quoteType" not in out_df.columns:
            out_df = out_df.join(df_delta.select(["tnum", "quoteType"]), on="tnum", how="left")

        out_schema = out_df.hyper.schema()
        nl_cols_present = [c for c in QT_TO_NEWLEVEL.values() if c in out_schema]
        if nl_cols_present:
            out_df = out_df.with_columns(
                pl.coalesce([
                    pl.when(pl.col("quoteType").cast(pl.String).str.to_uppercase() == pl.lit(qt))
                    .then(pl.col(col).cast(pl.Float64, strict=False))
                    .otherwise(pl.lit(None, pl.Float64))
                    for qt, col in QT_TO_NEWLEVEL.items()
                    if col in out_schema
                ]).alias("newLevel")
            )
            await log.rules(f"[s3_enrichment] set newLevel from matching quoteType column")

        await log.rules(f"[s3_enrichment] RETURNING out_df: cols={out_df.columns}, height={out_df.height}")
        return out_df

    return _s3_stream


def _parse_market_col(col: str) -> Optional[Tuple[str, str, str]]:
    m = MARKET_COL_REGEX.match(col or "")
    if not m:
        return None
    market, side, qt = m.group(1), m.group(2), m.group(3)
    return market.upper(), side, qt.upper()


def _build_impacted_updates_for_col(ctx, col: str, j) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    spec = _parse_market_col(col)
    if spec is None:
        return None, None

    market_u, side, qt_u = spec
    target_newlevel = QT_TO_NEWLEVEL.get(qt_u)
    if not target_newlevel:
        return None, None

    pk_cols = list(ctx.target_pks)

    is_target = (
            (pl.col("skewType") == 1) &
            (pl.col("relativeSkewTargetMkt").str.to_uppercase() == pl.lit(market_u)) &
            (pl.col("relativeSkewTargetSide") == pl.lit(side)) &
            (pl.col("relativeSkewTargetQuoteType").str.to_uppercase() == pl.lit(qt_u))
    )

    has_source = is_target & pl.col(col).is_not_null() & pl.col("relativeSkewValue").is_not_null()
    upd = j.filter(has_source)

    updates_df = None
    if not upd.is_empty():
        updates_df = upd.select(
            pk_cols + [
                (pl.col(col) + pl.col("relativeSkewValue")).alias(target_newlevel),
                (pl.col(col) + pl.col("relativeSkewValue")).alias("newLevel")
            ]
        )

    clears_df = None
    if CLEAR_RESETS_SKEW:
        to_clear = j.filter(is_target & pl.col(col).is_null())
        if not to_clear.is_empty():
            clears_df = to_clear.select(pk_cols).with_columns([
                pl.lit(None, dtype=pl.Float64).alias(target_newlevel),
                pl.lit(0, dtype=pl.Int8).alias("skewType"),
                pl.lit(None, dtype=pl.Utf8).alias("relativeSkewTargetMkt"),
                pl.lit(None, dtype=pl.Utf8).alias("relativeSkewTargetSide"),
                pl.lit(None, dtype=pl.Utf8).alias("relativeSkewTargetQuoteType"),
                pl.lit(None, dtype=pl.Float64).alias("relativeSkewValue"),
            ])

    return updates_df, clears_df


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

# FUZZY DESIG/ASSIGNED: Auto swaps an entered name for a cleanly formatted, full name.
# "~" overrides this behavior, entered information becomes the value (minus the tilda)
# "REMOVE", "x", etc. set as REMOVED which triggers 0 size
# Fast path check against common names where phonebook might fail
# Checks against names currently in the basket
# Checks against the phonebook
# Factors in tickers and name occurances into guesses

@rule(
    name="fuzzy_desig",
    room_pattern="*.PORTFOLIO",
    column_triggers_any=("desigName", ),
    priority=Priority.MEDIUM,
    emit_mode=EmitMode.IMMEDIATE,
    suppress_cascade=True,
    declared_column_outputs=("desigName", "assignedTrader"),
)
async def fuzzy_desig(ctx):
    try:
        from app.server import get_pbf
        fuzzy_matcher = await get_pbf()
    except Exception as e:
        await log.error(f"Fuzzy Match Rule failed to initialize with phone book: {e}")
        fuzzy_matcher = desigNameFuzzyMatchRule(None)
    res = await fuzzy_matcher.execute_fuzzy_match(ctx, base_field="desigName")
    return res

@rule(
    name="fuzzy_assigned",
    room_pattern="*.PORTFOLIO",
    column_triggers_any=("assignedTrader", ),
    priority=Priority.HIGH,
    emit_mode=EmitMode.IMMEDIATE,
    suppress_cascade=True,
    declared_column_outputs=("desigName", "assignedTrader"),
)
async def fuzzy_assigned(ctx):
    try:
        from app.server import get_pbf
        fuzzy_matcher = await get_pbf()
    except Exception as e:
        await log.error(f"Fuzzy Match Rule failed to initialize with phone book: {e}")
        fuzzy_matcher = desigNameFuzzyMatchRule(None)
    return await fuzzy_matcher.execute_fuzzy_match(ctx, base_field="assignedTrader")


# REMOVED_BOND: Sets the size to be 0 and the isReal flag to be false.
# Or, alternatively, re-sets the size as the originalSize and sets the isReal flag to true.

@rule(
    name="removed_bond",
    room_pattern="*.PORTFOLIO",
    priority=Priority.HIGH,
    emit_mode=EmitMode.IMMEDIATE,
    depends_on_any=(
            RuleDependency("fuzzy_desig", DepMode.FINISHED),
            RuleDependency("fuzzy_assigned", DepMode.FINISHED),
    ),
    declared_column_outputs=("grossSize", "isReal"),
)
async def removed_bond(ctx: RuleContext):
    slice = await ctx.running_delta_slice(columns=['assignedTrader', 'desigName'])
    prior = await ctx.prior_delta_slice(columns=['assignedTrader', 'desigName', 'originalSize'])
    data = slice.join(prior, on=['tnum', 'portfolioKey'], how='left', suffix='_old')

    removed = data.filter([
        (pl.col('desigName').cast(pl.String, strict=False) == 'REMOVED') |
        (pl.col('assignedTrader').cast(pl.String, strict=False) == 'REMOVED'),
        (pl.col('desigName_old').cast(pl.String, strict=False) != 'REMOVED') |
        (pl.col('assignedTrader_old').cast(pl.String, strict=False) != 'REMOVED')
    ]).with_columns([
        pl.lit(0, pl.Float64).alias('grossSize'),
        pl.lit(0, pl.Int8).alias('isReal')
    ])

    added = data.filter([
        (pl.col('desigName').cast(pl.String, strict=False) != 'REMOVED') |
        (pl.col('assignedTrader').cast(pl.String, strict=False) != 'REMOVED'),
        (pl.col('desigName_old').cast(pl.String, strict=False) == 'REMOVED') |
        (pl.col('assignedTrader_old').cast(pl.String, strict=False) == 'REMOVED')
    ]).with_columns([
        pl.col('originalSize').alias('grossSize'),
        pl.lit(1, pl.Int8).alias('isReal')
    ])

    return pl.concat([added, removed], how='vertical').select(['tnum', 'portfolioKey', 'grossSize', 'isReal'])


# RISK_ADJUST: Keeps risk metrics in line with any changes in size
# Note that notional amendments -> trigger dv01 adjustments but not the other way around
# This is intentional to allow for DV01 corrections whereas the Notional is provided from the client
@rule(
    name="risk_adjust",
    column_triggers_any=("grossSize", "unitAccrued", "unitDv01"),
    room_pattern="*.PORTFOLIO",
    target_grid_id="portfolio",
    priority=Priority.MEDIUM,
    emit_mode=EmitMode.IMMEDIATE,
    declared_column_outputs=("grossSize", "grossDv01", 'accruedInterest'),
)
async def risk_adjust(ctx):
    """When we adjust the size, adjust the dv01 as well."""
    from app.services.loaders.kdb_queries_dev_v2 import risk_transforms
    data = (await ctx.running_delta_slice(['tnum', 'isin', 'portfolioKey'] + ['unitDv01','unitAccrued','unitCs01','unitCs01Pct','grossSize','axeFullBidSize','axeFullAskSize','netFirmPosition','netAlgoPosition','netStrategyPosition','netDeskPosition','signalFlag','side']))
    res = await risk_transforms(data)
    return res.join(data.select('tnum', 'portfolioKey').lazy(), on='tnum', how='inner')






@rule(
    name="edit_audit",
    column_triggers_any=tuple(NEW_LEVEL_COLS) + SKEW_COLS,
    depends_on_all=(
            RuleDependency("new_level_expand", DepMode.FINISHED),
            RuleDependency("clear_levels", DepMode.FINISHED),
    ),
    room_pattern="*.PORTFOLIO",
    priority=Priority.AUDIT,
    emit_mode=EmitMode.END,
    declared_column_outputs=("lastEditUser", "lastEditTime"),
)
async def edit_audit(ctx):
    """Record who made the edit and when."""
    try:
        user_data = ctx.ingress_user
        username = user_data.username
        name = user_data.displayName

        if (name is None) or (str(name).upper()=="SERVER"): return


        delta = await ctx.running_delta_slice(['newLevel', 'lastAdminEditNewLevel', 'lastTraderEditNewLevel', 'desigName'])
        if not isinstance(user_data, User):
            await log.error(f"Wrong user_data type: {type(user_data)}")
            return

        ts = isonow(True)
        exprs = [pl.lit(ts).alias("lastEditTime")]

        if not user_data.impersonateMode:
            my_name = pl.lit(name, pl.String)
            exprs.append(my_name.alias("lastEditUser"))
        else:
            my_name = pl.col('desigName')
            exprs.append(my_name.alias("lastEditUser"))
            exprs.append(pl.lit('IMPERSONATE', pl.String).alias('lastEditSource'))

        pk_cols = ['portfolioKey', 'tnum']
        cols = pk_cols + ['newLevel', 'desigName']

        if (username in PACT_USERNAMES) and (not user_data.impersonateMode):
            exprs.extend([
                my_name.alias("lastAdminEditUser"),
                pl.lit(ts).alias("lastAdminEditTimestamp"),
                pl.when(pl.col('newLevel').is_not_null())
                    .then(pl.col('newLevel'))
                    .otherwise(pl.col('lastAdminEditNewLevel'))
                .alias('lastAdminEditNewLevel')
            ])
            cols.extend(['lastAdminEditNewLevel'])
            if not user_data.impersonateMode:
                exprs.append(pl.lit('ADMIN').alias('lastEditSource'))
        else:
            exprs.extend([
                my_name.alias("lastTraderEditUser"),
                pl.lit(ts).alias("lastTraderEditTimestamp"),
                pl.when(pl.col('newLevel').is_not_null())
                    .then(pl.col('newLevel'))
                    .otherwise(pl.col('lastTraderEditNewLevel'))
                .alias('lastTraderEditNewLevel')
            ])
            cols.extend(['lastTraderEditNewLevel'])
            if not user_data.impersonateMode:
                exprs.append(pl.lit('TRADER').alias('lastEditSource'))

        if exprs:
            res = delta.select(cols).with_columns(exprs).drop(['newLevel', 'desigName'], strict=False)
            return res
    except Exception as e:
        await log.error(f"audit error: {e}")
        return None


@rule(
    name="new_level_expand",
    column_triggers_any=NEW_LEVEL_COLS,
    room_pattern="*.PORTFOLIO",
    priority=Priority.HIGH,
    emit_mode=EmitMode.IMMEDIATE,
    declared_column_outputs=("newLevel",) + NEW_LEVEL_COLS,
)
async def new_level_expand(ctx):
    """When user edits newLevelPx/Spd/Mmy/Dm AND that column matches the row's
    quoteType, set newLevel to that value and clear the other quote-type columns.
    Null edits (deletions) are ignored — clear_levels handles those.
    Non-matching edits (user changed a column that doesn't match quoteType) are
    left alone — we don't touch newLevel."""
    try:
        await log.rules("[new_level_expand] START")

        # Which newLevel columns did the user ACTUALLY change?
        user_changed = set(ctx.triggering_delta.columns) & set(NEW_LEVEL_COLS)
        await log.rules(f"[new_level_expand] user actually changed: {user_changed}")
        if not user_changed:
            await log.rules("[new_level_expand] EXIT: no NEW_LEVEL_COLS in triggering delta")
            return None

        all_mkts = list(NEW_LEVEL_COLS)
        delta = await ctx.running_delta_slice(columns=all_mkts + ["quoteType"])
        await log.rules(f"[new_level_expand] delta cols={delta.columns}, height={delta.height}")
        if delta.is_empty():
            await log.rules("[new_level_expand] EXIT: delta is empty")
            return None

        pk_cols = list(ctx.target_pks)
        rows = []
        for row in delta.iter_rows(named=True):
            qt = str(row.get("quoteType") or "").upper().strip()
            matching_col = QT_TO_NEWLEVEL.get(qt)
            await log.rules(f"[new_level_expand] row quoteType={qt!r} -> matching_col={matching_col!r}")

            if matching_col is None or matching_col not in all_mkts:
                await log.rules(f"[new_level_expand] SKIP: no matching col for quoteType={qt!r}")
                continue

            # Only act if the user actually edited the matching column
            if matching_col not in user_changed:
                await log.rules(f"[new_level_expand] SKIP: user edited {user_changed} but matching col is {matching_col!r} — non-matching edit")
                continue

            val = row.get(matching_col)
            await log.rules(f"[new_level_expand] {matching_col}={val!r}")
            if val is None:
                await log.rules(f"[new_level_expand] SKIP: {matching_col} is null (deletion path)")
                continue  # deletion — handled by clear_levels

            out_row = {pk: row[pk] for pk in pk_cols}
            out_row["newLevel"] = val
            for col in all_mkts:
                out_row[col] = val if col == matching_col else None
            rows.append(out_row)

        await log.rules(f"[new_level_expand] produced {len(rows)} output rows")
        if rows:
            return pl.DataFrame(rows)
        return None
    except Exception as e:
        await log.error("new_level_expand error:", e=str(e))
        return None


@rule(
    name="trader_claim",
    column_triggers_any=("claimed",),
    priority=Priority.MEDIUM,
    emit_mode=EmitMode.END,
    declared_column_outputs=("assignedTrader",),
)
async def trader_claim(ctx):
    """Handle trader claiming a bond."""
    try:
        pks = list(ctx.target_pks)

        prior = await ctx.price_delta_slice(columns=["algoAssigned", "whichAlgo", "desigName", "desigTraderId"])
        delta = await ctx.running_delta_slice(columns=["algoAssigned", "whichAlgo", "desigTraderId"])
        new_delta = delta.join(prior, on=ctx.target_pks, how="left", suffix="_prior")

        user_data = getattr(ctx.state, "user_data", {})
        return new_delta.with_columns([
            pl.col("claimed").cast(pl.Int8, strict=False)
        ]).with_columns([
            pl.when(pl.col("claimed") == 2)
            .then(pl.lit(user_data.get("displayName", "UNKNOWN")))
            .when(pl.col("claimed") == 1).then(pl.col("desigName"))
            .otherwise(
                pl.when(pl.col("algoAssigned").cast(pl.Boolean, strict=False))
                .then(pl.concat_str(pl.col("whichAlgo"), pl.lit("ALGO"), separator=" "))
                .otherwise(pl.col("desigName"))
            ).cast(pl.String, strict=False).alias("assignedTrader"),
        ]).with_columns([
            pl.when(pl.col("desigName").is_null() & pl.col("assignedTrader").is_not_null() & (pl.col("claimed") > 0))
            .then(pl.col("assignedTrader")).otherwise(pl.col("desigName")).alias("desigName")
        ]).select(["assignedTrader", "desigName", "tnum", "portfolioKey"])
    except Exception as e:
        await log.error("trader claim:", e=str(e))
        return None


@rule(
    name="refresh_all_markets",
    column_triggers_any=("refSyncTime",),
    room_pattern="*.PORTFOLIO",
    priority=Priority.MEDIUM,
    emit_mode=EmitMode.END,
)
async def refresh_all_markets(ctx):
    """Trigger market refresh from KDB."""
    from app.services.loaders.kdb_queries_dev import get_all_quotes
    from app.services.loaders.kdb_queries_dev import coalesce_left_join

    pk_cols = set(ctx.target_pks or [])
    if not pk_cols:
        await log.warning("[refresh_markets] grid missing PKs; skipping")
        return None

    try:
        target = ctx.prior_delta_slice(columns=["isin", "cusip", "sym", "portfolioKey", "tnum"])
        try:
            quotes = await get_all_quotes(target.lazy())
            quotes = await coalesce_left_join(target.lazy(), quotes.lazy(), on="isin")
            quotes = quotes.drop(["cusip", "sym"], strict=False)
        except Exception as e:
            await log.error(f"[refresh_markets] quotes fetch failed: {e}")
            return None

        if (quotes is None) or quotes.hyper.is_empty():
            await log.error("[refresh_markets] quotes fetch empty")
            return None

        return quotes
    except Exception as e:
        await log.error(f"[refresh_markets] unexpected error: {e}")
        return None


@rule(
    name="state_check",
    column_triggers_any=("state",),
    room_pattern="*.META",
    priority=Priority.LOW,
    emit_mode=EmitMode.END,
    declared_column_outputs=("isReal",),
)
async def state_check(ctx):
    """If state is in TEST, ERROR, etc. set isReal to False."""
    try:
        FAKE_KEYS = list(STATE_TO_ISREAL.keys())
        return (await ctx.running_delta_slice(columns=["isReal", "bsrPct", "numDealers"])).with_columns([
            pl.col("state").cast(pl.String, strict=False).str.to_uppercase().alias("state"),
        ]).with_columns([
            pl.when(pl.col("state").is_in(FAKE_KEYS))
            .then(pl.lit(0, pl.Int8))
            .otherwise(pl.lit(1, pl.Int8))
            .alias("isReal"),
        ]).select(["date", "portfolioKey", "isReal"])
    except Exception as e:
        await log.error("state check:", e=str(e))
        return None

@rule(
    name="rank_update",
    column_triggers_any=("state",),
    room_pattern="*.META",
    priority=Priority.LOW,
    emit_mode=EmitMode.END,
    declared_column_outputs=("barcRank",),
)
async def rank_update(ctx):
    try:
        return (await ctx.running_delta_slice(columns=["barcRank", "numDealers"])).with_columns([
            pl.col("state").cast(pl.String, strict=False).str.to_uppercase().alias("state"),
        ]).with_columns([
            pl.when(pl.col('barcRank').is_null()).then(
                pl.when(pl.col("state") == 'WON').then(pl.lit(1, pl.Int64))
                .when(pl.col('state') == 'COVERED').then(pl.lit(2, pl.Int64))
                .when(pl.col('numDealers').is_not_null() & (pl.col('numDealers') == 3) & (pl.col('state')=='MISSED')).then(pl.lit(3, pl.Int64))
                .otherwise(pl.lit(None, pl.Int64))
            ).otherwise(pl.col('barcRank')).alias("barcRank"),
        ]).select(["date", "portfolioKey", "barcRank"])
    except Exception as e:
        await log.error("state check:", e=str(e))
        return None


@rule(
    name="test_in_name",
    column_triggers_any=("client",),
    room_pattern="*.META",
    priority=Priority.LOW,
    emit_mode=EmitMode.END,
    declared_column_outputs=("isReal", "state"),
)
async def test_in_name(ctx):
    """If 'test' in client name, set state to TEST."""
    delta = await ctx.running_delta_slice()
    if (delta is None) or (not delta.hyper.is_empty()): return
    d = delta.hyper.peek(["client", 'portfolioKey', 'date'])
    client, portfolioKey, portfolioDate = d.get('client'), d.get('portfolioKey'), d.get('date')
    if client and hyper_match(r"(?i)\btest\b|\btest(?=\w)", client, case_sensitive=False):
        return pl.DataFrame([{"portfolioKey": portfolioKey, "date": portfolioDate, "state": "TEST"}])


@rule(
    name="meta_update",
    depends_on_all=(
            RuleDependency("new_level_expand", DepMode.FINISHED),
            RuleDependency("clear_levels", DepMode.FINISHED),
            RuleDependency("s3_enrichment_stream", DepMode.FINISHED),
    ),
    column_triggers_all=("newLevel",),
    room_pattern="*.PORTFOLIO",
    target_grid_id="meta",
    target_primary_keys=("portfolioKey", "date"),
    priority=Priority.LOW,
    emit_mode=EmitMode.END,
    declared_column_outputs=("pctPriced",),
)
async def meta_update(ctx):
    """Update pct priced on meta grid."""
    try:

        new_basket = await ctx.running_frame_slice(['newLevel', 'rfqCreateDate'])
        delta = new_basket.filter([pl.col("newLevel").is_not_null()])
        delta_height = delta.hyper.height() or 0
        total = new_basket.hyper.height()

        key = new_basket.hyper.peek("portfolioKey")
        date = new_basket.hyper.peek("rfqCreateDate")

        if total > 0:
            priced_pct = round(delta_height / total, 4)
            return Delta(
                frame=pl.DataFrame([{"portfolioKey": key, "date": date, "pctPriced":priced_pct}]),
                pk_columns=["portfolioKey", "date"],
                changed_columns=["pctPriced"],
                mode="update",
            )
        else:
            await log.error("new_basket height is 0:", new_basket)
            return None
    except Exception as e:
        await log.error("meta update", e=str(e))
        return None


@rule(
    name="market_propegate",
    room_pattern="*.PORTFOLIO",
    priority=Priority.HIGH,
    emit_mode=EmitMode.IMMEDIATE,
    declared_column_outputs=tuple(QT_TO_NEWLEVEL.values()) + (
            "skewType", "relativeSkewTargetMkt", "relativeSkewTargetSide",
            "relativeSkewTargetQuoteType", "relativeSkewValue",
    ),
)
async def market_propegate(ctx):
    """When reference markets update, update dependent skews."""
    slice = await ctx.running_delta_slice()
    market_cols = [c for c in slice.columns if MARKET_COL_REGEX.match(str(c))]
    if not market_cols:
        return None

    updates, clears = [], []

    need_cols = [
        "skewType", "relativeSkewTargetMkt", "relativeSkewTargetSide",
        "relativeSkewTargetQuoteType", "relativeSkewValue",
    ]
    j = await ctx.running_delta_slice(columns=need_cols)

    j = j.with_columns([
        pl.col("skewType").cast(pl.Int8, strict=False),
        pl.col("relativeSkewTargetMkt").cast(pl.Utf8, strict=False),
        pl.col("relativeSkewTargetSide").cast(pl.Utf8, strict=False),
        pl.col("relativeSkewTargetQuoteType").cast(pl.Utf8, strict=False),
        pl.col("relativeSkewValue").cast(pl.Float64, strict=False),
    ])

    for col in market_cols:
        upd_df, clr_df = _build_impacted_updates_for_col(ctx, col, j)
        if isinstance(upd_df, pl.DataFrame) and not upd_df.is_empty():
            updates.append(upd_df)
        if isinstance(clr_df, pl.DataFrame) and not clr_df.is_empty():
            clears.append(clr_df)

    pk = list(ctx.target_pks)
    out = []

    if clears:
        clear_concat = pl.concat(clears, how="vertical_relaxed", rechunk=False) if len(clears) > 1 else clears[0]
        clear_concat = clear_concat.unique(subset=pk, keep="last", maintain_order=True)
        clear_concat.shrink_to_fit(in_place=True)
        out.append(clear_concat)

    if updates:
        upd_concat = pl.concat(updates, how="vertical_relaxed", rechunk=False) if len(updates) > 1 else updates[0]
        upd_concat = upd_concat.unique(subset=pk, keep="last", maintain_order=True)
        upd_concat.shrink_to_fit(in_place=True)
        out.append(upd_concat)

    return out if out else None


@rule(
    name="clear_levels",
    column_triggers_any=NEW_LEVEL_COLS,
    room_pattern="*.PORTFOLIO",
    depends_on_all=(RuleDependency("new_level_expand", DepMode.FINISHED),),
    priority=Priority.CRITICAL,
    emit_mode=EmitMode.IMMEDIATE,
    suppress_cascade=True,
    declared_column_outputs=("newLevel",) + NEW_LEVEL_COLS,
)
async def clear_levels(ctx):
    """When user deletes newLevel (was non-null, now null), clear all newLevel columns."""
    await log.rules("[clear_levels] START")

    df_delta = await ctx.ingress_delta_slice()
    await log.rules(f"[clear_levels] ingress delta cols={df_delta.columns}, height={df_delta.height}")

    s = df_delta.hyper.schema()
    pk_cols = list(ctx.target_pks)

    touched_cols = [c for c in NEW_LEVEL_COLS if c in s]
    await log.rules(f"[clear_levels] touched_cols={touched_cols}")
    if not touched_cols:
        await log.rules("[clear_levels] EXIT: no NEW_LEVEL_COLS in delta schema")
        return None

    prior = await ctx.prior_delta_slice(columns=touched_cols)
    await log.rules(f"[clear_levels] prior delta height={prior.height}")
    if prior.is_empty():
        await log.rules("[clear_levels] EXIT: prior delta is empty")
        return None

    joined = df_delta.select(pk_cols + touched_cols).join(
        prior.select(pk_cols + touched_cols),
        on=pk_cols, how="inner", suffix="_prior",
    )
    await log.rules(f"[clear_levels] joined height={joined.height}, cols={joined.columns}")

    # A column was deleted if it is null NOW and was non-null BEFORE
    delete_exprs = [
        (pl.col(c).is_null() & pl.col(f"{c}_prior").is_not_null())
        for c in touched_cols
    ]
    clear_mask = pl.any_horizontal(delete_exprs)
    rows_to_clear = joined.filter(clear_mask).select(pk_cols)
    await log.rules(f"[clear_levels] rows_to_clear={rows_to_clear.height}")
    if rows_to_clear.hyper.is_empty():
        await log.rules("[clear_levels] EXIT: no rows transitioned non-null -> null")
        return None

    await log.rules(f"[clear_levels] CLEARING {rows_to_clear.height} rows")
    clear_df = rows_to_clear.with_columns([
        pl.lit(None, pl.Float64).alias("newLevel"),
        pl.lit(None, pl.Float64).alias("newLevelPx"),
        pl.lit(None, pl.Float64).alias("newLevelSpd"),
        pl.lit(None, pl.Float64).alias("newLevelMmy"),
        pl.lit(None, pl.Float64).alias("newLevelYld"),
        pl.lit(None, pl.Float64).alias("newLevelDm"),
    ])

    return clear_df


@rule(
    name="desig_static_enhance",
    column_triggers_all=("desigName",),
    depends_on_all=(RuleDependency("fuzzy_name", DepMode.FINISHED),),
    priority=Priority.MEDIUM,
    emit_mode=EmitMode.IMMEDIATE,
)
async def desig_static_enhance(ctx):
    """Update trader static information after fuzzy name match."""
    data = await ctx.running_delta_slice(columns=[
        'desigBook',
        'desigRegion',
        'desigRole',
        'desigBusinessArea3',
        'desigBusinessArea4',
        'desigBusinessArea5',
        'desigDesk',
        'desigAsset',
        'desigNickname',
        'desigOrg',
        'desigEmail',
        'desigBrid',
        'desigFirstName',
        'desigLastName',
        'desigTraderId',
        'assignedTrader',
        'algoAssigned',
        'desigPosition',
        'isDesigBsr',
        'desigBsiSize',
        'desigBsrSize',
        'emRegion',
        'regionBarclaysDesk',
        'regionBarclaysRegion'
    ])
    print(data.to_dicts())



@rule(
    name="dv01_adjust_meta",
    depends_on_all=(RuleDependency("dv01_adjust", DepMode.SUCCEEDED),),
    room_pattern="*.PORTFOLIO",
    target_grid_id="meta",
    target_primary_keys=("portfolioKey", "date"),
    priority=Priority.MEDIUM,
    emit_mode=EmitMode.IMMEDIATE,
)
async def dv01_adjust_meta(ctx):
    """Update meta grid with summarized DV01 after adjustment."""
    from app.services.portfolio.meta import summarize_pt_for_meta
    data = ctx.running_frame_slice.with_columns([
        pl.col('grossSize').fill_null(0), pl.col('netSize').fill_null(0), pl.col('unitDv01').fill_null(0),
    ])
    m = await summarize_pt_for_meta(data)
    key, date = ctx.portfolioKey, ctx.portfolioDate
    d = Delta(
        frame=m.with_columns(pl.lit(key, pl.String).alias('portfolioKey'), pl.lit(date,pl.Date).alias('date')),
        pk_columns=["portfolioKey", "date"],
        mode="update"
    )
    return Publish(
        d, "update",
        grid_id="meta",
        room=f"{key.upper()}.META",
        grid_filters={"column": "portfolioKey", "op": "eq", "value": key.lower()},
        options={"persist": True, "broadcast": True, "trigger_rules": False, "relay": True},
    )


# ---------------------------------------------------------------------------
# wavg_levels helpers
# ---------------------------------------------------------------------------

def _wavg_level_exprs(prefix: str = "") -> list:
    """Build wavg select expressions for the 4 level columns.
    prefix="" → base names (newLevelSpd, ...).
    prefix="bwic"/"owic" → bwicNewLevelSpd, ...
    """
    def _name(base: str) -> str:
        return f"{prefix}{base[0].upper()}{base[1:]}" if prefix else base

    return [
        pl.col("newLevelSpd").hyper.wavg(pl.col("grossDv01")).alias(_name("newLevelSpd")),
        pl.col("newLevelPx").hyper.wavg(pl.col("grossSize")).alias(_name("newLevelPx")),
        pl.col("newLevelYld").hyper.wavg(pl.col("grossDv01")).alias(_name("newLevelYld")),
        pl.col("newLevelMmy").hyper.wavg(pl.col("grossDv01")).alias(_name("newLevelMmy")),
    ]


def _skew_wavg_exprs() -> list:
    """Build expressions for all skew wavg calculations.
    skew = newLevel - refMarket, then wavg by appropriate weight.
    Spd → grossDv01, Px → grossSize.
    """
    exprs: list = []
    # Spd skews (weight by grossDv01)
    for mkt_name, mkt_lower in WAVG_SKEW_MARKETS:
        for side in ("Bid", "Mid", "Ask"):
            exprs.append(
                (pl.col("newLevelSpd") - pl.col(f"{mkt_lower}{side}Spd"))
                .hyper.wavg(pl.col("grossDv01"))
                .alias(f"skew{mkt_name}{side}Spd")
            )
    # Px skews (weight by grossSize)
    for mkt_name, mkt_lower in WAVG_SKEW_MARKETS:
        for side in ("Bid", "Mid", "Ask"):
            exprs.append(
                (pl.col("newLevelPx") - pl.col(f"{mkt_lower}{side}Px"))
                .hyper.wavg(pl.col("grossSize"))
                .alias(f"skew{mkt_name}{side}Px")
            )
    # TraceAdj Px skews (weight by grossSize)
    for side in ("Bid", "Mid", "Ask"):
        exprs.append(
            (pl.col("newLevelPx") - pl.col(f"traceAdj{side}Px"))
            .hyper.wavg(pl.col("grossSize"))
            .alias(f"skewTraceAdj{side}Px")
        )
    return exprs


# ---------------------------------------------------------------------------
# Rule: wavg_levels  (cross-grid: PORTFOLIO -> META)
# ---------------------------------------------------------------------------

@rule(
    name="wavg_levels",
    room_pattern="*.PORTFOLIO",
    target_grid_id="meta",
    target_primary_keys=("portfolioKey", "date"),
    column_triggers_any=WAVG_TRIGGER_COLS,
    depends_on_all=(
        RuleDependency("new_level_expand", DepMode.FINISHED),
        RuleDependency("clear_levels", DepMode.FINISHED),
        RuleDependency("s3_enrichment_stream", DepMode.FINISHED),
    ),
    priority=Priority.LOW,
    emit_mode=EmitMode.END,
    declared_column_outputs=WAVG_ALL_OUTPUT_COLS,
)
async def wavg_levels(ctx: RuleContext):
    """Compute weighted-average levels from PORTFOLIO and emit to META grid."""
    try:
        await log.rules("[wavg_levels] START")

        fetch_cols = [
            "newLevelSpd", "newLevelPx", "newLevelYld", "newLevelMmy",
            "grossDv01", "grossSize", "isReal", "side",
            "portfolioKey", "rfqCreateDate",
        ] + list(WAVG_SKEW_REF_COLS)

        basket = await ctx.running_frame_slice(fetch_cols)
        if basket.is_empty():
            await log.rules("[wavg_levels] EXIT: basket empty")
            return None

        real = basket.filter(pl.col("isReal").cast(pl.Int8, strict=False) == 1)
        if real.is_empty():
            await log.rules("[wavg_levels] EXIT: no isReal=1 rows")
            return None

        key = basket.hyper.peek("portfolioKey")
        date = basket.hyper.peek("rfqCreateDate")

        # --- Base wavg levels (all isReal=1) ---
        base_vals = real.select(_wavg_level_exprs()).row(0, named=True)

        # --- BWIC = side "BUY" ---
        bwic_df = real.filter(pl.col("side") == "BUY")
        if not bwic_df.is_empty():
            bwic_vals = bwic_df.select(_wavg_level_exprs("bwic")).row(0, named=True)
        else:
            bwic_vals = {f"bwic{c[0].upper()}{c[1:]}": None
                         for c in ("newLevelSpd", "newLevelPx", "newLevelYld", "newLevelMmy")}

        # --- OWIC = side "SELL" ---
        owic_df = real.filter(pl.col("side") == "SELL")
        if not owic_df.is_empty():
            owic_vals = owic_df.select(_wavg_level_exprs("owic")).row(0, named=True)
        else:
            owic_vals = {f"owic{c[0].upper()}{c[1:]}": None
                         for c in ("newLevelSpd", "newLevelPx", "newLevelYld", "newLevelMmy")}

        # --- Widths = difference between OWIC and BWIC wavgs ---
        bwic_px = bwic_vals.get("bwicNewLevelPx")
        owic_px = owic_vals.get("owicNewLevelPx")
        bwic_spd = bwic_vals.get("bwicNewLevelSpd")
        owic_spd = owic_vals.get("owicNewLevelSpd")

        px_width = (owic_px - bwic_px) if (owic_px is not None and bwic_px is not None) else None
        spd_width = (bwic_spd - owic_spd) if (bwic_spd is not None and owic_spd is not None) else None

        width_vals = {
            "newLevelPxWidth": px_width,
            "newLevelSpdWidth": spd_width,
            "bwicNewLevelPxWidth": px_width,
            "bwicNewLevelSpdWidth": spd_width,
            "owicNewLevelPxWidth": px_width,
            "owicNewLevelSpdWidth": spd_width,
        }

        # --- Skew wavg calculations ---
        skew_vals = real.select(_skew_wavg_exprs()).row(0, named=True)

        # --- Merge all results into a single meta row ---
        result = {"portfolioKey": key, "date": date}
        result.update(base_vals)
        result.update(bwic_vals)
        result.update(owic_vals)
        result.update(width_vals)
        result.update(skew_vals)

        changed = tuple(k for k in result if k not in ("portfolioKey", "date"))
        await log.rules(f"[wavg_levels] EMIT {len(changed)} cols to {key}.META")

        return Delta(
            frame=pl.DataFrame([result]),
            pk_columns=("portfolioKey", "date"),
            changed_columns=changed,
            mode="update",
        )
    except Exception as e:
        await log.error("wavg_levels error:", e=str(e))
        return None


# ---------------------------------------------------------------------------
# Rule 1: pct_priced_update  (cross-grid: PORTFOLIO -> META)
# ---------------------------------------------------------------------------

@rule(
    name="pct_priced_update",
    room_pattern="*.PORTFOLIO",
    target_grid_id="meta",
    target_primary_keys=("portfolioKey", "date"),
    column_triggers_any=("newLevel", "isReal") + NEW_LEVEL_COLS,
    depends_on_all=(
        RuleDependency("new_level_expand", DepMode.FINISHED),
        RuleDependency("clear_levels", DepMode.FINISHED),
        RuleDependency("s3_enrichment_stream", DepMode.FINISHED),
    ),
    priority=Priority.LOW,
    emit_mode=EmitMode.END,
    declared_column_outputs=("pctPriced",),
)
async def pct_priced_update(ctx: RuleContext):
    """Update pctPriced on meta grid: % of non-null newLevel among isReal=1 rows."""
    try:
        await log.rules("[pct_priced_update] START")
        basket = await ctx.running_frame_slice(["newLevel", "isReal", "rfqCreateDate"])
        await log.rules(f"[pct_priced_update] basket height={basket.height}")
        if basket.is_empty():
            await log.rules("[pct_priced_update] EXIT: basket is empty")
            return None

        real_rows = basket.filter(pl.col("isReal").cast(pl.Int8, strict=False) == 1)
        total = real_rows.height
        await log.rules(f"[pct_priced_update] isReal=1 rows: {total}")
        if total == 0:
            await log.rules("[pct_priced_update] EXIT: no isReal=1 rows")
            return None

        priced = real_rows.filter(pl.col("newLevel").is_not_null()).height
        pct = round(priced / total, 4)
        await log.rules(f"[pct_priced_update] priced={priced}/{total} = {pct}")

        key = basket.hyper.peek("portfolioKey")
        date = basket.hyper.peek("rfqCreateDate")
        await log.rules(f"[pct_priced_update] EMIT pctPriced={pct} to {key}.META")

        return Delta(
            frame=pl.DataFrame([{"portfolioKey": key, "date": date, "pctPriced": pct}]),
            pk_columns=("portfolioKey", "date"),
            changed_columns=("pctPriced",),
            mode="update",
        )
    except Exception as e:
        await log.error("pct_priced_update error:", e=str(e))
        return None


# ---------------------------------------------------------------------------
# Rule 2: ref_level_expand  (portfolio -> portfolio)
# ---------------------------------------------------------------------------
# Front-end contract: when editing refLevel, the client MUST also send
# two companion columns in the same delta payload:
#   - activeQuoteType : "client" | "price" | "spread" | "mmy" | "dm"
#   - activeSide      : "Bid" | "Mid" | "Ask"
# These reflect the page-level observables:
#   page.activeQuoteType$.get('activeQuoteType')
#   page.activeRefMarketSide.get('activeSide')
# ---------------------------------------------------------------------------

@rule(
    name="ref_level_expand",
    room_pattern="*.PORTFOLIO",
    column_triggers_any=("refLevel",),
    priority=Priority.HIGH,
    emit_mode=EmitMode.IMMEDIATE,
    declared_column_outputs=ALL_MANUAL_COLS + ("manualRefMktUser", "manualRefreshTime"),
)
async def ref_level_expand(ctx: RuleContext):
    """Expand refLevel into manualBid/Mid/Ask columns for the active quote type,
    inferring bid/ask width from reference markets when side is Mid."""
    try:
        # Build the list of reference-market columns we need for the waterfall
        ref_cols = [
            f"{mkt}{side}{qt}"
            for mkt in REF_MARKET_WATERFALL
            for side in ("Bid", "Ask")
            for qt in MANUAL_QT_SUFFIXES
        ]
        extra = ["quoteType", "activeQuoteType", "activeSide"] + ref_cols

        delta = await ctx.running_delta_slice(columns=extra)
        if delta.is_empty():
            return None

        pk_cols = list(ctx.target_pks)
        user_data = ctx.ingress_user
        username = user_data.displayName if user_data else "UNKNOWN"
        now_utc = datetime.now(timezone.utc)

        rows = []
        for row in delta.iter_rows(named=True):
            ref_level = row.get("refLevel")
            if ref_level is None:
                continue
            try:
                ref_level = float(ref_level)
            except (ValueError, TypeError):
                continue

            # --- resolve quote-type suffix ---
            active_qt = str(row.get("activeQuoteType") or "").strip()
            if active_qt.lower() == "client":
                qt_raw = str(row.get("quoteType") or "").upper().strip()
                suffix = CLIENT_QT_TO_SUFFIX.get(qt_raw)
            else:
                suffix = ACTIVE_QT_TO_SUFFIX.get(active_qt.lower())

            if suffix is None:
                continue  # unrecognised quote type — skip row

            # --- resolve bid / mid / ask values ---
            active_side = str(row.get("activeSide") or "").strip().capitalize()

            bid_val = ref_level
            mid_val = ref_level
            ask_val = ref_level

            if active_side == "Mid":
                # Waterfall: macp -> cbbt -> bval -> house
                for mkt in REF_MARKET_WATERFALL:
                    mkt_ask = row.get(f"{mkt}Ask{suffix}")
                    mkt_bid = row.get(f"{mkt}Bid{suffix}")
                    if mkt_ask is not None and mkt_bid is not None:
                        try:
                            full_width = float(mkt_ask) - float(mkt_bid)
                            if full_width >= 0:
                                half = full_width / 2.0
                                bid_val = ref_level - half
                                mid_val = ref_level
                                ask_val = ref_level + half
                                break
                        except (ValueError, TypeError):
                            continue
                # If no ref market found, bid = mid = ask = refLevel (already set)

            # --- build output row ---
            out_row = {pk: row[pk] for pk in pk_cols}

            # Clear ALL manual columns first
            for col in ALL_MANUAL_COLS:
                out_row[col] = None

            # Set the three columns for the active quote type
            out_row[f"manualBid{suffix}"] = bid_val
            out_row[f"manualMid{suffix}"] = mid_val
            out_row[f"manualAsk{suffix}"] = ask_val

            # Always stamp user + timestamp
            out_row["manualRefMktUser"] = username
            out_row["manualRefreshTime"] = now_utc

            rows.append(out_row)

        if rows:
            return pl.DataFrame(rows)
        return None
    except Exception as e:
        await log.error("ref_level_expand error:", e=str(e))
        return None


# --- Registration --------------------------------------------------------------


def register_portfolio_rules(engine: RulesEngine):
    """Register all portfolio rules with the rules engine."""

    # Micro-grid rules
    from app.services.rules.micro_grid_rules import register_micro_grid_rules
    register_micro_grid_rules(engine)

    # Meta Rules
    engine.register(pct_priced_update)         # Cross-grid: pctPriced = non-null newLevel % among isReal=1
    engine.register(wavg_levels)               # Cross-grid: wavg levels, bwic/owic, skews → META
    # engine.register(meta_update)             # (superseded by pct_priced_update)
    engine.register(test_in_name)              # if 'test' in client name, state -> TEST
    engine.register(state_check)               # if state in TEST, etc. -> isReal to False
    engine.register(rank_update)               # if won=1, cover=2, etc.

    # Market Rules
    engine.register(clear_levels)                # When user deletes newLevel, clear other newLevels
    engine.register(build_streaming_s3_rule()) # S3 conversions
    engine.register(refresh_all_markets)       # Triggers market refresh
    # engine.register(market_propegate)        # When ref markets updates -> update dependent skews
    engine.register(new_level_expand)          # set newLevel when matching quote type is defined
    engine.register(ref_level_expand)          # Expand refLevel into manual bid/mid/ask columns

    # Trader Rules
    engine.register(trader_claim)              # Trader claimed a bond
    engine.register(fuzzy_desig)               # Match trader name input
    engine.register(fuzzy_assigned)            # Match trader name input
    engine.register(edit_audit)                # Record who made the edit
    engine.register(removed_bond)              # Update for removal
    # engine.register(desig_static_enhance)    # Update trader static

    # Risk
    engine.register(risk_adjust)               # When we adjust the size, adjust the dv01 as well
    engine.register(dv01_adjust_meta)
