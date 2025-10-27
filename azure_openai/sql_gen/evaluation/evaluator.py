"""  
evaluator.py   – reference implementation of a 1-to-5 SQL answer grader  
                using OpenAI’s new Response API and function-calling.  
  
Author:        <you>  
Python:        3.10+  
Third-party:   pip install openai azure-identity  
"""  
  
from __future__ import annotations  
  
import json  
import math  
import sqlite3  
import uuid  
from dataclasses import dataclass  
from datetime import date  
from typing import Any, List, Tuple  
  
from openai import AzureOpenAI  
from azure.identity import DefaultAzureCredential, get_bearer_token_provider 
import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file 
import os
  
  
# ──────────────────────────────────────────────────────────────────────  
# 0.  CONFIGURATION  – edit to match your environment  
# ──────────────────────────────────────────────────────────────────────  
AZURE_OPENAI_ENDPOINT  = os.getenv("SQL_JUDGE_ENDPOINT")  # <- EDIT  
AZURE_OPENAI_API_VER   = "preview"                                  # fixed for Response API  
AZURE_OPENAI_DEPLOYMENT= "gpt-4.1"                                   # <- EDIT  
SQLITE_DB_PATH         = "../data/northwind.db"                         # <- EDIT  
SQL_JUDGE_API_KEY = os.getenv("SQL_JUDGE_API_KEY")  # <- EDIT

  
def _standardize_answer(ans: Any) -> List[Tuple]:  
    """  
    Convert *any* reasonable answer representation into  
    a list[tuple] so that the existing diff-logic can stay untouched.  
  
    Accepted formats  
    • scalar                        → [(value,)]  
    • list of scalars               → [(v1,), (v2,), ...]  
    • list/tuple of lists/tuples    → [tuple(row), ...]  
    • dict                          → [(v1, v2, ...)]  (sorted keys)  
    • list of dicts                 → as above, one per element  
    """  
    # scalar  
    if isinstance(ans, (int, float, str, type(None))):  
        return [(ans,)]  
  
    # list / tuple  
    if isinstance(ans, (list, tuple)):  
        if not ans:                       # empty → empty table  
            return []  
        first = ans[0]  
        # list of rows already  
        if isinstance(first, (list, tuple)):  
            return [tuple(_normalize(x) for x in row) for row in ans]  
        # list of dicts  
        if isinstance(first, dict):  
            keys = sorted(first.keys())  
            return [tuple(_normalize(row[k]) for k in keys) for row in ans]  
        # list of scalars  
        return [(x,) for x in ans]  
  
    # dict → single “row”  
    if isinstance(ans, dict):  
        keys = sorted(ans.keys())  
        return [tuple(_normalize(ans[k]) for k in keys)]  
  
    raise ValueError("Unsupported answer format supplied by the model/API.")    
# ─────────────────────────────────────────────────────────────  
# 1.  STATIC  KNOWLEDGE GRAPH  BITS  (complete list of 9 metrics)  
# ─────────────────────────────────────────────────────────────  
METRICS: dict[str, dict[str, Any]] = {  
    # 1  Gross Profit  ────────────────────────────────────────  
    "Gross Profit": {  
        "tables": {"Orders", "order_details"},  
        "sql": """  
            SELECT  
                {dims}  
                SUM(od.UnitPrice * od.Quantity * (1 - od.Discount))  
              - SUM(o.Freight)                       AS value  
            FROM Orders o  
            JOIN order_details od ON o.OrderID = od.OrderID  
            {extra_joins}  
            WHERE 1=1  
              {date_filter}  
            {group_by_clause}  
        """,  
    },  
  
    # 2  Freight-to-Sales Ratio  ──────────────────────────────  
    "Freight to Sales Ratio": {  
        "tables": {"Orders", "order_details"},  
        "sql": """  
            SELECT  
                {dims}  
                SUM(o.Freight) * 1.0 /  
                NULLIF(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 0)  
                AS value  
            FROM Orders o  
            JOIN order_details od ON o.OrderID = od.OrderID  
            {extra_joins}  
            WHERE 1=1  
              {date_filter}  
            {group_by_clause}  
        """,  
    },  
  
    # 3  Late Shipment Rate  ─────────────────────────────────  
    "Late Shipment Rate": {  
        "tables": {"Orders"},  
        "sql": """  
            SELECT  
                {dims}  
                COUNT(CASE WHEN o.ShippedDate > o.RequiredDate THEN 1 END) * 1.0  
              / COUNT(o.OrderID)                AS value  
            FROM Orders o  
            {extra_joins}  
            WHERE 1=1  
              {date_filter}  
            {group_by_clause}  
        """,  
    },  
  
    # 4  Regional Freight Distribution  (not time-based)  ────  
    "Regional Freight Distribution": {  
        "tables": {"Orders", "Employees", "EmployeeTerritories",  
                   "Territories", "Regions"},  
        "sql": """  
            SELECT  
                Regions.RegionDescription         AS region,  
                SUM(o.Freight)                    AS value  
            FROM Orders o  
            JOIN Employees            e  ON o.EmployeeID  = e.EmployeeID  
            JOIN EmployeeTerritories  et ON e.EmployeeID  = et.EmployeeID  
            JOIN Territories          t  ON et.TerritoryID = t.TerritoryID  
            JOIN Regions                 ON t.RegionID     = Regions.RegionID  
            GROUP BY Regions.RegionDescription  
        """,   # ← no {date_filter} because the metric is snapshot-style  
    },  
  
    # 5  Multi-Item Order Percentage  ─────────────────────────  
    "Multi-Item Order Percentage": {  
        "tables": {"Orders"},  
        "sql": """  
            SELECT  
                {dims}  
                SUM(  
                    CASE  
                        WHEN (SELECT COUNT(*)  
                              FROM order_details od2  
                              WHERE od2.OrderID = o.OrderID) > 1  
                        THEN 1 ELSE 0 END  
                ) * 1.0 / COUNT(o.OrderID)        AS value  
            FROM Orders o  
            {extra_joins}  
            WHERE 1=1  
              {date_filter}  
            {group_by_clause}  
        """,  
    },  
  
    # 6  Reorder Necessity Index  ────────────────────────────  
    "Reorder Necessity Index": {  
        "tables": {"Products"},  
        "sql": """  
            SELECT  
                SUM(  
                    CASE WHEN p.UnitsInStock + p.UnitsOnOrder  
                              < p.ReorderLevel THEN 1 ELSE 0 END  
                ) * 1.0 / COUNT(p.ProductID)       AS value  
            FROM Products p  
            {group_by_clause}  
        """,   # not time-dependent and no Order-based joins  
    },  
  
    # 7  High-Value Account Ratio  ───────────────────────────  
    "High-Value Account Ratio": {  
        "tables": {"Orders", "order_details"},  
        "sql": """  
            WITH cust_sales AS (  
                SELECT c.CustomerID,  
                       SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS sales  
                FROM Customers c  
                JOIN Orders        o  ON c.CustomerID = o.CustomerID  
                JOIN order_details od ON o.OrderID    = od.OrderID  
                WHERE 1=1  
                  {date_filter}  
                GROUP BY c.CustomerID  
            )  
            SELECT  
                SUM(CASE WHEN sales > 10000 THEN 1 ELSE 0 END) * 1.0  
              / COUNT(*)                              AS value  
            FROM cust_sales  
        """,  
    },  
  
    # 8  Order Processing Speed (days)  ──────────────────────  
    "Order Processing Speed": {  
        "tables": {"Orders"},  
        "sql": """  
            SELECT  
                {dims}  
                AVG(julianday(o.ShippedDate) - julianday(o.OrderDate)) AS value  
            FROM Orders o  
            {extra_joins}  
            WHERE 1=1  
              AND o.ShippedDate IS NOT NULL  
              {date_filter}  
            {group_by_clause}  
        """,  
    },  
  
    # 9  Year-over-Year Growth  ──────────────────────────────  
    "Year-over-Year Growth": {  
        "tables": {"Orders", "order_details"},  
        "sql": """  
            /* YoY growth = (latest-year sales – previous-year sales) / previous */  
            WITH yearly AS (  
                SELECT CAST(strftime('%Y', o.OrderDate) AS INT)      AS yr,  
                       SUM(od.UnitPrice * od.Quantity * (1 - od.Discount))  
                                                                AS sales  
                FROM Orders o  
                JOIN order_details od ON o.OrderID = od.OrderID  
                {extra_joins}  
                GROUP BY yr  
            ),  
            calc AS (  
                SELECT  
                    (   (SELECT sales FROM yearly  
                         WHERE yr = (SELECT MAX(yr) FROM yearly))  
                      - (SELECT sales FROM yearly  
                         WHERE yr = (SELECT MAX(yr) FROM yearly) - 1)  
                    ) * 1.0  
                    / NULLIF(  
                         (SELECT sales FROM yearly  
                          WHERE yr = (SELECT MAX(yr) FROM yearly) - 1), 0  
                    ) AS value  
            )  
            SELECT {dims} value  
            FROM   calc  
            {group_by_clause}  
        """,  
    },  
}   
# quick helper: columns that obviously belong to a table and need a join  
DIMENSION_TABLE_MAP = {  
    "Customers.": ("Customers", "LEFT JOIN Customers c ON o.CustomerID = c.CustomerID"),  
    "Employees.": ("Employees",  "LEFT JOIN Employees  e ON o.EmployeeID = e.EmployeeID"),  
    "Regions.":   (  
        "Regions",  
        """JOIN Employees e           ON o.EmployeeID = e.EmployeeID  
           JOIN EmployeeTerritories et ON e.EmployeeID = et.EmployeeID  
           JOIN Territories t          ON et.TerritoryID = t.TerritoryID  
           JOIN Regions                ON t.RegionID = Regions.RegionID""",  
    ),  
    # extend if you need more  
}  
  
  
# ──────────────────────────────────────────────────────────────────────  
# 2.  DATA CLASSES  
# ──────────────────────────────────────────────────────────────────────  
@dataclass  
class ParsedQuestion:  
    metric: str  
    start_date: str | None = None  
    end_date:   str | None = None  
    dimensions: List[str] | None = None  
  
  
@dataclass  
class DiffSummary:  
    kind: str                    # "scalar" | "table"  
    rel_error: float | None = None  
    f1: float | None       = None  
  
  
@dataclass  
class EvalResult:  
    score: int  
    comment: str  
  
  
# ──────────────────────────────────────────────────────────────────────  
# 3.  LOW-LEVEL UTILITIES  
# ──────────────────────────────────────────────────────────────────────  
def _execute_sql(cursor: sqlite3.Cursor, sql: str) -> List[Tuple]:  
    """Run SQL and fetch *all* rows (no autocommit mutations expected)."""  
    try:  
        cursor.execute(sql)  
        rows = cursor.fetchall()  
        return rows  
    except sqlite3.Error as exc:  
        raise RuntimeError(f"SQL failed: {exc}\n--- SQL ---\n{sql}")  
  
  
def _rows_to_set(rows: List[Tuple]) -> set[Tuple]:  
    """Order-insensitive representation for set diff / precision-recall."""  
    return {tuple(_normalize(x) for x in row) for row in rows}  
  
  
def _normalize(x: Any) -> Any:  
    """Round floats for stable comparison; leave others untouched."""  
    if isinstance(x, float):  
        return round(x, 8)  
    return x  
  
  
# ──────────────────────────────────────────────────────────────────────  
# 4.  GROUND-TRUTH  SQL GENERATOR   (deterministic)  
# ──────────────────────────────────────────────────────────────────────  
def build_metric_sql(pq: ParsedQuestion) -> str:  
    meta     = METRICS[pq.metric]  
    base_sql = meta["sql"]  
  
    # ── handle dimensions ─────────────────────────────  
    dims_parts, group_cols, extra_joins = [], [], ""  
    if pq.dimensions:  
        for dim in pq.dimensions:  
            prefix = dim.split(".")[0] + "."  
            if prefix in DIMENSION_TABLE_MAP:  
                extra_joins += "\n" + DIMENSION_TABLE_MAP[prefix][1]  
            dims_parts.append(f"{dim} AS `{dim}`")  
            group_cols.append(dim)  
  
    dims = (", ".join(dims_parts) + ",") if dims_parts else ""  
  
    # ── GROUP BY clause ──────────────────────────────  
    if group_cols:                        # at least one dimension  
        group_by_clause = "GROUP BY " + ", ".join(group_cols)  
    else:                                 # no dimensions → no GROUP BY  
        group_by_clause = ""  
  
    # ── date filter ─────────────────────────────────  
    date_filter = ""  
    if pq.start_date and pq.end_date and "Orders" in meta["tables"]:  
        date_filter = (f" AND date(o.OrderDate) "  
                       f"BETWEEN '{pq.start_date}' AND '{pq.end_date}'")  
  
    # ── fill template ───────────────────────────────  
    sql_filled = base_sql.format(  
        dims=dims,  
        extra_joins=extra_joins,  
        date_filter=date_filter,  
        group_by_clause=group_by_clause,  
    )  
    return sql_filled.strip()    
  
# ──────────────────────────────────────────────────────────────────────  
# 5.  OPENAI RESPONSE-API  CLIENT  SETUP  
# ──────────────────────────────────────────────────────────────────────  
oaiclient = AzureOpenAI(  
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/v1/",  
    api_version=AZURE_OPENAI_API_VER,  
    api_key=SQL_JUDGE_API_KEY,  # use API key for auth
)  
  
  
# ──────────────────────────────────────────────────────────────────────  
# 6.  TOOL  DECLARATIONS  (for function-calling)  
# ──────────────────────────────────────────────────────────────────────  
TOOLS = [  
    # ────────────────────────────────────────────────  
    # 1.  parse_question  
    # ────────────────────────────────────────────────  
    {  
        "type": "function",  
        "name": "parse_question",  
        "description": (  
            "Convert ONE English business question about the Northwind dataset "  
            "into a structured request for a pre-defined metric."  
        ),  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "metric": {  
                    "type": "string",  
                    "enum": [  
                        "Gross Profit",  
                        "Freight to Sales Ratio",  
                        "Late Shipment Rate",  
                        "Regional Freight Distribution",  
                        "Multi-Item Order Percentage",  
                        "Reorder Necessity Index",  
                        "High-Value Account Ratio",  
                        "Order Processing Speed",  
                        "Year-over-Year Growth"  
                    ],  
                    "description": "Exactly one KPI name from the enum."  
                },  
                "start_date": {  
                    "type": "string",  
                    "description": (  
                        "Inclusive first date (YYYY-MM-DD). "  
                        "Null when the question gives no period."  
                    )  
                },  
                "end_date": {  
                    "type": "string",  
                    "description": (  
                        "Inclusive last date (YYYY-MM-DD). "  
                        "Null when the question gives no period."  
                    )  
                },  
                "dimensions": {  
                    "type": ["array"],  
                    "description": (  
                        "Optional grouping columns from like "  
                        " 'Customers.Country','Customers.City','Regions.RegionDescription','Employees.LastName'"  
                    ),  
                    "items": {  
                        "type": "string",  
                    },  
                }  
            },  
            "required": ["metric"],  
            "additionalProperties": False  
        }  
    },  
  
    # # ────────────────────────────────────────────────  
    # # 2.  adjust_score  
    # # ────────────────────────────────────────────────  
    # {  
    #     "type": "function",  
    #     "name": "adjust_score",  
    #     "description": (  
    #         "Called only after an automatic comparison pass has produced a "  
    #         "provisional 1-to-5 score.  May shift the score by ±1 and must "  
    #         "explain why."  
    #     ),  
    #     "parameters": {  
    #         "type": "object",  
    #         "properties": {  
    #             "final_score": {  
    #                 "type": "integer",  
    #                 "description": (  
    #                     "New score (1-5). Must differ from the provisional "  
    #                     "score by no more than ±1."  
    #                 )  
    #             },  
    #             "comment": {  
    #                 "type": "string",  
    #                 "description": "≤200-character human-readable justification."  
    #             }  
    #         },  
    #         "required": ["final_score", "comment"],  
    #         "additionalProperties": False  
    #     }  
    # }  
]  
# ──────────────────────────────────────────────────────────────────────  
# 7.  HIGH-LEVEL  EVALUATOR  ENTRYPOINT  
# ──────────────────────────────────────────────────────────────────────  
# ──────────────────────────────────────────────────────────  
# 7.  HIGH-LEVEL  EVALUATOR  ENTRYPOINT  
# ──────────────────────────────────────────────────────────  
def evaluate(question: str, candidate_sql: str) -> EvalResult:  
    """  
    1. Runs the candidate SQL against the Northwind DB to obtain a result.  
    2. Uses GPT-4 (function-calling) to convert the *question* — not the  
       candidate SQL — into a metric description.  
    3. Builds and executes a *deterministic* ground-truth query for that  
       metric.  
    4. Diffs the two result sets and buckets a 1-to-5 score.  
    5. Lets GPT optionally shift the score by ±1 and attach a comment.  
  
    NOTE: we treat the candidate SQL as an opaque string; we only execute  
          it.  All reasoning about “what was asked” is based on the  
          natural-language question, not on the SQL the model/API wrote.  
    """  
    # 7-1.  execute the candidate query --------------------------------  
    with sqlite3.connect(SQLITE_DB_PATH) as conn:  
        cur = conn.cursor()  
        try:  
            cand_rows = _execute_sql(cur, candidate_sql)  
        except RuntimeError as err:  
            # SQL crashed → automatic grade 1  
            return EvalResult(score=1, comment=str(err))  
  
    # 7-2.  ask GPT to parse the question into a metric ----------------  
    sys_msg = {  
        "role": "system",  
        "content": (  
            "You are a deterministic parser that converts ONE English "  
            "business question about the Northwind dataset into a JSON "  
            "function call selecting exactly one metric name.  "  
            "Output *only* function calls."  
        ),  
    }  
    first = oaiclient.responses.create(  
        model=AZURE_OPENAI_DEPLOYMENT,  
        tools=TOOLS,  
        input=[sys_msg, {"role": "user", "content": question}],  
    )  
  
    call = first.output[0]  
    if call.type != "function_call" or call.name != "parse_question":  
        raise RuntimeError("Parser LLM did not call parse_question().")  
  
    pq_args = json.loads(call.arguments)  
    pq      = ParsedQuestion(**pq_args)  
  
    # 7-3.  build + execute deterministic ground-truth SQL -------------  
    with sqlite3.connect(SQLITE_DB_PATH) as conn:  
        cur = conn.cursor()  
        gt_sql  = build_metric_sql(pq)  
        gt_rows = _execute_sql(cur, gt_sql)  
  
    # 7-4. diff the two result sets -----------------------------------  
    diff        = _diff_results(gt_rows, cand_rows)  
    provisional = _bucket_score(diff)  
    print(f"Provisional score: {provisional} (diff={diff})")
  
    # 7-5. give GPT a chance to tweak the score -----------------------  
    #      (NO SQL STRINGS ARE PASSED)  
    # adj_input = [  
    #     {  
    #         "type": "function_call_output",  
    #         "call_id": call.call_id,  
    #         "output": json.dumps(pq_args),  
    #     },  
    #     {  
    #         "role": "user",  
    #         "content": json.dumps(  
    #             {  
    #                 "provisional_score": provisional,  
    #                 "diff": diff.__dict__,  
    #                 # SQL is *not* shown to the model  
    #             },  
    #             indent=2,  
    #         ),  
    #     },  
    # ]  
    # final = oaiclient.responses.create(  
    #     model=AZURE_OPENAI_DEPLOYMENT,  
    #     previous_response_id=first.id,  
    #     tools=TOOLS,  
    #     input=adj_input,  
    # )  
  
    # out = final.output[0]  
    # if out.type == "function_call" and out.name == "adjust_score":  
    #     payload = json.loads(out.arguments)  
    #     return EvalResult(score=payload["final_score"],  
    #                       comment=payload["comment"])  
  
    return EvalResult(score=provisional,  
                      comment="Auto-graded; LLM gave no extra comment.")  
  
# ──────────────────────────────────────────────────────────────────────  
# 8.  DIFF & SCORING HELPERS  
# ──────────────────────────────────────────────────────────────────────  
def _diff_results(gt_rows: List[Tuple], cand_rows: List[Tuple]) -> DiffSummary:  
    if len(gt_rows) == 1 and len(gt_rows[0]) == 1:  
        # scalar  
        gt_val   = gt_rows[0][0]  
        cand_val = cand_rows[0][0] if cand_rows else None  
        try:  
            rel_err = abs(cand_val - gt_val) / (abs(gt_val) + 1e-9)  
        except Exception:  
            rel_err = math.inf  
        return DiffSummary(kind="scalar", rel_error=rel_err)  
  
    # otherwise treat as table  
    gt_set   = _rows_to_set(gt_rows)  
    cand_set = _rows_to_set(cand_rows)  
    tp = len(gt_set & cand_set)  
    fp = len(cand_set - gt_set)  
    fn = len(gt_set - cand_set)  
    precision = tp / (tp + fp + 1e-9)  
    recall    = tp / (tp + fn + 1e-9)  
    f1 = 2*precision*recall / (precision+recall+1e-9)  
    return DiffSummary(kind="table", f1=f1)  
  
  
def _bucket_score(diff: DiffSummary) -> int:  
    if diff.kind == "scalar":  
        if diff.rel_error <= 0.001:  
            return 5  
        if diff.rel_error <= 0.01:  
            return 4  
        if diff.rel_error <= 0.10:  
            return 3  
        if diff.rel_error <= 0.25:  
            return 2  
        return 1  
    else:  # table  
        if diff.f1 >= 0.95:  
            return 5  
        if diff.f1 >= 0.85:  
            return 4  
        if diff.f1 >= 0.70:  
            return 3  
        if diff.f1 >= 0.50:  
            return 2  
        return 1  
  
  
# ──────────────────────────────────────────────────────────────────────  
# 9.  CLI  DEMO  (python evaluator.py "question" "SQL")  
# ──────────────────────────────────────────────────────────────────────  
if __name__ == "__main__":  
    import sys, textwrap  
    if len(sys.argv) != 3:  
        print(  
            textwrap.dedent(  
                """\  
                Usage:  
                    python evaluator.py "<question>" "<candidate SQL>"  
  
                Example:  
                    python evaluator.py \\  
                      "What is the late shipment rate for 2021?" \\  
                      "SELECT COUNT(CASE WHEN ShippedDate>RequiredDate THEN 1 END)*1.0/COUNT(OrderID) FROM Orders WHERE OrderDate BETWEEN '2021-01-01' AND '2021-12-31';"  
                """  
            )  
        )  
        sys.exit(1)  
  
    q, sql = sys.argv[1:3]  
    result = evaluate(q, sql)  
    print("Final score :", result.score)  
    print("Comment     :", result.comment)  