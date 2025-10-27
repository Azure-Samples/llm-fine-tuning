#!/usr/bin/env python3  
"""  
Evaluate two Azure-OpenAI deployments against a set of SQL questions  
using a purely automatic 1-to-5 grader (no LLM judge).  
"""  
import os, json, re, sqlite3, signal, math  
from concurrent.futures import ThreadPoolExecutor  
from dotenv import load_dotenv  
from tenacity import retry, stop_after_attempt, wait_fixed  
from openai import AzureOpenAI  
import threading
  
# ──────────────────────────────────────────────────────────────  
# 0.  Configuration & helpers  
# ──────────────────────────────────────────────────────────────  
load_dotenv()  
  
# Azure–OpenAI (generation only)  
AZURE_OPENAI_ENDPOINT     = os.getenv("AZURE_OPENAI_ENDPOINT")  
AZURE_OPENAI_API_KEY      = os.getenv("AZURE_OPENAI_API_KEY")  
AZURE_OPENAI_API_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION")  
DEPLOYMENT1               = os.getenv("DEPLOYMENT1")    # without reasoning  
DEPLOYMENT2               = os.getenv("DEPLOYMENT2")    # with reasoning  
  
DB_PATH   = "../data/northwind.db"      # read-only snapshot  
TIMEOUT_S = 5                           # seconds per query  
ROW_LIMIT = 10_000                      # safety valve  
  
# ──────────────────────────────────────────────────────────────  
# 1.  Utility: extract the SQL block returned by the LLM  
# ──────────────────────────────────────────────────────────────  
def extract_sql_query(output: str) -> str:  
    blocks = re.findall(r"```sql\s+(.*?)```", output, re.DOTALL | re.IGNORECASE)  
    return blocks[-1].strip() if blocks else output.strip()  
  
def has_order_by(sql: str) -> bool:  
    """Return True if the *outermost* query contains ORDER BY."""  
    sql = re.sub(r"--.*", "", sql)                       # line comments  
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL) # block comments  
    sql = re.sub(r"\s+", " ", sql).upper()  
    pos = sql.rfind(" ORDER BY ")  
    if pos == -1:  
        return False  
    # crude heuristic: if a close-paren appears after ORDER BY we're  
    # probably inside a sub-query  
    return ")" not in sql[pos:]  
  
  
# ──────────────────────────────────────────────────────────────  
# 2.  Safe query runner (timeout + row limit)  
# ──────────────────────────────────────────────────────────────  
def _strip_trailing_semicolon(sql: str) -> str:  
    """  
    Remove a single semicolon that terminates the statement.  
    Keeps semicolons inside strings or between statements untouched.  
    """  
    # 1. trim right-hand whitespace  
    sql = sql.rstrip()  
    # 2. if the last non-space character is a semicolon — drop it  
    if sql.endswith(";"):  
        sql = sql[:-1]  
    return sql  

def _run_sql(sql: str, conn: sqlite3.Connection):  
    """Run SQL with a row limit and (when possible) a 5-s timeout."""  
    sql = _strip_trailing_semicolon(sql)
    main_thread = threading.current_thread() is threading.main_thread()  
  
    if main_thread:  
        def _timeout(sig, frame): raise TimeoutError("query timed out")  
        signal.signal(signal.SIGALRM, _timeout)  
        signal.alarm(TIMEOUT_S)  
  
    try:  
        cur  = conn.execute(f"SELECT * FROM ({sql}) LIMIT {ROW_LIMIT}")  
        rows = cur.fetchall()  
        ncol = len(cur.description) if cur.description else 0  
    finally:  
        if main_thread:  
            signal.alarm(0)     # cancel alarm  
  
    return ncol, rows  
  
  
# ──────────────────────────────────────────────────────────────  
# 3.  Normalisation helpers  
# ──────────────────────────────────────────────────────────────  
def _norm_cell(v):  
    if isinstance(v, float):  
        return round(v, 8)  
    if isinstance(v, (bytes, bytearray)):  
        v = v.decode()  
    return str(v).strip()  
  
def _norm_row(row):  
    return tuple(_norm_cell(c) for c in row)  
  
  
# ──────────────────────────────────────────────────────────────  
# 4.  Grader  
# ──────────────────────────────────────────────────────────────  
def grade_two_results(gt_cols, gt_rows, cand_cols, cand_rows, order_matters):  
    """  
    Return (score 1-5, explanation).  
    """  
    # ─── Runtime errors / shape mismatch ─────────────────────  
    if gt_rows is None or cand_rows is None:  
        return 1, "candidate or ground-truth failed to run"  
  
    # ─── SCALAR  (1 row, 1 column, numeric) ──────────────────  
    if gt_cols == cand_cols == 1 and len(gt_rows) == len(cand_rows) == 1:  
        try:  
            gt_val   = float(gt_rows[0][0])  
            cand_val = float(cand_rows[0][0])  
        except Exception:  
            pass              # not numeric → fall through to table  
        else:  
            abs_err = abs(cand_val - gt_val)  
            if abs(gt_val) < 1e-9:  
                rel_err = math.inf  
            else:  
                rel_err = abs_err / abs(gt_val)  
  
            if rel_err <= 0.001:  
                return 5, f"rel_error={rel_err:.4f} ≤0.001"  
            elif rel_err <= 0.01:  
                return 4, f"rel_error={rel_err:.4f}"  
            elif rel_err <= 0.05:  
                return 3, f"rel_error={rel_err:.4f}"  
            elif rel_err <= 0.10:  
                return 2, f"rel_error={rel_err:.4f}"  
            return 1,  f"rel_error={rel_err:.4f}"  
  
    # ─── TABLE  (compare as sets) ─────────────────────────────  
    gt_set   = {_norm_row(r) for r in gt_rows}  
    cand_set = {_norm_row(r) for r in cand_rows}  
  
    if not gt_set and not cand_set:  
        return 5, "both result sets empty (correct)"  
  
    tp = len(gt_set & cand_set)  
    fp = len(cand_set - gt_set)  
    fn = len(gt_set - cand_set)  
  
    prec = tp / (tp + fp) if (tp + fp) else None  
    rec  = tp / (tp + fn) if (tp + fn) else None  
    f1   = (2*prec*rec/(prec+rec)) if prec and rec else 0.0  
  
    # bucket to 1-5  
    if f1 >= 0.97:  
        score = 5  
    elif f1 >= 0.90:  
        score = 4  
    elif f1 >= 0.80:  
        score = 3  
    elif f1 >= 0.60:  
        score = 2  
    else:  
        score = 1  
  
    expl = f"F1={f1:.2f}  tp={tp} fp={fp} fn={fn}"  
    return score, expl  
  
  
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))  
def compare_sql_results(predicted_sql, ground_truth_sql, question):  
    """  
    Run both queries and return (score 1-5, explanation string).  
    """  
    try:  
        with sqlite3.connect(DB_PATH) as conn:  
            gt_cols,  gt_rows   = _run_sql(ground_truth_sql, conn)  
            cand_cols, cand_rows = _run_sql(predicted_sql,   conn)  
    except Exception as exc:  
        return 1, f"runtime error: {str(exc)}"  
  
    order_matters = has_order_by(ground_truth_sql)  
    return grade_two_results(  
        gt_cols, gt_rows,  
        cand_cols, cand_rows,  
        order_matters  
    )  
  
  
# ──────────────────────────────────────────────────────────────  
# 5.  Query the language model (same as before)  
# ──────────────────────────────────────────────────────────────  
client_gen = AzureOpenAI(  
    azure_endpoint = AZURE_OPENAI_ENDPOINT,  
    api_key        = AZURE_OPENAI_API_KEY,  
    api_version    = AZURE_OPENAI_API_VERSION  
)  
  
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))  
def query_openai_deployment(deployment_name, question, kg_context):  
    date_format = "YYYY-MM-DD"  
    if '"date_format":' in kg_context:  
        try:  
            date_format = kg_context.split('"date_format":')[1].split(',')[0].strip()  
        except IndexError:  
            pass  
  
    prompt = f"""  
Your task is to write an efficient SQLite query that answers the USER  
question.  Use the schema and business context below.  Output only the  
SQL in a ```sql ...``` block.  
  
Database context:  
{kg_context}  
  
USER QUESTION:  
{question}  
  
Remember the date format is {date_format}.  
""".strip()  
  
    resp = client_gen.chat.completions.create(  
        model = deployment_name,  
        messages = [  
            {"role": "system", "content": "You are an expert SQL generator for SQLite."},  
            {"role": "user",   "content": prompt}  
        ]  
    )  
    return resp.choices[0].message.content  
  
  
# ──────────────────────────────────────────────────────────────  
# 6.  Process one test case  
# ──────────────────────────────────────────────────────────────  
def process_record(record, kg_context):  
    question      = record.get("user", "")  
    ground_truth  = record.get("sql_result", "")  
  
    results = {}  
    for dep in (DEPLOYMENT1, DEPLOYMENT2):  
        llm_out  = query_openai_deployment(dep, question, kg_context)  
        pred_sql = extract_sql_query(llm_out)  
        score, expl = compare_sql_results(pred_sql, ground_truth, question)  
  
        results[dep] = {  
            "score": score,  
            "pass" : score >= 4,  
            "explanation": expl  
        }  
  
    return question, results  
  
  
# ──────────────────────────────────────────────────────────────  
# 7.  Main runner  
# ──────────────────────────────────────────────────────────────  
def run_tests(test_data, kg_context):  
    aggregated = {  
        DEPLOYMENT1: {"score_sum": 0, "passes": 0, "total": len(test_data)},  
        DEPLOYMENT2: {"score_sum": 0, "passes": 0, "total": len(test_data)}  
    }  
  
    with ThreadPoolExecutor() as pool:  
        futures = [pool.submit(process_record, rec, kg_context)  
                   for rec in test_data]  
  
        for fut in futures:  
            question, res = fut.result()  
            print("─" * 80)  
            print("Q:", question)  
            for dep, info in res.items():  
                aggregated[dep]["score_sum"] += info["score"]  
                if info["pass"]:  
                    aggregated[dep]["passes"] += 1  
                print(f"  {dep}: score={info['score']}  "  
                      f"{'PASS' if info['pass'] else 'fail'}  "  
                      f"({info['explanation']})")  
  
    return aggregated  
  
  
def main():  
    # ── load test cases ───────────────────────────────────────  
    with open("../data/sql_result_test_v5.jsonl", "r") as f:  
        test_cases = [json.loads(line) for line in f]  
  
    # ── knowledge-graph context (sent to the LLMs) ────────────  
    with open("../data/analytic_graph.json", "r") as f:  
        kg_context = json.dumps(json.load(f), indent=2)  
  
    # ── run the evaluation ────────────────────────────────────  
    aggregated = run_tests(test_cases, kg_context)  
  
    print("\nSummary\n" + "="*60)  
    for dep, meta in aggregated.items():  
        avg_score = meta["score_sum"] / meta["total"]  
        pass_rate = meta["passes"]    / meta["total"]  
        print(f"{dep}:  avg score = {avg_score:.2f} / 5   "  
              f"pass rate (score≥4) = {pass_rate:.1%}")  
  
  
if __name__ == "__main__":  
    main()  