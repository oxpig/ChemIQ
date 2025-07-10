import pandas as pd

def panel_table_latex(
        subset_df: pd.DataFrame,
        group_keys,
        group_to_df,
        *,
        title: str,
        question_order: list[str] | None = None,          # top-level ordering
        sub_order: dict[str, list[str]] | None = None,    # per-category ordering
        label: str | None = None,
        caption: str | None = None,
    ) -> None:
    r"""
    Render a LaTeX panel table (one stub column with indented sub-categories).
    """

    # ───────────────────────────────────────── PREP
    cat_cols = ["question_category", "sub_category"]

    # denominator per (category, sub-category)
    denom_df = (
        subset_df.groupby(cat_cols)
                 .agg(total=("uuid", "nunique"))
                 .reset_index()
    )

    # per-group correct counts per (category, sub)
    stats = {}
    # overall denominators / numerators for the TOTAL row
    total_correct = {}
    total_total   = {}

    for key in group_keys:
        grp_df = group_to_df(key)

        # store overall numbers for TOTAL row
        total_total[key]   = grp_df["uuid"].nunique()
        total_correct[key] = grp_df.loc[grp_df["is_correct"], "uuid"].nunique()

        stats[key] = (
            grp_df.groupby(cat_cols)
                  .agg(correct=("is_correct", "sum"))
                  .reset_index()
                  .set_index(cat_cols)["correct"]
        )

    esc       = lambda s: s.replace("_", r"\_").replace("%", r"\%")
    indent    = r"\hspace{1em}"
    sub_label = lambda sub, n: fr"{indent}{esc(sub)} (n={n})"

    # ────────────────────────────── CATEGORY LIST
    data_cats = denom_df["question_category"].drop_duplicates().tolist()

    if question_order is None:
        cats = data_cats
    else:
        cats = question_order + [c for c in data_cats if c not in question_order]

    # ────────────────────────────── TABLE HEADER
    columns = ["\\textbf{Category}"] + [esc(str(k)) for k in group_keys]
    align   = r">{\raggedright\arraybackslash}p{4cm} " + " ".join("r" * len(group_keys))

    lines = [
        r"\begin{table}[H]",
        r"\centering",
    ]
    if caption:
        lines.append(rf"\caption{{{esc(caption)}}}")
    if label:
        lines.append(rf"\label{{{esc(label)}}}")
    lines += [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]

    # ────────────────────────────── BODY
    for cat in cats:
        denom_cat = denom_df[denom_df["question_category"] == cat]
        if denom_cat.empty:        # category genuinely filtered out
            continue

        # ---- decide sub-category sequence
        data_subs = denom_cat["sub_category"].unique().tolist()
        if sub_order and cat in sub_order:
            ordered = sub_order[cat] + [s for s in data_subs if s not in sub_order[cat]]
            subs = ordered
        else:
            subs = sorted(data_subs)

        first = True
        for sub in subs:
            row = denom_cat[denom_cat["sub_category"] == sub]
            if row.empty:          # sub listed in sub_order but not in data
                continue
            total = int(row["total"].iloc[0])

            stub = (
                esc(cat) + r"\\" + sub_label(sub, total)   # first sub-row
                if first else
                sub_label(sub, total)                      # later sub-rows
            )
            first = False

            cells = []
            for key in group_keys:
                correct = int(stats[key].get((cat, sub), 0))
                pct = 0.0 if total == 0 else correct / total * 100
                cells.append(f"{pct:5.1f}\\")

            lines.append(stub + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    # ────────────────────────────── TOTAL ROW
    total_cells = []
    for key in group_keys:
        tot   = total_total.get(key, 0)
        corr  = total_correct.get(key, 0)
        pct   = 0.0 if tot == 0 else corr / tot * 100
        total_cells.append(f"{pct:5.1f}\\")

    total_questions = subset_df["uuid"].nunique()
    lines.append(f"Total (n={total_questions})" + " & " + " & ".join(total_cells) + r" \\")

    # ────────────────────────────── FOOTER
    lines.append(r"\bottomrule")
    lines += [
        r"\end{tabular}",
        r"\end{table}",
    ]

    print("\n".join(lines))



def panel_tokens_table_latex(
        subset_df: pd.DataFrame,
        group_keys,
        group_to_df,
        *,
        value_col: str = "reasoning_tokens",              # ← NEW (default column)
        title: str | None = None,
        question_order: list[str] | None = None,
        sub_order: dict[str, list[str]] | None = None,
        label: str | None = None,
        caption: str | None = None,
    ) -> None:
    r"""
    Render a LaTeX panel table whose cells show the **average number of tokens**
    (``value_col``) for each (category, sub-category) × *group_key*.

    • Same visual layout as ``panel_table_latex`` (stub column, indented subs,
      final *Total* row).
    • ``value_col`` selects the numeric column to average (default:
      ``"reasoning_tokens"``).
    """
    # ───────────────────────────────────────── PREP
    cat_cols = ["question_category", "sub_category"]

    # counts per (category, sub-category) — used only for “n=…”
    denom_df = (
        subset_df.groupby(cat_cols)
                 .agg(total=("uuid", "nunique"))
                 .reset_index()
    )

    # per-group mean token counts per (category, sub)
    stats = {}
    # overall means for the TOTAL row
    total_mean = {}

    for key in group_keys:
        grp_df = group_to_df(key).copy()

        # store mean for TOTAL row
        total_mean[key] = grp_df[value_col].mean()

        stats[key] = (
            grp_df.groupby(cat_cols)[value_col]
                  .mean()                               # ← average tokens
        )  # Series indexed by (category, sub)

    esc       = lambda s: s.replace("_", r"\_").replace("%", r"\%")
    indent    = r"\hspace{1em}"
    sub_label = lambda sub, n: fr"{indent}{esc(sub)} (n={n})"

    # ────────────────────────────── CATEGORY LIST
    data_cats = denom_df["question_category"].drop_duplicates().tolist()
    cats = (question_order or []) + [c for c in data_cats
                                     if c not in (question_order or [])]

    # ────────────────────────────── TABLE HEADER
    columns = ["\\textbf{Category}"] + [esc(str(k)) for k in group_keys]
    align   = (r">{\raggedright\arraybackslash}p{4cm} "
               + " ".join("r" * len(group_keys)))

    lines = [
        r"\begin{table}[H]",
        r"\centering",
    ]
    if caption:
        lines.append(rf"\caption{{{esc(caption)}}}")
    if label:
        lines.append(rf"\label{{{esc(label)}}}")
    lines += [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]

    # ────────────────────────────── BODY
    for cat in cats:
        denom_cat = denom_df[denom_df["question_category"] == cat]
        if denom_cat.empty:
            continue

        # ---- decide sub-category sequence
        data_subs = denom_cat["sub_category"].unique().tolist()
        if sub_order and cat in sub_order:
            subs = sub_order[cat] + [s for s in data_subs if s not in sub_order[cat]]
        else:
            subs = sorted(data_subs)

        first = True
        for sub in subs:
            row = denom_cat[denom_cat["sub_category"] == sub]
            if row.empty:
                continue
            total = int(row["total"].iloc[0])

            stub = (
                esc(cat) + r"\\" + sub_label(sub, total) if first
                else sub_label(sub, total)
            )
            first = False

            cells = []
            for key in group_keys:
                mean_tokens = stats[key].get((cat, sub), float("nan"))
                cells.append(f"{mean_tokens:5.1f}")

            lines.append(stub + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    # ────────────────────────────── TOTAL ROW
    total_cells = [f"{total_mean.get(key, float('nan')):5.1f}" for key in group_keys]
    total_questions = subset_df["uuid"].nunique()
    lines.append(f"Total (n={total_questions})" + " & "
                 + " & ".join(total_cells) + r" \\")

    # ────────────────────────────── FOOTER
    lines.append(r"\bottomrule")
    lines += [
        r"\end{tabular}",
        r"\end{table}",
    ]

    print("\n".join(lines))
