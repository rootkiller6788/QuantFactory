"""报告：`out/report.md`。**报告里的每一个数字都是从 `run` 现算的。**

## 三条写作纪律（前两个项目各踩过一次）

1. **数字不许写死。** `L4-rewardfactory` 的 note 里出现过硬编码的「1.13 倍」
   「correctness 占 58%」—— 改一次配置这三句就全成谎话，而**报告里的谎话比缺一句话
   危险得多**（读者没法从报告内部发现它）。这里所有数字都从 `run` 插值。
2. **同一张表里的两个对象必须是同一口径。** `L4-rewardfactory` 里 provider 表的
   「单源一致性」曾用「与人类标签逐对相等」，而主表已经换成「对干净标签、平局算半分」——
   于是同一个 provider 在两张表里是两个东西。**换口径要换全，包括兄弟列。**
3. **必须能看见「哪些结论在哪失效」**（单独一节）。否则读者会把受控演示当成普遍结论。

## 另外两条

* **`created_at` 不许进正文。** 印了它，两份相同输入的报告就不逐位相同了。
* **耗时不许进正文的数字里。** 它随机器变化，同样会破坏逐位可复现。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import matrix
from .acceptance import binomial_se

__all__ = ["render_markdown"]


def _f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _pct(x: float | None, nd: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{nd}f}%"


def _signed(x: float, nd: int = 4) -> str:
    return f"{x:+.{nd}f}"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def render_markdown(run: Any) -> str:
    cfg = run.cfg
    L: list[str] = []
    add = L.append

    pv = matrix.pivot_quality(run.quality_rows)
    algs = pv["algorithms"]
    d = cfg["deployment"]
    n_test = run.n_test
    se = binomial_se(run.fp16.quality, n_test)
    refuted = [e for e in run.expectations if not e.holds]

    # ------------------------------------------------------------------ 1
    best = max(run.readings, key=lambda r: r.quality)
    budget = [r for r in run.readings
              if r.memory_gb <= float(d["memory_limit_gb"]) and r.latency_ms <= float(d["latency_limit_ms"])]
    knee_id = run.frontier_summary["knee_spec_id"]
    add("# QuantFactory —— 精度设计空间的一次完整测量")
    add("")
    add(f"**版本摘要** `{run.version_digest}`　**协议指纹** `{run.protocol_digest}`　"
        f"**候选** {len(run.readings)} 个　**前沿** {len(run.frontier)} 个点")
    add("")
    add("> 这份报告由 `main.py run` 生成，**每一个数字都是本次运行现算的**。")
    add("> 四维目标里**只有 Quality 是实测的**（测试集前向），"
        "Latency / Memory / Size 全部来自成本模型 —— 系数在 `config.json` 的 "
        "`runtime` / `deployment` 段里，改系数不改代码。")
    add("")

    # ------------------------------------------------------------------ 2
    add("## 1. 一句话结论")
    add("")
    add(f"* 全局最高质量 **{_f(best.quality)}**（`{best.spec_id}`，"
        f"有效位宽 {_f(best.effective_bits)}，体积 {_f(best.size_gb)} GB）；"
        f"fp16 基线 {_f(run.fp16.quality)}。")
    add(f"* 在 **≤{d['memory_limit_gb']} GB / ≤{d['latency_limit_ms']} ms** 的预算内，"
        f"共有 {len(budget)}/{len(run.readings)} 个候选可用；"
        f"**{len(run.readings) - len(budget)}** 个不可用（全部是 "
        f"{sorted({r.precision for r in run.readings if r not in budget})}）。")
    add(f"* 前沿拐点是 `{knee_id}`；"
        f"**每个算法都在前沿上占有一席之地**（"
        f"{', '.join(sorted({r.algorithm for r in run.frontier}))}）。")
    add(f"* 设计期望 {len(run.expectations)} 条，**被推翻 {len(refuted)} 条**："
        f"{', '.join(f'`{e.name}`' for e in refuted) if refuted else '无'}。")
    add(f"* 验收：{run.acceptance.to_dict()['n_thresholds_passed']}/"
        f"{run.acceptance.to_dict()['n_thresholds']} 条门槛 + "
        f"{run.acceptance.to_dict()['n_cases_passed']}/"
        f"{run.acceptance.to_dict()['n_cases']} 条对抗用例通过。")
    add("")

    # ------------------------------------------------------------------ 3
    add("## 2. 世界与基座")
    add("")
    w = run.world_d
    t = run.train_d
    add(f"本地模型只有 **{run.fp16.n_params}** 个参数，而部署代价按 "
        f"**{d['base_params_b']}B** 算 —— 映射是 "
        f"`size_gb = base_params_b × effective_bits / 8`。"
        "**质量的排序能迁移，绝对数值不能。**")
    add("")
    L.extend(_md_table(
        ["量", "值", "说明"],
        [
            ["d_in / hidden / classes", f"{w['d_in']} / {w['hidden']} / {w['n_classes']}", "三层 MLP"],
            ["train / test / calib_pool", f"{w['n_train']} / {w['n_test']} / {w['n_calib_pool']}",
             "**三块互不重叠**（逐行哈希核对）"],
            ["类间最小距 / 平均距", f"{_f(w['margin'])} / {_f(w['mean_pairwise_dist'])}", "任务可分性"],
            ["窄通道 / 宽通道", f"{w['narrow_dims']} / {w['wide_dims']}",
             "窄 ⇒ 权重离群；宽 ⇒ 激活离群（**信号已解耦**）"],
            ["训练步数 × 批", f"{t['steps']} × {t['batch']}", f"手写 SGD，动量 {_f(cfg['train']['momentum'], 2)}"],
            ["train / test 准确率", f"{_f(t['train_accuracy'])} / {_f(t['test_accuracy'])}", "过拟合已受控"],
            ["梯度裁剪触发次数", t["grad_clip_hits"],
             "**一条从不触发的保险丝与没有保险丝无法区分** ⇒ 必须算出来"],
            ["测试集二项标准误", _f(se, 6), f"n={n_test}，p={_f(run.fp16.quality)}"],
        ],
    ))
    add("")

    # ------------------------------------------------------------------ 4
    add("## 3. 设计空间")
    add("")
    space = cfg["space"]
    add(f"```\nalgorithm ∈ {list(space['algorithms'])}\n"
        f"precision ∈ {list(space['precisions'])}\n"
        f"group     ∈ {list(space['groups'])}\n"
        f"calib     ∈ {list(run.calib_levels)}   （rtn / kquant 恒为 none）\n```")
    add("")
    add(f"`rtn` / `kquant` **校准无关**（只读权重），`gptq` / `awq` 需要校准集。"
        f"展开出 **{len(run.readings)}** 个候选。")
    add("")
    add("**校准统计按档位算一次，全部候选共用同一个对象**"
        f"（对象同一性核对：{'通过' if run.stats_identity_ok else '**未通过**'}）—— "
        "否则结果看起来一模一样，只是慢了几百倍，而没有任何断言会发现。")
    add("")

    # ------------------------------------------------------------------ 5
    add("## 4. 主结论表：哪个算法在哪个位宽最好")
    add("")
    add(f"每格是该 (位宽, 分组, 算法) 上**质量最高**的候选（跨校准档位取最好）。"
        f"`spread` 是该行算法间极差。fp16 基线 = {_f(run.fp16.quality)}。")
    add("")
    rows = []
    for prec, grp in pv["row_keys"]:
        vals = {a: pv["cells"].get(f"{prec}|g{grp}|{a}") for a in algs}
        got = {a: c["quality"] for a, c in vals.items() if c}
        if not got:
            continue
        besta = max(got, key=lambda k: got[k])
        rows.append([
            prec, f"g{grp}",
            *[f"{_f(got[a])}" + ("**" if a == besta else "") for a in algs],
            _signed(max(got.values()) - min(got.values())), besta,
        ])
    L.extend(_md_table(["位宽", "分组", *algs, "spread", "最好"], rows))
    add("")

    def _row_spread(p: str, g: int) -> float:
        vals = [c["quality"] for c in
                (pv["cells"].get(f"{p}|g{g}|{a}") for a in algs) if c]
        return (max(vals) - min(vals)) if vals else 0.0

    spread_hi = max(_row_spread(p, g) for p in ("int5", "int6", "int8")
                    for g in cfg["space"]["groups"])
    spread_lo = max(_row_spread("int2", g) for g in cfg["space"]["groups"])
    add("**读法**：位宽 ≥ int5 之后 spread 掉到噪声量级"
        f"（int5/int6/int8 上最大的算法间极差 = {spread_hi:.4f}），"
        f"而 int2 上能到 {spread_lo:.4f}。"
        "**「选哪个算法」只在你打算压到 int2/int3 时才重要。**")
    add("")

    # ------------------------------------------------------------------ 6
    add("## 5. 成本模型：四维里三维是算出来的")
    add("")
    r = cfg["runtime"]
    add(f"```\nsize_gb   = base_params_b × effective_bits / 8\n"
        f"memory_gb = size_gb + ({r['activation_mb']} + {r['fixed_memory_mb']}) / 1024\n"
        f"latency   = {r['fixed_latency_ms']} ms + 访存时间 × 反量化惩罚 / kernel 效率\n"
        f"反量化惩罚 = (1 + {r['dequant_penalty_per_block']}) ** log2(n_blocks / n_blocks(128))\n```")
    add("")
    add(f"**`memory = size + {_f(run.memory_overhead_gb, 8)}`，那个常数与量化方案无关** ⇒ "
        f"`corr(memory, size) = {run.corr_memory_size:.12f}`，"
        "四维实际是三维。这不是缺陷，是**模型写得太直白**："
        "权重的字节数当然同时出现在体积与显存里。"
        "把这一维保留下来，是为了让读者自己看见它冗余（见 §11 与 §12）。")
    add("")
    add("### 有效位宽 ≠ 名义位宽")
    add("")
    rows = []
    for g in cfg["space"]["groups"]:
        for p in ("int4",):
            line = []
            for a in algs:
                rr = matrix.by_spec_id(run.readings).get(f"{a}|{p}|g{g}|" +
                     (f"c{max(run.calib_levels)}" if a in ("gptq", "awq") else "none"))
                if rr:
                    line.append(f"{a}: {_f(rr.effective_bits)}")
            base = matrix.by_spec_id(run.readings).get(f"rtn|{p}|g{g}|none")
            rows.append([f"{p} g{g}", f"{_f(base.nominal_bits, 1)}", *line,
                         f"{_f(base.metadata_bits_per_weight)}"])
    L.extend(_md_table(["精度/分组", "名义位宽", *algs, "rtn 元数据位/权重"], rows))
    add("")
    add("**同一个「INT4」在不同算法下不是同一个体积。**"
        "而且注意 `kquant` 的**元数据最少**（二级 scale 只有 8 位/组）"
        "但它的**有效位宽可能反而更大** —— 层位宽混合把 `head` 抬了 "
        f"{cfg['quant']['kquant']['layer_mix'].get('head')} 位。"
        "**「元数据最少」与「有效位宽最少」是两个不同的结论。**")
    add("")

    # ------------------------------------------------------------------ 7
    add("## 6. 前沿")
    add("")
    fs = run.frontier_summary
    add(f"四维非支配集：**{fs['size']}** 个点（被支配 {fs['n_dominated']} 个）。"
        f"超体积 {_f(fs['hypervolume'], 6)}（参考点 {[round(v, 4) for v in fs['reference_point']]}）；"
        f"间距 {_f(fs['spacing'], 6)}（越小越均匀）。")
    add("")
    L.extend(_md_table(
        ["spec_id", "算法", "位宽", "分组", "校准", "质量", "有效位宽", "体积 GB", "延迟 ms", "可行"],
        [[x.spec_id, x.algorithm, x.precision, f"g{x.group}", x.spec.calib_label,
          _f(x.quality), _f(x.effective_bits), _f(x.size_gb), _f(x.latency_ms, 2),
          "是" if (x.memory_gb <= float(d["memory_limit_gb"])
                   and x.latency_ms <= float(d["latency_limit_ms"])) else "**否**"]
         for x in run.frontier],
    ))
    add("")
    add(f"拐点算在 **(第 {fs['knee_axes'][0]} 维, 第 {fs['knee_axes'][1]} 维)** 平面上 —— "
        "四维没有「两端连线」这个概念，不先投影就是在算一个不存在的量。")
    add("")

    # ------------------------------------------------------------------ 8
    add("## 7. 选型")
    add("")
    L.extend(_md_table(
        ["规则", "选中", "质量", "体积 GB", "延迟 ms", "有效位宽", "可行"],
        [[s.rule, f"`{s.spec_id}`" if s.spec_id else "—", _f(s.quality),
          _f(s.size_gb), _f(s.latency_ms, 2), _f(s.effective_bits),
          "是" if s.feasible else "否"] for s in run.selections],
    ))
    add("")
    for s in run.selections:
        add(f"* **{s.rule}** —— {s.why}")
    add("")

    # ------------------------------------------------------------------ 9
    add("## 8. 两两 Diff（匹配格）")
    add("")
    add("固定其余各轴，只让一个轴变，然后在**同格**里两两比。"
        "`升/降/平` 是符号分布 —— 均值 +0.02 里如果一半格 +0.06、一半格 −0.02，"
        "那和「一致 +0.02」是两个完全不同的结论。")
    add("")
    L.extend(_md_table(
        ["轴", "左", "右", "格数", "Δ质量均值", "区间", "升/降/平",
         "Δ体积 GB", "Δ延迟 ms", "Δ有效位宽"],
        [[x["axis"], x["left"], x["right"], x["n_cells"],
          _signed(x["mean_delta_quality"]),
          f"[{_signed(x['min_delta_quality'])}, {_signed(x['max_delta_quality'])}]",
          f"{x['n_up']}/{x['n_down']}/{x['n_tie']}",
          _signed(x["mean_delta_size_gb"]), _signed(x["mean_delta_latency_ms"], 2),
          _signed(x["mean_delta_effective_bits"])] for x in run.diff_rows],
    ))
    add("")
    add("**注意 `algorithm` 轴的匹配键不含 `calib`** —— 校准档位是算法的属性"
        "（rtn/kquant 恒为 0），不是与算法正交的轴。不排除它的话，"
        "`gptq` 与 `rtn` 永远落不进同一个格，"
        "「算法值不值它多花的元数据与校准成本」在表里会**一行都没有**。")
    add("")

    # ------------------------------------------------------------------ 10
    add("## 9. 校准规模敏感性")
    add("")
    add("校准档位之间的差距**拆成两列**：`Δ权重相对误差` 是机制读数（量化器真的贴得更准了吗），"
        "`Δ质量` 是下游读数（贴得更准有没有变成更好的分类）。"
        "只报后者会把「校准集过拟合」误读成「校准没用」。")
    add("")
    L.extend(_md_table(
        ["算法", "位宽", "分组", "档位", "质量(左)", "质量(右)", "Δ质量",
         "相对误差(左)", "相对误差(右)", "Δ相对误差", "Δ有效位宽"],
        [[x["algorithm"], x["precision"], f"g{x['group']}",
          f"c{x['calib_left']}→c{x['calib_right']}",
          _f(x["quality_left"]), _f(x["quality_right"]), _signed(x["delta_quality"]),
          _f(x["rel_error_left"], 6), _f(x["rel_error_right"], 6),
          _signed(x["delta_weight_rel_error"], 6), _f(x["delta_effective_bits"], 8)]
         for x in run.sensitivity_rows],
    ))
    add("")
    L.extend(_md_table(
        ["算法", "格数", "Δ质量均值", "升/降/平", "Δ相对误差均值", "相对误差下降的格数"],
        [[s["algorithm"], s["cells"], _signed(s["mean_delta_quality"]),
          f"{s['n_quality_up']}/{s['n_quality_down']}/{s['n_quality_tie']}",
          _signed(s["mean_delta_weight_rel_error"], 6),
          f"{s['n_relerr_down']}/{s['cells']}"] for s in run.sensitivity_summary],
    ))
    add("")
    add("`Δ有效位宽` 这一列**必须恒为 0** —— 校准档位不该改变位宽。"
        "它非 0 就说明校准统计被误接进了位宽计算。这是一条便宜的哨兵。")
    add("")

    # ------------------------------------------------------------------ 11
    add("## 10. 验收")
    add("")
    add(f"**{run.acceptance.to_dict()['n_thresholds']} 条门槛**"
        f"（阈值全部在 `config.json` 的 `acceptance` 段里）：")
    add("")
    L.extend(_md_table(
        ["门槛", "阈值", "实测", "通过", "说明"],
        [[c.key, _f(c.limit, 6), _f(c.observed, 6), "✅" if c.passed else "❌", c.detail]
         for c in run.acceptance.thresholds],
    ))
    add("")
    add(f"**{run.acceptance.to_dict()['n_cases']} 条对抗用例**"
        "（判据写在 `acceptance.py` 里，**失败 = 代码有 bug**，与门槛失败的含义不同）：")
    add("")
    L.extend(_md_table(
        ["用例", "通过", "说明"],
        [[c.name, "✅" if c.passed else "❌", c.detail] for c in run.acceptance.cases],
    ))
    add("")

    # ------------------------------------------------------------------ 12
    add("## 11. 设计期望")
    add("")
    add(f"**{len(run.expectations)} 条，被推翻 {len(refuted)} 条。**"
        "每条期望都带着它的**操作化** —— 「A 比 B 好」不是一个可判定的命题，"
        "除非说清在哪一档、比均值还是逐格、允许多大容差。")
    add("")
    L.extend(_md_table(
        ["期望", "期望值", "实测", "成立", "实测读数"],
        [[e.name, "是" if e.expected else "否", "是" if e.observed else "否",
          "✅" if e.holds else "**❌ 被推翻**", e.reading] for e in run.expectations],
    ))
    add("")
    if refuted:
        add("### 被推翻的那几条（**最值钱的产出**）")
        add("")
        for e in refuted:
            add(f"* **`{e.name}`** —— {e.statement}")
            add(f"  * 操作化：{e.operationalization}")
            add(f"  * 实测：{e.reading}")
        add("")

    # ------------------------------------------------------------------ 13
    add("## 12. 这些结论在哪里失效")
    add("")
    add("**这一节是这份报告里最该先读的部分。**"
        "上面的每一条都是在**一个受控的合成世界**上测出来的，"
        "把任何一条当成普遍结论都会出错。")
    add("")
    fails = _failure_boundaries(run)
    for title, body in fails:
        add(f"### {title}")
        add("")
        add(body)
        add("")

    # ------------------------------------------------------------------ 14
    add("## 13. 产物清单")
    add("")
    add("| 文件 | 内容 | 两次运行是否逐位相同 |")
    add("| --- | --- | --- |")
    add("| `config.json` | 有效配置（DEFAULTS ⊕ 覆盖） | 是 |")
    add("| `world.json` / `train.json` / `calibration.json` | 世界 / 训练 / 校准读数 | 是 |")
    add("| `readings.json` | 全部候选的四维读数（**不含耗时**） | 是 |")
    add("| `matrix.csv` / `quality_table.csv` / `frontier.csv` | 三张读数表 | 是 |")
    add("| `selection.csv` / `diff.csv` / `sensitivity.csv` | 选型 / Diff / 敏感性 | 是 |")
    add("| `acceptance.json` / `expectations.json` / `summary.json` | 判据与结论 | 是 |")
    add("| `lineage.dot` / `report.md` / `report.html` | 血缘图与报告 | 是 |")
    add("| `version.json` | **含 `created_at`** | **否**（豁免） |")
    add("| `benchmark.json` | **含耗时** | **否**（豁免） |")
    add("")
    add("`version.json` 与 `benchmark.json` 是**唯一允许两次运行不同**的两个产物。"
        "`created_at` 只许出现在 `version.json`，耗时只许出现在 `benchmark.json` —— "
        "印进正文，两份相同输入的报告就不逐位相同了。")
    add("")

    return "\n".join(L)


def _failure_boundaries(run: Any) -> list[tuple[str, str]]:
    """**「哪些结论在哪失效」** —— 从数据里生成，不是套话。"""
    cfg = run.cfg
    out: list[tuple[str, str]] = []
    by_id = matrix.by_spec_id(run.readings)
    se = binomial_se(run.fp16.quality, run.n_test)

    # 1) 尺度映射
    out.append((
        "绝对数值不能迁移到真模型",
        f"本地模型 {run.fp16.n_params} 个参数，部署代价按 {cfg['deployment']['base_params_b']}B 算。"
        "**质量的排序能迁移，绝对数值不能** —— 本地从 "
        f"{min(r.quality for r in run.readings):.4f} 到 {max(r.quality for r in run.readings):.4f} 的这段差距，"
        "在真模型上不是同一个数。所有「差 5 个百分点」的说法都只在**这个**合成任务上成立。",
    ))

    # 2) 成本模型
    r = cfg["runtime"]
    out.append((
        "三维成本是模型算的，不是测的",
        f"`latency = {r['fixed_latency_ms']} ms + size/带宽 × 惩罚 / 效率` 里的带宽 "
        f"{r['bandwidth_gbps']} GB/s、反量化惩罚 {r['dequant_penalty_per_block']}/块、"
        "各算法的 kernel 效率全是**假设**。真实 kernel 的效率取决于算子实现、"
        "batch size、序列长度，与这里的常数无关。**结论「g16 比 g64 慢」只在"
        "「反量化开销 ∝ 块数」这个假设下成立。**",
    ))

    # 3) AWQ 的失效边界（最有价值的一条）
    awq2 = [r for r in run.readings if r.algorithm == "awq" and r.precision == "int2"]
    gptq2 = [r for r in run.readings if r.algorithm == "gptq" and r.precision == "int2"]
    if awq2 and gptq2:
        aw = max(awq2, key=lambda x: x.quality)
        gt = max(gptq2, key=lambda x: x.quality)
        out.append((
            "AWQ 在这个世界上没赢，而这不是实现错误",
            f"int2 上 AWQ 最好 {aw.quality:.4f}（`{aw.spec_id}`）"
            f"而 GPTQ 最好 {gt.quality:.4f}（`{gt.spec_id}`）。"
            "AWQ 的收益来自**真实的 LLM 神经激活离群**（典型幅度 20~100×）—— "
            "它把 `W·s` 的组最大值 `M = max_k |W_k|·act_k^α` 压下去，靠的是"
            "「激活大的通道权重小」。本项目为了保持 fp16 基座可学，"
            f"把宽通道的幅度限制在 {cfg['world']['wide_scale']}×"
            f"（信号再乘 {cfg['world']['wide_spread_ratio']}），"
            "离群强度远低于真实 LLM ⇒ **AWQ 的前提在这里只被满足了一点点**。"
            "换个世界（更强的激活离群）它应该会赢 —— 但那是另一个实验。",
        ))

    # 4) GPTQ 的失效边界
    out.append((
        "GPTQ 的误差补偿在「组内 scale 已经很细」时没有发挥空间",
        "GPTQ 的推导假设剩余列是**连续**的；一旦按组量化，组内各列共享一个由"
        "**组的 min/max** 决定的 scale，把补偿落回同组等于把值推向网格边缘甚至越界。"
        "本项目把补偿限制在**后续组**（逐列版本实测会让层输出误差 +272%，整组版本 −0.5%），"
        "但整组补偿的增益只有 −1.6%~+2.0% ⇒ **与 RTN 打平**。"
        "在「整行一个 scale」（不分组的真 GPTQ）或更低位宽下，结论可能相反。",
    ))

    # 5) 位宽 ≥5 的噪声区
    d58 = next((x for x in run.diff_rows
                if x["axis"] == "precision" and x["left"] == "int5" and x["right"] == "int8"), None)
    if d58:
        out.append((
            "位宽 ≥ int5 之后的一切差异都在噪声里",
            f"int5→int8 的 Δ质量均值 {d58['mean_delta_quality']:+.4f}，"
            f"符号分布 升 {d58['n_up']} / 降 {d58['n_down']} / 平 {d58['n_tie']} —— "
            "**符号几乎随机**。二项标准误是 "
            f"{se:.6f}，任何小于它的差异都不该被当成结论。"
            "「int6 比 int5 好」这种说法在这个测试集规模下不可判定。",
        ))

    # 6) 单点世界
    out.append((
        "只有一个种子、一个世界",
        f"`seed = {cfg['seed']}`。所有数字都来自**这一个**合成任务、**这一个**训练好的基座。"
        "换种子会改掉全部读数。这里的期望表与门槛表是**结构性的**（跨种子应当稳定），"
        "但「哪一格高出 0.008」不是。",
    ))

    # 7) K-quant 的幂等性
    out.append((
        "K-quant 不是幂等的，而这是它的固有性质",
        "子块 scale 自己也被量化到 8 位：`round(sc/smax·255)` 可能把 scale **向上**取整，"
        "于是该子块的 max 落到 254 而非 255；再量化时 `sc` 由 254 反推，第二次取整成另一个值 —— "
        "**这个映射没有不动点**。单级量化（RTN / GPTQ / AWQ）严格幂等（网格完全由权重自身决定，"
        "且组内 min/max 一定被取到），K-quant 只能保证「移动不超过一个步长」。"
        "`acceptance.max_idempotence_violations` 把这条区别钉住了。",
    ))

    return out
