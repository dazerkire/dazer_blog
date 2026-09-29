#!/usr/bin/env python3
"""Regenerate the series' numerical SVGs (optional dependency: matplotlib).

All curves are illustrative cost models, not measured performance. Run with a
CJK font, e.g. --font /path/to/NotoSansCJK-Regular.ttc. Jekyll uses the committed
SVGs and does not run this script. --preview-dir writes optional PNG previews.
"""
from pathlib import Path
import argparse
import math
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/images/posts/llm-inference'
BW, BF16_PEAK, FP8_PEAK = 4.8e12, 989e12, 1979e12
KV_BYTES = 131072


def prefill_flops(length):
    """32 Llama 8B blocks, causal useful attention FLOPs, no full-prompt LM head."""
    return 32 * (5 * length * 4096**2 + 6 * length * 4096 * 14336
                 + 2 * length * (length + 1) * 4096)


def workload_ms(length, output, weights, peak):
    prefill = prefill_flops(length) / peak * 1000
    decode = sum((weights + (length + j) * KV_BYTES) / BW * 1000
                 for j in range(1, output))
    return prefill, decode


def expected_tokens(alpha, candidates):
    return sum(alpha**i for i in range(candidates + 1))


def serving_ms(batch, rows=1):
    # Rounded GB and GFLOP coefficients deliberately match article 7's equations.
    return max((15 + .268 * batch) / 4.8, rows * 16.1 * batch / 989)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, help='A local font supporting Chinese')
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import ScalarFormatter
    if args.font:
        font_manager.fontManager.addfont(str(args.font))
        font = font_manager.FontProperties(fname=str(args.font)).get_name()
    else:
        candidates = ('Noto Sans CJK SC', 'Noto Sans SC', 'Microsoft YaHei', 'PingFang SC')
        names = {f.name for f in font_manager.fontManager.ttflist}
        font = next((f for f in candidates if f in names), None)
        if font is None:
            parser.error('No Chinese font found; provide --font /path/to/font.ttf')
    plt.rcParams.update({'font.family': font, 'font.size': 12, 'axes.unicode_minus': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'llm-inference-v2',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#344054', 'axes.labelcolor': '#344054',
                         'xtick.color': '#64748b', 'ytick.color': '#64748b',
                         'axes.edgecolor': '#64748b', 'figure.facecolor': '#fbfcfe',
                         'axes.facecolor': '#fbfcfe', 'savefig.facecolor': '#fbfcfe'})
    blue, orange, purple, gray = '#2563eb', '#f97316', '#7c3aed', '#64748b'
    posts = '\n'.join(p.read_text() for p in sorted((ROOT / '_posts').glob('*.md')))
    alts = dict(re.findall(r'<img\s[^>]*src="[^"]*/([^/"]+\.svg)"[^>]*alt="([^"]+)"', posts, re.S))

    def base(title, height=6.2):
        fig, ax = plt.subplots(figsize=(11.2, height), dpi=100)
        fig.subplots_adjust(left=.10, right=.96, bottom=.24, top=.78)
        fig.text(.055, .93, title, fontsize=18)
        ax.grid(axis='y', color='#dce3ea', linestyle=(0, (4, 5)), zorder=0)
        ax.set_axisbelow(True)
        return fig, ax

    def note(fig, lines):
        for i, line in enumerate(lines):
            fig.text(.055, .12 - .044*i, line, fontsize=10.5, color=gray)

    def save(fig, name, title):
        path = OUT / name
        fig.savefig(path, format='svg', metadata={'Date': None, 'Creator': 'generate_inference_figures.py'})
        s = path.read_text()
        s = re.sub(r'width="[^"]+pt" height="[^"]+pt"',
                   f'width="1120" height="{round(fig.get_figheight()*100)}"', s, count=1)
        # Keep editable text and portable browser fallbacks; the local font is for QA only.
        s = re.sub(r'font-family: [^;]+;',
                   'font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;', s)
        # Quotes inside a style attribute must be XML-escaped.
        s = s.replace('"PingFang SC", "Microsoft YaHei"', "'PingFang SC', 'Microsoft YaHei'")
        s = s.replace('<svg ', '<svg role="img" aria-labelledby="title desc" ', 1)
        from xml.sax.saxutils import escape
        a = s.index('>', s.index('<svg ')) + 1
        s = s[:a] + f'\n<title id="title">{escape(title)}</title>\n<desc id="desc">{escape(alts[name])}</desc>' + s[a:]
        ET.fromstring(s)
        s = "\n".join(line.rstrip() for line in s.splitlines()) + "\n"
        path.write_text(s)
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / name.replace('.svg', '.png'), dpi=100)
        plt.close(fig)

    title = 'Prefill 的因果有效 FLOPs：Llama 3.1 8B，L = 2048'
    fig, ax = base(title, 4.7)
    fig.subplots_adjust(left=.28, right=.82, bottom=.29, top=.77)
    length = 2048
    values = [32*6*length*4096*14336/1e12,
              32*5*length*4096**2/1e12,
              32*2*length*(length+1)*4096/1e12]
    labels = ['SwiGLU MLP', 'QKV + 输出投影', '因果 attention']
    bars = ax.barh(labels, values, color=[blue, orange, purple], height=.56)
    ax.invert_yaxis(); ax.set_xlim(0, 25); ax.set_xlabel('TFLOPs（32 层合计）')
    for bar, value in zip(bars, values):
        ax.text(value+.35, bar.get_y()+bar.get_height()/2,
                f'{value:.2f} · {value/sum(values):.1%}', va='center', fontsize=11)
    note(fig, ['合计约 29.69 TFLOPs；attention 使用 2L(L+1)d，而非满矩阵 4L²d。',
               '不含 embedding、LM Head 与小算子；这里只统计工作量，不表示实际耗时。'])
    save(fig, 'prefill-flops-breakdown.svg', title)

    title = '输入与输出长度改变了量化收益：简化模型估算'
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.2), dpi=100)
    fig.subplots_adjust(left=.085, right=.97, bottom=.28, top=.74, wspace=.28)
    fig.text(.055, .93, title, fontsize=18)
    for ax, length, output, subtitle, ymax in zip(axes, [2048,32768],[128,200],
                                                 ['对话：2048 / 128','RAG：32768 / 200'],[500,1800]):
        ax.set_title(subtitle, fontsize=14, pad=14)
        vals=[workload_ms(length,output,w,p) for w,p in
              [(15e9,BF16_PEAK),(3.75e9,BF16_PEAK),(7.5e9,FP8_PEAK)]]
        for i, ((pre, dec), color) in enumerate(zip(vals,[blue,orange,purple])):
            ax.bar(i,pre,width=.55,color=color,alpha=.24,zorder=3)
            ax.bar(i,dec,bottom=pre,width=.55,color=color,zorder=3)
            ax.text(i,pre+dec+ymax*.025,f'{pre+dec:.0f} ms',ha='center',fontsize=12)
        ax.set_xticks(range(3),['BF16','W4A16','FP8 W8A8']); ax.set_ylim(0,ymax)
        ax.set_ylabel('ms'); ax.grid(axis='y',color='#dce3ea',linestyle='--',zorder=0)
        ax.set_axisbelow(True)
    note(fig,['浅色：因果 Prefill；深色：M-1 次 Decode，历史长度随输出增长。两面板纵轴不同。',
              '三种方案均保留 BF16 KV；忽略量化转换、调度、网络等成本，不能当作实测选型结论。'])
    save(fig,'quant-end-to-end-workloads.svg',title)

    title = '验证行数改变计算项：两个汇总下界的交点'
    fig,ax=base(title)
    bs=list(range(0,351))
    ax.plot(bs,[(15+.268*b)/4.8 for b in bs],color=blue,lw=2.7,label='主要读流量项')
    for rows,color in [(1,gray),(5,orange),(45,purple)]:
        ax.plot(bs,[rows*16.1*b/989 for b in bs],color=color,lw=2.2,label=f'计算项：{rows} 行/路')
    ax.set_xlim(0,350);ax.set_ylim(0,32);ax.set_xlabel('并发 B');ax.set_ylabel('时间下界（ms）')
    cross=(15/4.8)/(5*16.1/989-.268/4.8)
    ax.scatter([cross],[(15+.268*cross)/4.8],s=55,color=orange,zorder=5)
    ax.annotate('五行交点约 122',xy=(cross,(15+.268*cross)/4.8),xytext=(130,4.0),
                arrowprops={'arrowstyle':'->','color':gray},fontsize=11)
    ax.annotate('45 行交点约 4.6',xy=(4.6,3.38),xytext=(35,14),
                arrowprops={'arrowstyle':'->','color':gray},fontsize=11)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.22),ncol=2,frameon=False,fontsize=10.5)
    note(fig,['L=2048，H200，主要读量与计算量系数按正文近似取值，效率系数取 1。',
              '交点不是“免费行数”或盈亏平衡点；未包含完整执行开销与容量限制。'])
    save(fig,'spec-serving-iteration-cost.svg',title)

    title='不同结构假设下的吞吐：共同比较区间 B ≤ 350'
    fig,ax=base(title)
    bs=list(range(1,351)); e=expected_tokens(.8,4)
    baseline=[b/serving_ms(b) for b in bs]
    external=[b*e/(4*(2.5+.067*b)/4.8+serving_ms(b,5)) for b in bs]
    light=[b*e/(4*(1.50+.268*b/32)/4.8+serving_ms(b,5)) for b in bs]
    tree=[b*4.1/(4*(1.50+.268*b/32)/4.8+serving_ms(b,45)) for b in bs]
    for vals,color,label in [(baseline,blue,'普通 Decode'),(external,orange,'外部 1B + 四步链'),
                              (light,purple,'单层 draft + 四步链'),(tree,gray,'单层 draft + 45 行树')]:
        ax.plot(bs,vals,lw=2.7,color=color,label=label)
    ax.set_xlim(0,350);ax.set_ylim(0,40);ax.set_xlabel('并发 B');ax.set_ylabel('吞吐估算（k tok/s）')
    ax.axvline(64,color=gray,ls='--',lw=1);ax.text(68,37,'B=64',fontsize=10,color=gray)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.22),ncol=2,frameon=False,fontsize=10.5)
    note(fig,['链 E[N]=3.3616，树 E[N]=4.1；相同接受率仅是比较假设，曲线不是论文实测。',
              '外部 draft 在 120 GB KV 预算下最多 350 路（含 draft 权重与持久 KV；额外临时状态另计）。',
              '渐近吞吐仅用于分析主导项，不表示有限显存内可以达到。'])
    save(fig,'spec-serving-throughput.svg',title)

    title='固定步时下的投机收益：接受概率与候选长度'
    fig,ax=base(title)
    alphas=[i/200 for i in range(201)]
    for k,color in [(2,blue),(4,orange),(8,purple)]:
        ax.plot(alphas,[3.2*expected_tokens(a,k)/(3.2+k*.535) for a in alphas],
                label=f'k={k}',color=color,lw=2.8)
    ax.axhline(1,color=gray,ls='--',lw=1);ax.axvline(.8,color=gray,ls=':',lw=1)
    ax.set_xlim(0,1);ax.set_ylim(0,4.1);ax.set_xlabel('各位置的条件接受概率 α（假设相同）')
    ax.set_ylabel('模型内加速比');ax.legend(frameon=False,ncol=3,loc='upper left')
    note(fig,['T_base = T_verify = 3.2 ms；T_draft = 0.535 ms/步；E[N] = Σ α^i（i=0…k）。',
              '未包含其他开销；真实接受率随位置变化时，应改用前缀存活概率与实测轮次时间。'])
    save(fig,'spec-speedup-vs-alpha.svg',title)

    title='每轮成本的结构算例：不代表算法实测排名'
    fig,ax=base(title,6.4)
    fig.subplots_adjust(left=.21,right=.77,bottom=.24,top=.80)
    labels=['等效保留 16 层','等效保留 8 层','外部 1B','单层 draft + LM Head','Medusa 两头示例','文本查找理想示例']
    draft=[6.8,3.84,2.14,1.24,0,0];verify=[3.2]*6;extra=[0,0,0,0,.44,0]
    ax.barh(labels,draft,color=orange,height=.56,label='起草总时间')
    ax.barh(labels,verify,left=draft,color=purple,height=.56,label='验证')
    ax.barh(labels,extra,left=[d+v for d,v in zip(draft,verify)],color=blue,height=.56,label='附加头')
    for i,(d,v,x) in enumerate(zip(draft,verify,extra)):
        total=d+v+x; ax.text(total+.12,i,f'{total:.2f} ms   E*={total/3.2:.2f}',va='center',fontsize=10.5)
    ax.invert_yaxis();ax.set_xlim(0,10.5);ax.set_xlabel('一轮时间（ms）')
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.15),ncol=3,frameon=False,fontsize=10.5)
    note(fig,['串行起草取 k=4；验证统一假设 3.2 ms；Medusa 的附加头计入一轮，再按产出摊销。',
              '跳层按等效完整 block 近似；文本查找忽略查询时间。E* = 一轮时间 / 3.2 ms。'])
    save(fig,'spec-round-cost-breakdown.svg',title)

    title='主要读流量模型中的量化收益'
    fig,ax=base(title)
    bs=[10**(i/100*math.log10(512)) for i in range(101)]
    for w,kv,color,label in [(3.75,.268,blue,'W4A16，仅权重'),(3.75,.134,purple,'W4A16 + FP8 KV'),
                             (15,.134,orange,'仅 FP8 KV')]:
        ax.plot(bs,[(15+.268*b)/(w+kv*b) for b in bs],color=color,lw=2.7,label=label)
    ax.axhline(2,color=gray,ls='--',lw=1);ax.set_xscale('log');ax.set_xticks([1,4,16,64,256,512])
    ax.xaxis.set_major_formatter(ScalarFormatter());ax.set_xlim(1,512);ax.set_ylim(.9,4.1)
    ax.set_xlabel('并发 B（对数轴，仅作流量模型）');ax.set_ylabel('相对 BF16 的读流量比')
    ax.legend(frameon=False,loc='upper right',fontsize=11)
    note(fig,['L=2048；S = 1 / (1-f+f/q)，f 为压缩前的占比，q 为压缩倍数。',
              '忽略 scale、反量化与其他流量；读流量比不等于实测加速比，容量需另行约束。'])
    save(fig,'quant-speedup-vs-batch.svg',title)

    # Print a small reproducible numeric ledger for editorial review.
    print({'prefill_TFLOPs':{L:prefill_flops(L)/1e12 for L in [2048,32768]},
           'target_capacity':int(120e9/(2048*KV_BYTES)),
           'external_draft_capacity':int(117.5e9/(2048*(KV_BYTES+32768))),
           'verification_crossing_rounded':cross,
           'serving_speedup_B64':external[63]/baseline[63]})


if __name__ == '__main__':
    main()
