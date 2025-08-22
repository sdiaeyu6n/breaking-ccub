# ui_style.py
import matplotlib.pyplot as plt

FIG_SIZE = (6, 3)
HEATMAP_SIZE = (4, 3)
TITLE_SIZE = 7
LABEL_SIZE = 6
TICK_SIZE = 6
PLOTLY_HEIGHT = 360

def apply_matplotlib_rc():
    plt.rcParams.update({
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,       # ⬅ 기본 레전드 폰트
    })

def style_axis(ax, title=None, xlabel="", ylabel="", rotate_x=0, rotate_y=0):
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    if rotate_x:
        for label in ax.get_xticklabels():
            label.set_rotation(rotate_x)
    if rotate_y:
        for label in ax.get_yticklabels():
            label.set_rotation(rotate_y)

def style_legend(ax, loc="best"):
    """축 스타일과 맞춘 레전드 사이즈/위치"""
    leg = ax.get_legend()
    if leg:
        leg.set_title(leg.get_title().get_text(), prop={"size": LABEL_SIZE})
        for t in leg.get_texts():
            t.set_fontsize(TICK_SIZE)
        leg.set_bbox_to_anchor(None)  # 기본 박스
        leg.set_loc(loc)
