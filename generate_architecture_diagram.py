from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent / "rk3588_architecture_diagram.png"
WIDTH = 1800
HEIGHT = 1080

BG = "#f7f9fc"
NAVY = "#143a6f"
BLUE = "#2f6db4"
BLUE_LIGHT = "#eaf2fd"
TEAL = "#0f766e"
TEAL_LIGHT = "#e8faf6"
ORANGE = "#d97706"
ORANGE_LIGHT = "#fff4e8"
PURPLE = "#6d4aa2"
PURPLE_LIGHT = "#f5effd"
GRAY = "#4b5563"
BORDER = "#d3dce8"
TEXT = "#1f2937"


def load_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for item in candidates:
        path = Path(item)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


TITLE_FONT = load_font(42, bold=True)
SUBTITLE_FONT = load_font(20)
SECTION_FONT = load_font(18, bold=True)
CARD_TITLE_FONT = load_font(22, bold=True)
CARD_BODY_FONT = load_font(16)
TAG_FONT = load_font(15, bold=True)
FOOTER_FONT = load_font(15)


def text_size(draw, text, font_obj):
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font_obj)
        return right - left, bottom - top
    return draw.textsize(text, font=font_obj)


def wrap_text(draw, text, font_obj, width):
    lines = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        current = ""
        for word in raw.split(" "):
            probe = word if not current else current + " " + word
            if text_size(draw, probe, font_obj)[0] <= width:
                current = probe
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def rounded_box(draw, box, radius, fill, outline, width=1):
    x1, y1, x2, y2 = box
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        return

    draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill)
    draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill)
    draw.ellipse((x1, y1, x1 + radius * 2, y1 + radius * 2), fill=fill)
    draw.ellipse((x2 - radius * 2, y1, x2, y1 + radius * 2), fill=fill)
    draw.ellipse((x1, y2 - radius * 2, x1 + radius * 2, y2), fill=fill)
    draw.ellipse((x2 - radius * 2, y2 - radius * 2, x2, y2), fill=fill)

    if outline and width > 0:
        draw.arc((x1, y1, x1 + radius * 2, y1 + radius * 2), 180, 270, fill=outline, width=width)
        draw.arc((x2 - radius * 2, y1, x2, y1 + radius * 2), 270, 360, fill=outline, width=width)
        draw.arc((x1, y2 - radius * 2, x1 + radius * 2, y2), 90, 180, fill=outline, width=width)
        draw.arc((x2 - radius * 2, y2 - radius * 2, x2, y2), 0, 90, fill=outline, width=width)
        draw.line((x1 + radius, y1, x2 - radius, y1), fill=outline, width=width)
        draw.line((x1 + radius, y2, x2 - radius, y2), fill=outline, width=width)
        draw.line((x1, y1 + radius, x1, y2 - radius), fill=outline, width=width)
        draw.line((x2, y1 + radius, x2, y2 - radius), fill=outline, width=width)


def draw_card(draw, box, title, body, fill, outline, title_fill=NAVY):
    rounded_box(draw, box, radius=28, fill=fill, outline=outline, width=3)
    x1, y1, x2, _ = box
    draw.text((x1 + 22, y1 + 16), title, font=CARD_TITLE_FONT, fill=title_fill)
    top = y1 + 58
    for line in body:
        draw.text((x1 + 22, top), line, font=CARD_BODY_FONT, fill=TEXT)
        top += 29


def draw_tag(draw, center, text, fill="#ffffff", outline=BORDER, text_fill=GRAY):
    x, y = center
    w = text_size(draw, text, TAG_FONT)[0] + 24
    h = 34
    box = (x - w // 2, y - h // 2, x + w // 2, y + h // 2)
    rounded_box(draw, box, radius=12, fill=fill, outline=outline, width=2)
    draw.text((box[0] + 12, box[1] + 7), text, font=TAG_FONT, fill=text_fill)


def arrow(draw, start, end, color=BLUE, width=6):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        draw.polygon([(x2, y2), (x2 - 18 * sign, y2 - 10), (x2 - 18 * sign, y2 + 10)], fill=color)
    else:
        sign = 1 if y2 >= y1 else -1
        draw.polygon([(x2, y2), (x2 - 10, y2 - 18 * sign), (x2 + 10, y2 - 18 * sign)], fill=color)


def elbow_arrow(draw, points, color=BLUE, width=6):
    draw.line(points, fill=color, width=width)
    arrow(draw, points[-2], points[-1], color=color, width=width)


img = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(img)

draw.text((56, 42), "RK3588 Multimodal PCB Fault Diagnosis Architecture", font=TITLE_FONT, fill=NAVY)
draw.text(
    (58, 96),
    "Visible imaging + thermal imaging + board coordinates + local/cloud LLM collaborative reasoning",
    font=SUBTITLE_FONT,
    fill=GRAY,
)
draw.line((56, 132, WIDTH - 56, 132), fill="#bdd0e7", width=4)

draw.text((56, 168), "Input Layer", font=SECTION_FONT, fill=BLUE)
draw_card(
    draw,
    (56, 204, 360, 362),
    "Visible Camera",
    [
        "UVC high-resolution video stream",
        "Component texture and silkscreen context",
        "V4L2 + MJPG real-time capture",
    ],
    BLUE_LIGHT,
    BLUE,
)
draw_card(
    draw,
    (56, 390, 360, 548),
    "Thermal Camera",
    [
        "Hikvision pseudo-color thermal stream",
        "Crosshair and hotspot cues",
        "High-temperature region evidence",
    ],
    ORANGE_LIGHT,
    ORANGE,
    title_fill=ORANGE,
)
draw_card(
    draw,
    (56, 576, 360, 734),
    "Board Data",
    [
        "Coordinates / netlist-derived board model",
        "Mounting-hole geometry and board size",
        "Component attributes for diagnosis context",
    ],
    TEAL_LIGHT,
    TEAL,
    title_fill=TEAL,
)

rounded_box(draw, (412, 166, 1278, 876), radius=34, fill="#ffffff", outline=BORDER, width=3)
draw.text((438, 186), "RK3588 Edge Runtime", font=SECTION_FONT, fill=NAVY)
draw.text((620, 186), "On-device acquisition, fusion, localization and reasoning", font=FOOTER_FONT, fill=GRAY)

draw_card(
    draw,
    (442, 226, 1248, 366),
    "Acquisition and Spatial Registration",
    [
        "Dual-thread capture for visible and thermal streams",
        "Four-hole calibration for initial perspective matrix",
        "Board-contour refinement, thermal rotation and ROI alignment",
    ],
    "#eef7ff",
    BLUE,
)
draw_card(
    draw,
    (442, 394, 1248, 552),
    "Fusion, Mapping and Hotspot Parsing",
    [
        "Thermal overlay on visible PCB view",
        "Board-model mapping from coordinates to component references",
        "Hotspot ownership from crosshair, color peak and OCR temperature",
    ],
    BLUE_LIGHT,
    BLUE,
)
draw_card(
    draw,
    (442, 580, 814, 786),
    "Local Inference Path",
    [
        "YOLOv8 OBB / RKNN / RKNNLite / NPU",
        "Local RKLLM diagnostic text generation",
        "Offline and low-latency demo mode",
    ],
    PURPLE_LIGHT,
    PURPLE,
    title_fill=PURPLE,
)
draw_card(
    draw,
    (876, 580, 1248, 786),
    "Cloud Inference Path",
    [
        "DeepSeek API enhanced diagnosis",
        "Probability-based fault explanation",
        "Richer interpretation when network is available",
    ],
    "#f2f8ff",
    BLUE,
)

draw.text((1312, 168), "Presentation Layer", font=SECTION_FONT, fill=BLUE)
draw_card(
    draw,
    (1312, 204, 1738, 430),
    "Web Dashboard",
    [
        "Visible / thermal / fusion live preview",
        "Manual calibration and fusion tuning",
        "Model switching and report triggering",
    ],
    BLUE_LIGHT,
    BLUE,
)
draw_card(
    draw,
    (1312, 468, 1738, 736),
    "Outputs",
    [
        "Fusion frame and hotspot snapshots",
        "Component reference localization",
        "Chinese PDF diagnostic report",
        "Competition-ready visual demo loop",
    ],
    TEAL_LIGHT,
    TEAL,
    title_fill=TEAL,
)

draw_card(
    draw,
    (442, 908, 1248, 1026),
    "Engineering Deployment",
    [
        "SSH / Paramiko delivery   |   systemd --user service   |   desktop start / restart / stop automation",
    ],
    "#f8fafc",
    "#94a3b8",
    title_fill=GRAY,
)

elbow_arrow(draw, [(360, 284), (412, 284), (412, 296), (442, 296)], color=BLUE)
draw_tag(draw, (388, 258), "video stream")
elbow_arrow(draw, [(360, 470), (412, 470), (412, 296), (442, 296)], color=ORANGE)
draw_tag(draw, (388, 444), "thermal stream", fill="#fffaf3", outline="#f3cb98", text_fill=ORANGE)
elbow_arrow(draw, [(360, 656), (412, 656), (412, 474), (442, 474)], color=TEAL)
draw_tag(draw, (388, 630), "board model", fill="#f3fffb", outline="#98d6c7", text_fill=TEAL)

arrow(draw, (846, 366), (846, 394), color=BLUE)
draw_tag(draw, (934, 380), "fusion ROI")
arrow(draw, (628, 552), (628, 580), color=PURPLE)
draw_tag(draw, (562, 566), "hotspot context", fill="#faf5ff", outline="#d7c5ef", text_fill=PURPLE)
arrow(draw, (1060, 552), (1060, 580), color=BLUE)
draw_tag(draw, (1142, 566), "diagnosis context")

elbow_arrow(draw, [(1248, 294), (1286, 294), (1286, 316), (1312, 316)], color=BLUE)
elbow_arrow(draw, [(1248, 474), (1286, 474), (1286, 602), (1312, 602)], color=BLUE)
elbow_arrow(draw, [(1248, 682), (1286, 682), (1286, 602), (1312, 602)], color=PURPLE)
draw_tag(draw, (1286, 646), "reasoning result", fill="#faf5ff", outline="#d7c5ef", text_fill=PURPLE)

arrow(draw, (846, 786), (846, 908), color=GRAY)
draw_tag(draw, (846, 852), "deployment loop", outline="#d8e0ea")

footer = (
    "The pipeline forms a closed loop on RK3588: heterogeneous sensing -> calibration and fusion -> "
    "component-level hotspot attribution -> local/cloud reasoning -> visual report output."
)
draw.text((56, 1042), footer, font=FOOTER_FONT, fill=GRAY)

img.save(str(OUT), quality=95)
print(OUT)
