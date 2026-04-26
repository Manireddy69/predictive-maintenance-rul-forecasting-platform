from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SLIDES = [
    ("01_repo_structure.png", "Project structure: code, data, reports, tests, dashboard, deployment"),
    ("02_eda_summary.png", "Data foundation: NASA CMAPSS FD001 and degradation signals"),
    ("03_anomaly_results.png", "Anomaly detection: F1 target met by multiple methods"),
    ("05_rul_metrics.png", "RUL forecasting: useful prototype, MAPE limitation documented"),
    ("06_scheduler_summary.png", "Maintenance optimization: predictions become scheduled actions"),
    ("07_cost_risk_plot.png", "Sensitivity analysis: cost and risk trade-off review"),
    ("09_dashboard_overview.png", "Dashboard overview: operational schedule health"),
    ("10_dashboard_equipment.png", "Equipment detail: unit-level forecast inspection"),
    ("11_dashboard_alerts.png", "Alert configuration: thresholds and escalation draft"),
    ("12_dashboard_reports.png", "Reports: exportable evidence for reviewers"),
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _font(size: int) -> ImageFont.ImageFont:
    for path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_frame(source: Path, caption: str, size: tuple[int, int] = (1280, 720)) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(34)
    caption_font = _font(24)

    image = Image.open(source).convert("RGB")
    image.thumbnail((size[0] - 80, size[1] - 150), Image.Resampling.LANCZOS)
    x = (size[0] - image.width) // 2
    y = 72
    canvas.paste(image, (x, y))

    draw.rectangle((0, 0, size[0], 54), fill="#173a63")
    draw.text((28, 12), "Predictive Maintenance & RUL Forecasting Platform", fill="white", font=caption_font)
    draw.rectangle((0, size[1] - 72, size[0], size[1]), fill="#eef3f8")
    draw.text((28, size[1] - 50), caption, fill="#17202a", font=title_font)
    return canvas


def main() -> None:
    root = project_root()
    screenshot_dir = root / "reports" / "screenshots"
    output_dir = root / "reports" / "demo_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    for index, (filename, caption) in enumerate(SLIDES, start=1):
        frame = make_frame(screenshot_dir / filename, caption)
        frame_path = output_dir / f"demo_slide_{index:02d}.png"
        frame.save(frame_path)
        frames.append(frame)

    gif_path = output_dir / "demo_storyboard.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=2200,
        loop=0,
        optimize=True,
    )
    print(f"Demo storyboard GIF written to: {gif_path}")
    print(f"Demo slide PNGs written to: {output_dir}")


if __name__ == "__main__":
    main()
