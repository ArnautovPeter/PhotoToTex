"""Run a local web interface for photo-to-LaTeX inference."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError

import torch

from .infer import load_checkpoint, predict_image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

PAGE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhotoToTex — распознавание формулы</title>
  <style>
    :root { color-scheme: light; font: 16px/1.5 system-ui, sans-serif; color: #172033; background: #f3f5fa; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 48px 20px; }
    main { max-width: 760px; margin: auto; padding: 32px; border-radius: 18px; background: white; box-shadow: 0 12px 40px #26334d14; }
    h1 { margin: 0 0 8px; font-size: 1.8rem; }
    p { color: #586174; }
    .hint { margin: 0 0 24px; }
    .tabs { display: flex; gap: 8px; margin: 24px 0; border-bottom: 1px solid #e4e8f0; }
    .tab { border-radius: 8px 8px 0 0; background: transparent; color: #586174; }
    .tab[aria-selected="true"] { color: #465bd1; box-shadow: inset 0 -3px #465bd1; }
    .panel[hidden] { display: none; }
    .picker { display: block; padding: 24px; border: 2px dashed #aab5c8; border-radius: 12px; text-align: center; cursor: pointer; }
    .picker:hover { border-color: #5267d8; background: #f8f9ff; }
    input[type=file] { display: block; width: 100%; margin-top: 12px; }
    img { display: none; max-width: 100%; max-height: 320px; margin: 20px auto; border-radius: 8px; }
    button { border: 0; border-radius: 8px; padding: 11px 16px; background: #465bd1; color: white; font: inherit; cursor: pointer; }
    button:disabled { opacity: .55; cursor: wait; }
    .actions { display: flex; gap: 10px; margin: 16px 0; }
    .secondary { background: #e9ecf5; color: #26334d; }
    textarea { width: 100%; min-height: 110px; padding: 12px; border: 1px solid #cbd2df; border-radius: 8px; font: 15px/1.5 ui-monospace, monospace; }
    #status { min-height: 24px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 18px 0; }
    .card, .chart-card { padding: 16px; border: 1px solid #e4e8f0; border-radius: 12px; background: #fbfcff; }
    .card strong { display: block; margin-top: 4px; font-size: 1.35rem; color: #26334d; }
    .card span, .muted { color: #68738a; font-size: .9rem; }
    .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
    .chart-card h3 { margin: 0 0 10px; font-size: 1rem; }
    .chart { width: 100%; height: auto; overflow: visible; }
    .chart text { fill: #8791a4; font: 11px system-ui, sans-serif; }
    .chart .grid { stroke: #e8ecf3; stroke-width: 1; }
    .chart .line { fill: none; stroke: #5369dc; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }
    .chart .dot { fill: #5369dc; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; margin: 14px 0 22px; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e9edf4; text-align: left; }
    th { color: #68738a; font-size: .85rem; font-weight: 600; }
    .split-table td:not(:first-child) { font-variant-numeric: tabular-nums; }
    .notice { padding: 12px 14px; border-radius: 8px; background: #f3f5fa; color: #586174; }
    @media (max-width: 520px) { body { padding: 16px 10px; } main { padding: 22px 16px; } }
  </style>
</head>
<body>
  <main>
    <h1>PhotoToTex</h1>
    <p class="hint">Загрузите изображение с одной обрезанной формулой. Оно обрабатывается локально и не отправляется в интернет.</p>
    <nav class="tabs" role="tablist" aria-label="Разделы PhotoToTex">
      <button class="tab" id="recognition-tab" role="tab" aria-selected="true" aria-controls="recognition-panel">Распознавание</button>
      <button class="tab" id="model-tab" role="tab" aria-selected="false" aria-controls="model-panel">Модель и результаты</button>
    </nav>
    <section class="panel" id="recognition-panel" role="tabpanel" aria-labelledby="recognition-tab">
      <label class="picker">
        Выберите изображение (PNG, JPEG, WebP или BMP, до 10 МБ)
        <input id="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp">
      </label>
      <img id="preview" alt="Предпросмотр изображения">
      <div class="actions">
        <button id="recognize" disabled>Распознать формулу</button>
        <button id="copy" class="secondary" disabled>Скопировать LaTeX</button>
        <button id="download" class="secondary" disabled>Скачать .tex</button>
      </div>
      <p id="status" role="status"></p>
      <label for="result">Результат LaTeX</label>
      <textarea id="result" placeholder="Здесь появится результат распознавания" spellcheck="false"></textarea>
      <p>Скачанный файл содержит фрагмент формулы; компиляция в PDF пока не включена.</p>
    </section>
    <section class="panel" id="model-panel" role="tabpanel" aria-labelledby="model-tab" hidden>
      <div id="model-status" class="notice">Загружаем сведения о модели…</div>
      <div id="model-content" hidden>
        <div class="cards" id="summary-cards"></div>
        <h2>Результаты по выборкам</h2>
        <p class="notice">Значения рассчитаны на синтетических формулах из этого проекта и не предсказывают точность на реальных фото или рукописном тексте.</p>
        <div class="table-wrap">
          <table class="split-table">
            <thead><tr><th>Выборка</th><th>Примеров</th><th>Точное совпадение</th><th>Ошибки символов (CER)</th><th>Неизвестные символы</th></tr></thead>
            <tbody id="metrics-table"></tbody>
          </table>
        </div>
        <h2>История обучения</h2>
        <div id="history-note" class="notice" hidden></div>
        <div class="charts">
          <div class="chart-card"><h3>Train loss</h3><svg id="loss-chart" class="chart" viewBox="0 0 480 220" role="img" aria-label="График train loss"></svg></div>
          <div class="chart-card"><h3>Validation CER</h3><svg id="cer-chart" class="chart" viewBox="0 0 480 220" role="img" aria-label="График validation character error rate"></svg></div>
        </div>
        <h2>Параметры модели</h2>
        <div class="table-wrap">
          <table><tbody id="config-table"></tbody></table>
        </div>
        <p class="muted" id="checkpoint-name"></p>
      </div>
    </section>
  </main>
  <script>
    const fileInput = document.querySelector("#file");
    const preview = document.querySelector("#preview");
    const recognize = document.querySelector("#recognize");
    const copy = document.querySelector("#copy");
    const download = document.querySelector("#download");
    const result = document.querySelector("#result");
    const status = document.querySelector("#status");
    const recognitionTab = document.querySelector("#recognition-tab");
    const modelTab = document.querySelector("#model-tab");
    const recognitionPanel = document.querySelector("#recognition-panel");
    const modelPanel = document.querySelector("#model-panel");
    const modelStatus = document.querySelector("#model-status");
    const modelContent = document.querySelector("#model-content");
    let selectedFile = null;
    let modelPolling = null;

    function selectTab(tab) {
      const isModel = tab === modelTab;
      recognitionTab.setAttribute("aria-selected", String(!isModel));
      modelTab.setAttribute("aria-selected", String(isModel));
      recognitionPanel.hidden = isModel;
      modelPanel.hidden = !isModel;
      if (isModel) {
        loadModelInfo();
        if (!modelPolling) modelPolling = setInterval(loadModelInfo, 5000);
      } else if (modelPolling) {
        clearInterval(modelPolling);
        modelPolling = null;
      }
    }

    recognitionTab.addEventListener("click", () => selectTab(recognitionTab));
    modelTab.addEventListener("click", () => selectTab(modelTab));

    function formatPercent(value) {
      return Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "—";
    }

    function appendCell(row, value, heading = false) {
      const cell = document.createElement(heading ? "th" : "td");
      cell.textContent = value;
      row.append(cell);
    }

    function drawChart(element, values, label) {
      element.replaceChildren();
      if (!values.length) return;
      const ns = "http://www.w3.org/2000/svg";
      const make = (name, attrs) => {
        const node = document.createElementNS(ns, name);
        for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
        element.append(node);
        return node;
      };
      const left = 48, right = 462, top = 14, bottom = 180;
      const numbers = values.map(item => item.value).filter(Number.isFinite);
      if (!numbers.length) return;
      let low = Math.min(...numbers), high = Math.max(...numbers);
      if (low === high) { low = Math.max(0, low - 0.5); high += 0.5; }
      const padding = (high - low) * 0.08;
      low = Math.max(0, low - padding);
      high += padding;
      for (let index = 0; index < 4; index++) {
        const y = top + (bottom - top) * index / 3;
        make("line", { x1: left, y1: y, x2: right, y2: y, class: "grid" });
        const tick = make("text", { x: left - 7, y: y + 4, "text-anchor": "end" });
        tick.textContent = (high - (high - low) * index / 3).toFixed(2);
      }
      const coordinates = values.map((item, index) => {
        const x = values.length === 1 ? (left + right) / 2 : left + (right - left) * index / (values.length - 1);
        const y = bottom - (item.value - low) / (high - low) * (bottom - top);
        return `${x},${y}`;
      }).join(" ");
      make("polyline", { points: coordinates, class: "line" });
      const first = values[0].epoch, last = values[values.length - 1].epoch;
      const firstLabel = make("text", { x: left, y: 205, "text-anchor": "start" });
      firstLabel.textContent = `эпоха ${first}`;
      const lastLabel = make("text", { x: right, y: 205, "text-anchor": "end" });
      lastLabel.textContent = `эпоха ${last}`;
      const title = make("title", {});
      title.textContent = `${label}: ${values.length} эпох`;
    }

    async function loadModelInfo() {
      try {
        const response = await fetch("/api/model", { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Не удалось загрузить сведения о модели.");
        modelStatus.hidden = true;
        modelContent.hidden = false;
        const cards = document.querySelector("#summary-cards");
        cards.replaceChildren();
        const cardData = [
          ["Параметров", Number(payload.parameter_count).toLocaleString("ru-RU")],
          ["Размер словаря", payload.vocab_size],
          ["Лучшая эпоха", payload.epoch ?? "—"],
          ["Validation exact", formatPercent(payload.validation?.exact_match)]
        ];
        for (const [label, value] of cardData) {
          const card = document.createElement("div");
          card.className = "card";
          const caption = document.createElement("span");
          caption.textContent = label;
          const strong = document.createElement("strong");
          strong.textContent = value;
          card.append(caption, strong);
          cards.append(card);
        }

        const splitRows = document.querySelector("#metrics-table");
        splitRows.replaceChildren();
        const splitMetrics = payload.metrics?.splits || {};
        const splitLabels = { val: "Validation", test: "Test", train: "Train", real_val: "Реальные фото (val)" };
        const splitNames = ["val", "test", ...Object.keys(splitMetrics).filter(name => !["val", "test"].includes(name))];
        for (const split of splitNames) {
          const metrics = splitMetrics[split];
          const row = document.createElement("tr");
          appendCell(row, splitLabels[split] || split);
          appendCell(row, metrics ? Number(metrics.count).toLocaleString("ru-RU") : "не рассчитано");
          appendCell(row, formatPercent(metrics?.exact_match));
          appendCell(row, formatPercent(metrics?.cer));
          appendCell(row, formatPercent(metrics?.unknown_char_rate));
          splitRows.append(row);
        }

        const configRows = document.querySelector("#config-table");
        configRows.replaceChildren();
        const displayConfig = [
          ["Устройство обучения", payload.config.device || "—"],
          ["Размер изображения", Array.isArray(payload.config.image_size) ? payload.config.image_size.join(" × ") : payload.config.image_size],
          ["Слойность Transformer", payload.config.layers],
          ["Размерность модели", payload.config.d_model],
          ["Голов внимания", payload.config.nhead],
          ["Размер FFN", payload.config.ff_dim],
          ["Максимальная длина", payload.config.max_length],
          ["Размер обучающей выборки", payload.dataset_size ?? "—"]
        ];
        for (const [key, value] of displayConfig) {
          const row = document.createElement("tr");
          appendCell(row, key, true);
          appendCell(row, String(value ?? "—"));
          configRows.append(row);
        }

        const history = Array.isArray(payload.history) ? payload.history : [];
        const historyNote = document.querySelector("#history-note");
        historyNote.hidden = history.length > 0;
        historyNote.textContent = "История эпох для этого checkpoint не сохранена. Графики появятся после следующего обучения с обновлённой версией train-кода.";
        drawChart(document.querySelector("#loss-chart"), history.map(item => ({ epoch: item.epoch, value: item.train_loss })), "Train loss");
        drawChart(document.querySelector("#cer-chart"), history.map(item => ({ epoch: item.epoch, value: item.val_cer })), "Validation CER");
        document.querySelector("#checkpoint-name").textContent = `Checkpoint: ${payload.checkpoint}`;
      } catch (error) {
        modelStatus.hidden = false;
        modelStatus.textContent = error.message;
      }
    }

    fileInput.addEventListener("change", () => {
      selectedFile = fileInput.files[0] || null;
      recognize.disabled = !selectedFile;
      result.value = "";
      copy.disabled = true;
      download.disabled = true;
      status.textContent = "";
      if (selectedFile) {
        preview.src = URL.createObjectURL(selectedFile);
        preview.style.display = "block";
      } else {
        preview.removeAttribute("src");
        preview.style.display = "none";
      }
    });

    recognize.addEventListener("click", async () => {
      if (!selectedFile) return;
      recognize.disabled = true;
      status.textContent = "Распознаём формулу…";
      try {
        const dataUrl = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = () => reject(new Error("Не удалось прочитать файл."));
          reader.readAsDataURL(selectedFile);
        });
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: dataUrl.split(",", 2)[1] })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Ошибка распознавания.");
        result.value = payload.latex;
        copy.disabled = false;
        download.disabled = false;
        status.textContent = "Готово. Проверьте распознанный LaTeX.";
      } catch (error) {
        status.textContent = error.message;
      } finally {
        recognize.disabled = !selectedFile;
      }
    });

    copy.addEventListener("click", async () => {
      await navigator.clipboard.writeText(result.value);
      status.textContent = "LaTeX скопирован.";
    });

    download.addEventListener("click", () => {
      const blob = new Blob([result.value + "\\n"], { type: "text/plain;charset=utf-8" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "formula.tex";
      link.click();
      URL.revokeObjectURL(link.href);
    });
  </script>
</body>
</html>
"""


def decode_image(encoded: str) -> Image.Image:
    if not isinstance(encoded, str) or len(encoded) > ((MAX_UPLOAD_BYTES + 2) // 3) * 4:
        raise ValueError("Изображение превышает ограничение в 10 МБ.")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Не удалось прочитать изображение.") from error
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Изображение превышает ограничение в 10 МБ.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP", "BMP"}:
                raise ValueError("Поддерживаются только PNG, JPEG, WebP и BMP.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Изображение слишком большое по разрешению.")
            source.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Файл не является корректным изображением.") from error


def make_handler(
    predict: Callable[[Image.Image], str],
    model_info: Callable[[], dict[str, object]] | None = None,
) -> type[BaseHTTPRequestHandler]:
    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/api/model":
                if model_info is None:
                    self._send_json(503, {"error": "Сведения о модели недоступны."})
                    return
                try:
                    self._send_json(200, model_info())
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    logging.exception("Could not load model information")
                    self._send_json(500, {"error": f"Не удалось загрузить сведения о модели: {error}"})
                return
            if self.path != "/":
                self.send_error(404)
                return
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/api/predict":
                self.send_error(404)
                return
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._send_json(415, {"error": "Ожидался запрос с JSON."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_json(400, {"error": "Некорректный размер запроса."})
                return
            if content_length <= 0 or content_length > MAX_UPLOAD_BYTES * 2:
                self._send_json(413, {"error": "Запрос пустой или слишком большой."})
                return

            try:
                payload = json.loads(self.rfile.read(content_length))
                image = decode_image(payload.get("image") if isinstance(payload, dict) else None)
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError, ValueError) as error:
                self._send_json(400, {"error": str(error) or "Некорректный запрос."})
                return

            try:
                latex = predict(image)
            except (RuntimeError, KeyError, ValueError) as error:
                logging.exception("Formula inference failed")
                self._send_json(500, {"error": f"Ошибка распознавания: {error}"})
                return
            self._send_json(200, {"latex": latex})

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return RequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальный интерфейс PhotoToTex")
    parser.add_argument("--checkpoint", default="checkpoints/baseline.pt", help="Путь к файлу модели")
    parser.add_argument("--port", type=int, default=8000, help="Порт локального веб-сервера")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, tokenizer, config = load_checkpoint(args.checkpoint, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    validation = checkpoint.get("validation", {})
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_history = checkpoint.get("history", [])
    del checkpoint
    checkpoint_path = Path(args.checkpoint)

    predict = lambda image: predict_image(image, model, tokenizer, config)
    def model_info() -> dict[str, object]:
        history_path = checkpoint_path.with_suffix(".history.json")
        metrics_path = checkpoint_path.with_suffix(".metrics.json")
        history = checkpoint_history
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
        metrics = {"splits": {}}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        train_file = Path(config["data_dir"]) / "train.jsonl"
        dataset_size = sum(1 for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()) if train_file.is_file() else None
        return {
            "checkpoint": checkpoint_path.name,
            "epoch": checkpoint_epoch,
            "validation": validation,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "vocab_size": len(tokenizer),
            "config": config,
            "dataset_size": dataset_size,
            "history": history,
            "metrics": metrics,
        }

    server = HTTPServer(("127.0.0.1", args.port), make_handler(predict, model_info))
    print(f"Откройте http://127.0.0.1:{args.port} в браузере. Для остановки нажмите Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
