# Laypa Line Segmentation — Setup Guide

Laypa is a deep-learning baseline detection engine built on
[Detectron2](https://github.com/facebookresearch/detectron2).
It takes page images as input and produces PageXML files containing
`<TextLine>` and `<Baseline>` elements, which are the entry point for the
ODAOCR inference pipeline.

The canonical Laypa repo used in this project is:
```
/home/tockier/loghi/laypa          ← preferred (has public pretrained models)
/home/tockier/uni/radio/laypa      ← project fork (same codebase)
```

---

## 1. Conda environment

```bash
conda env create -f /home/tockier/loghi/laypa/environment.yml
conda activate laypa
```

Key packages installed by the environment:

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.12 | — |
| PyTorch (GPU) | 2.x | model inference |
| CUDA toolkit | 12.x | GPU acceleration |
| Detectron2 | latest | model backbone |
| OpenCV | — | image I/O |
| Flask / Gunicorn | — | optional API server |

> **No sudo needed** — everything goes into the conda env.

---

## 2. Pretrained model weights

Public models live at:
```
/home/tockier/loghi/models/public-models/laypa/
    general/
        baseline/   model_best_mIoU.pth   ← generic baseline detector
        baseline2/  model_best_mIoU.pth   ← improved generic baseline detector ✓
    republic/
        baseline/   model_best_mIoU.pth
    republicprint/
        baseline/   model_best_mIoU.pth
    cBAD/
        baseline/   model_best_mIoU.pth
```

For Radio Barcelona use **`general/baseline2/model_best_mIoU.pth`** — it
generalises best across mixed typewritten/handwritten documents.

Additional models can be downloaded from the
[SURFdrive repository](https://surfdrive.surf.nl/files/index.php/s/YA8HJuukIUKznSP?path=%2Flaypa).

---

## 3. Running inference

### Important flag: `POSTPROCESS.EXTERNAL.ENABLED False`

By default Laypa saves raw probability-map PNGs and writes empty PageXML,
expecting a separate Java tool (MinionExtractBaselines) to extract baselines.
Pass `POSTPROCESS.EXTERNAL.ENABLED False` to have Laypa do the extraction
inline — no Docker or Java required.

```bash
cd /home/tockier/loghi/laypa

conda run --no-capture-output -n laypa python inference.py \
    -c configs/segmentation/baseline/baseline_general.yaml \
    -i /path/to/input/images \
    -o /path/to/output \
    --num_workers 4 \
    --opts \
        TEST.WEIGHTS /home/tockier/loghi/models/public-models/laypa/general/baseline2/model_best_mIoU.pth \
        POSTPROCESS.EXTERNAL.ENABLED False
```

| Argument | Description |
|---|---|
| `-c` | YAML config (use `baseline_general.yaml` for most documents) |
| `-i` | Input directory — all `.png`/`.jpg` images are processed |
| `-o` | Output directory — `page/` subdirectory will contain the XMLs |
| `--num_workers` | Dataloader workers; 4–8 is typically optimal |
| `TEST.WEIGHTS` | Path to the `.pth` checkpoint |
| `POSTPROCESS.EXTERNAL.ENABLED False` | **Required** for inline baseline extraction |

### Cross-device link error

If the input and output directories are on different filesystems (e.g. input
on `/home`, output on `/data/storage`) you will get:

```
OSError: [Errno 18] Invalid cross-device link
```

**Fix:** copy input images to the same filesystem as the output before running:

```bash
cp /home/.../images/*.png /data/storage/.../input/
```

---

## 4. Output structure

```
output/
  page/
    image_0001.xml    ← PageXML with <TextLine> + <Baseline> elements
    image_0002.xml
    …
```

Each XML follows the
[PAGE XML schema](https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15).
The `<Baseline points="x1,y1 x2,y2 …">` attribute gives the polyline along
the writing baseline of each line, which is what the ODAOCR pipeline uses to
crop line images.

---

## 5. Benchmark (Radio Barcelona `page_examples`, 19 pages)

Measured on a single GPU after model warm-up:

| Metric | Value |
|---|---|
| Model load (cold) | ~6 s |
| Throughput (warm) | ~0.7 s/page |
| Full 19-page batch | ~14 s |
| Lines detected | ~50–320 per page depending on era |

---

## 6. Connecting Laypa output to ODAOCR

Once Laypa has produced the `page/*.xml` files, pass them to the OCR
pipeline:

```bash
conda activate docs
python RadioBarcelona/line-classifier/run_ocr_pipeline.py \
    --xml-dir        /path/to/laypa/output/page \
    --img-dir        /path/to/input/images \
    --output-xml-dir /path/to/ocr/output/page \
    --work-dir       /tmp/ocr_work \
    --tesseract-lang spa+cat \
    --odaocr-checkpoint    /home/tockier/uni/radio/ODAOCR/MODELS/model_metalearnt.pt \
    --odaocr-tokenizer-dir /home/tockier/uni/radio/ODAOCR/MODELS
```

Or use the standalone ODAOCR inference directly on individual crops:

```bash
conda activate docs
python /home/tockier/uni/radio/ODAOCR/infer.py /path/to/crops/ --device cuda
```
