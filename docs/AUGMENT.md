# GPU augmentation for training (`slangimage.augment`)

`slangimage.augment` runs training augmentation on CUDA for PyTorch. You
write a pipeline with torchvision's transform names; SlangImage compiles it
into a few fused Slang kernels that draw separate random values for every
image in the batch.

```python
import torch
import slangimage.augment as A

train = A.Compose([
    A.RandomResizedCrop(224),
    A.RandomHorizontalFlip(),
    A.ColorJitter(0.4, 0.4, 0.4, 0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

images = torch.randint(0, 256, (64, 3, 480, 640), dtype=torch.uint8, device="cuda")
x = train(images)      # (64, 3, 224, 224) float16, on the current CUDA stream
```

- **Per-image randomness.** On a batch, torchvision's v2 transforms draw one
  random setting for the whole batch; getting a different crop and colour
  for every image takes a Python loop over the images. `slangimage.augment`
  draws per image by default; `same_on_batch=True` on a transform shares its
  values across the batch.
- **torchvision results.** Each op matches torchvision's float functionals
  image by image, within float rounding (see [Accuracy](#accuracy)).
- **Fused.** A pipeline compiles into a handful of kernels (one pass per run
  of colour ops, one more per blur), and reads the decoded images where they
  already are.
- **Your own ops** in a few lines of Python and one Slang function
  ([Custom ops](#custom-ops)).

It does not decode images: pair it with your decoder
([Inputs and decoders](#inputs-and-decoders)).

## Requirements

An NVIDIA GPU, PyTorch with CUDA, and NVRTC (as for the CUDA backend; see the
README section [CUDA on Linux: install NVRTC](https://github.com/tekintatar/slangimage#cuda-on-linux-install-nvrtc)):

```bash
pip install slangimage torch torchvision
```

torchvision is needed only for `Compose.max_reference_error`. The compiled
`slangpy-torch` bridge (the three-step [PyTorch extra](https://github.com/tekintatar/slangimage#pytorch-extra-slangimagetorch)
for `slangimage.diff`) is optional here: without it, `slangimage.augment`
uses SlangPy's pure-Python torch bridge for its own calls, which adds under
0.1 ms per batch. `import slangimage` never imports torch; `slangimage.augment` imports
it when a pipeline runs.

The first call of a pipeline compiles it (about 1 s, plus about 1 s to set up
the GPU device once per process); later calls reuse the compiled kernels.

## Transforms

| transform | as in torchvision, except |
|---|---|
| `RandomResizedCrop(size, scale, ratio, antialias=True)` | box drawn per image |
| `Resize(size, antialias=True)` | an int resizes the shorter side; follow it with `CenterCrop` so all images match |
| `CenterCrop(size)` | — (a crop larger than the image pads with zeros, as torchvision does) |
| `RandomHorizontalFlip(p=0.5)` | decided per image |
| `ColorJitter(brightness, contrast, saturation, hue)` | factors and op order drawn per image |
| `RandomApply(transforms, p=0.5)` | decided per image; holds colour ops or exactly one blur |
| `RandomGrayscale(p=0.1)` | decided per image; output keeps 3 channels |
| `GaussianBlur(kernel_size, sigma=(0.1, 2.0))` | sigma drawn per image; the kernel radius must be smaller than the image, as torchvision's reflect padding requires |
| `RandomSolarize(threshold, p=0.5)` | `threshold` is on the 0-1 scale (0.5, not 128) |
| `Normalize(mean, std)` | must be last |

Every random transform takes `same_on_batch=False`. Bilinear interpolation
only.

Pipeline rules, checked when you build the `Compose`:
- Geometry comes first: `RandomResizedCrop`, or `Resize` then `CenterCrop`,
  or `CenterCrop` alone. It fixes the output size. With no geometric
  transform, the input must be one same-size batch and the output keeps its
  size.
- `RandomHorizontalFlip` goes anywhere after the geometric transforms and
  before any `NeighbourhoodOp` you define (it commutes with colour ops and
  with Gaussian blur).
- Each transform object appears once; create a second one to use it twice.
- `Normalize`, if present, is last. Without it the output is in [0, 1].

`Compose(transforms, dtype=torch.float16)`; `torch.float32` is the other
choice. The output is a contiguous `(B, 3, H, W)` tensor; for another dtype
or `channels_last`, call `.to(...)` on it.

## Inputs and decoders

`pipe(images)` takes:
- a `(B, 3, H, W)` uint8 CUDA tensor, or
- a list of uint8 CUDA tensors of any sizes, `(3, H, W)` by default or
  `(H, W, 3)` with `layout="hwc"` (a `(B, H, W, 3)` tensor also works with
  `layout="hwc"`).

Images are read in place, without copying them into one buffer
(non-contiguous tensors are made contiguous first). **They must stay valid
until the GPU has run the pipeline.** Tensors PyTorch allocated on
the current stream always are, because PyTorch reuses freed memory only in
stream order. Memory another library owns needs a sync before that library
reuses it.

Decoding usually limits training throughput, so pick a fast decoder:

```py
# torchvision (0.28) GPU decode, CHW
from torchvision.io import ImageReadMode, decode_jpeg
x = train(decode_jpeg(list_of_jpeg_byte_tensors, mode=ImageReadMode.RGB, device="cuda"))

# nvImageCodec (hardware JPEG decoder on A100/H100-class GPUs), HWC
from nvidia import nvimgcodec
decoder = nvimgcodec.Decoder()
imgs = decoder.decode(list_of_jpeg_bytes, cuda_stream=torch.cuda.current_stream().cuda_stream)
x = train([torch.as_tensor(im, device="cuda") for im in imgs], layout="hwc")
torch.cuda.current_stream().synchronize()   # before `imgs` is freed

# DALI decoding (fn.decoders.image, device="mixed"), HWC. Import
# slangimage.augment before building the DALI pipeline (see below). DALI's
# outputs are valid only until its next run(), so wait for the augmentation
# (not the whole training step) before calling it again.
ready = None
while training:
    if ready is not None:
        ready.synchronize()
    images, labels = dali_pipeline.run()
    x = train([torch.as_tensor(images[i], device="cuda") for i in range(len(images))],
              layout="hwc")
    ready = torch.cuda.Event()
    ready.record()
    ...                                      # training step on x
```

**With DALI, import `slangimage.augment` (or `slangimage`) before building a
DALI pipeline.** If DALI loads its native libraries first, SlangPy crashes
with a segmentation fault while creating its CUDA device (seen with DALI 2.4
and SlangPy 0.42); in the other order both work in one process.

Without a GPU decoder, keep your `DataLoader` but let the workers only
decode, and augment the batch on the GPU. Have the workers pack each batch
into one buffer: with `collate_fn=list`, PyTorch shares every image between
processes separately, which made training slower than augmenting on the CPU
(2,037 against 2,721 images/s for ResNet-18 in the benchmark below).

```py
def collate(batch):                  # runs in the workers
    images, labels = zip(*batch)     # decoded uint8 (3, H, W) tensors
    return (torch.cat([im.reshape(-1) for im in images]),
            torch.tensor([tuple(im.shape) for im in images]), torch.tensor(labels))

loader = torch.utils.data.DataLoader(decoded_dataset, batch_size=256, num_workers=16,
                                     pin_memory=True, collate_fn=collate)
for flat, shapes, labels in loader:
    parts = flat.cuda(non_blocking=True).split(shapes.prod(1).tolist())
    x = train([part.view(*s) for part, s in zip(parts, shapes.tolist())])
```

## Randomness

Random values come from a counter-based generator in the sampler kernel,
seeded on every call from PyTorch's CUDA generator: `torch.manual_seed`
makes runs repeat, and there is no host sync. Values are drawn per image;
`same_on_batch=True` on a transform gives every image the same values for
that transform (for `RandomApply`, the same apply/skip decision).

## Custom ops

A colour op declares its random parameters and one Slang function:

```python
import slangimage.augment as A

class RandomGamma(A.ColorOp):
    """Gamma correction with a random exponent per image."""
    params = {"gamma": A.Uniform(0.8, 1.25)}
    slang = "float4 apply(float4 c, float gamma) { return clamp01(pow(c, float4(gamma))); }"

    def reference(self, img, gamma):          # optional: torch, one (3, H, W) image
        return img.clamp(0, 1) ** gamma

pipe = A.Compose([A.RandomResizedCrop(224), RandomGamma(p=0.5)])
```

- `apply` receives the pixel (`float4`, rgb in `xyz`, values in [0, 1]) and
  one `float` per entry of `params`, in order, and returns the new pixel.
  Clamp to [0, 1] if your op can leave that range, as torchvision's ops do.
- `params` values are `A.Uniform(low, high)`, `A.Choice(values)` or a plain
  number (a constant). Pass a `params` dict in your own `__init__` to make
  the ranges arguments.
- The constructor takes `p` (probability of applying, per image) and
  `same_on_batch`.
- Library helpers such as `gray(c)` and `clamp01(c)` are in scope (see
  `src/slangimage/shaders/augment/augment.slang`). Each op's code is placed in
  its own namespace, so helper functions you define never clash.

An op that reads neighbouring pixels subclasses `A.NeighbourhoodOp`:

```python
import slangimage.augment as A

class BoxBlur(A.NeighbourhoodOp):
    params = {"r": A.Choice([1, 2])}
    border = "reflect"                        # or "clamp"
    slang = """
    float4 apply<I : IImage>(I img, int x, int y, float r)
    {
        int k = int(r);
        float4 acc = float4(0.0);
        for (int dy = -k; dy <= k; ++dy)
            for (int dx = -k; dx <= k; ++dx)
                acc += img.load(x + dx, y + dy);
        return acc / float((2 * k + 1) * (2 * k + 1));
    }"""
```

`img.load(x, y)` reads the image as it is at that point of the pipeline,
applying `border` outside it; `img.width()` and `img.height()` give its size.
Each neighbourhood op starts a new pass over the image.

Check an op against its `reference` with
`pipe.max_reference_error(images)`: it runs the pipeline, replays each
image's random values through the transforms' references (torchvision for
the built-ins) and returns the largest absolute difference. `pipe.source`
shows the generated Slang.

## Accuracy

| | difference from torchvision's float functionals |
|---|---|
| float32 output | about 1e-5 (float rounding) |
| float16 output | fp16 rounding of the output, up to about 2e-3 at normalized values |
| threshold ops (`RandomSolarize`) | a value landing exactly on the threshold may be inverted on one side and not the other; all other values match |

torchvision's uint8 path rounds after every op; this matches its float path.

## Performance

Augmentation of a batch already on the GPU (`benchmarks/compare_augment_pipelines.py`,
batch 256, 512x384 uint8 in, 224x224 fp16 out, NVIDIA H100 NVL), wall ms:

| recipe | slangimage | torchvision batch (one setting) | torchvision per image | Kornia fp16 per image |
|---|---|---|---|---|
| ImageNet: crop + flip + ColorJitter | 1.31 | 7.32 | 389 | 16.6 |
| eval: Resize + CenterCrop | 0.68 | 2.90 | 51.9 | — |
| SimCLR: + grayscale, blur, solarize | 2.30 | 7.6-12.6 | 390 | 22.3 |

torchvision's batch mode does less work (one random setting for all 256
images, so its SimCLR time swings with whether that batch is blurred); its
per-image loop is the like-for-like comparison. Kornia's
RandomResizedCrop does not antialias.

In training, decoding usually decides throughput. Training from scratch on
an ImageNet subset (Elriggs/imagenet-50-subset: 45,000 real ImageNet JPEGs),
batch 256, bf16, eager PyTorch, on one H100 NVL with 18 CPU cores; images
per second over 45 s after warm-up, with the RandomResizedCrop + flip +
ColorJitter + Normalize recipe:

| input pipeline | ResNet-18 | ResNet-50 |
|---|---|---|
| model alone (no input pipeline) | 10,884 | 3,704 |
| DataLoader (16 workers): decode + torchvision augmentation | 2,721 | 2,671 |
| DataLoader decode + `slangimage.augment` (packed batches) | 3,781 | 3,254 |
| DALI decode + DALI augmentation | 8,655 | 3,518 |
| DALI decode + `slangimage.augment` | 8,393 | 3,395 |

- On this machine the CPU pipeline holds both models near 2,700 images/s.
  Moving only the augmentation to the GPU gives 1.4x (ResNet-18) and 1.2x
  (ResNet-50); GPU decoding gives about 3x and 1.3x.
- With DALI decoding, our augmentation runs within about 3% of DALI's own
  and gives torchvision's results; DALI's colour ops are linear
  approximations.
- More CPU cores narrow the gap to the CPU pipeline; `torch.compile` widens
  it (faster models need data faster). One run per row; the CPU rows vary by
  about ±10% between runs.

## Limits

- CUDA only (Linux with an NVIDIA GPU); the texture API keeps Metal and
  Vulkan.
- Not inside CUDA graph capture: SlangPy's CUDA layer queries the stream
  while capturing. Keep augmentation outside captured regions.
- One call handles fewer than 2^31 output values (up to about batch 14,000
  at 224x224); split larger batches.
- Run pipelines from one thread: the GPU device is created by, and tied to,
  the thread that runs the first pipeline (SlangPy devices are thread-affine).
- Not yet: RandAugment, TrivialAugment and the ops they use, affine and
  perspective geometry, MixUp and CutMix, masks and boxes.
