// Pre- and post-processing for JerseyNet, kept free of DOM code so it can be tested in Node.
// They reproduce the Python pipeline in src/jersey_detect (transforms.letterbox, detector.decode).

export const SIZE = 384;        // network input, square
export const STRIDE = 4;        // output map is SIZE / STRIDE
export const PAD = 114;         // letterbox fill, as in training
export const MEAN = 0.5, STD = 0.25;

// Bilinear resize with Pillow's antialiasing (Image.resize(..., BILINEAR)) for one 8-bit channel.
export function resizeChannel(src, w, h, outW, outH) {
  const pass = (input, inLen, outLen, lines, lineStride, step, outLineStride, outStep) => {
    const scale = inLen / outLen;
    const fscale = Math.max(scale, 1);
    const taps = [];
    for (let i = 0; i < outLen; i++) {
      const center = (i + 0.5) * scale;
      const xmin = Math.max(Math.trunc(center - fscale + 0.5), 0);
      const xmax = Math.min(Math.trunc(center + fscale + 0.5), inLen);
      const ws = [];
      let total = 0;
      for (let x = xmin; x < xmax; x++) {
        const t = Math.abs((x - center + 0.5) / fscale);
        const wgt = t < 1 ? 1 - t : 0;
        ws.push(wgt);
        total += wgt;
      }
      taps.push([xmin, ws.map((v) => (total ? v / total : 0))]);
    }
    const out = new Uint8ClampedArray(outLen * lines);
    for (let l = 0; l < lines; l++) {
      for (let i = 0; i < outLen; i++) {
        const [xmin, ws] = taps[i];
        let acc = 0;
        for (let k = 0; k < ws.length; k++) acc += ws[k] * input[l * lineStride + (xmin + k) * step];
        out[l * outLineStride + i * outStep] = Math.round(acc);
      }
    }
    return out;
  };
  const tmp = pass(src, w, outW, h, w, 1, outW, 1);
  return pass(tmp, h, outH, outW, 1, outW, 1, outW);
}

// Letterbox an RGBA image (w x h) into a SIZE x SIZE input tensor (1, 3, SIZE, SIZE), CHW, normalised.
export function letterbox(rgba, w, h) {
  const scale = Math.min(SIZE / w, SIZE / h);
  const nw = Math.max(1, Math.round(w * scale)), nh = Math.max(1, Math.round(h * scale));
  const padX = Math.round((SIZE - nw) / 2), padY = Math.round((SIZE - nh) / 2);
  const n = SIZE * SIZE;
  const data = new Float32Array(3 * n).fill((PAD / 255 - MEAN) / STD);
  for (let c = 0; c < 3; c++) {
    const ch = new Uint8ClampedArray(w * h);
    for (let i = 0; i < w * h; i++) ch[i] = rgba[4 * i + c];
    const r = resizeChannel(ch, w, h, nw, nh);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) data[c * n + (y + padY) * SIZE + (x + padX)] = (r[y * nw + x] / 255 - MEAN) / STD;
    }
  }
  return { data, scale: nw / w, padX, padY };
}

// Decode the three output maps into boxes in original-image pixels, highest score first.
export function decode(heatmap, wh, offset, box, topK = 100) {
  const m = SIZE / STRIDE, n = m * m;
  const heat = new Float32Array(n);
  for (let i = 0; i < n; i++) heat[i] = 1 / (1 + Math.exp(-heatmap[i]));
  const cand = [];
  for (let y = 0; y < m; y++) {
    for (let x = 0; x < m; x++) {
      const v = heat[y * m + x];
      let peak = true;
      for (let dy = -1; dy <= 1 && peak; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy >= 0 && yy < m && xx >= 0 && xx < m && heat[yy * m + xx] > v) { peak = false; break; }
        }
      }
      if (peak) cand.push(y * m + x);
    }
  }
  cand.sort((a, b) => heat[b] - heat[a]);
  return cand.slice(0, topK).map((i) => {
    const x = i % m, y = Math.floor(i / m);
    const cx = x + offset[i], cy = y + offset[n + i];
    const bw = Math.max(wh[i], 0), bh = Math.max(wh[n + i], 0);
    const toOrig = (v, pad) => (v * STRIDE - pad) / box.scale;
    return {
      score: heat[i],
      x1: toOrig(cx - bw / 2, box.padX), y1: toOrig(cy - bh / 2, box.padY),
      x2: toOrig(cx + bw / 2, box.padX), y2: toOrig(cy + bh / 2, box.padY),
    };
  });
}
