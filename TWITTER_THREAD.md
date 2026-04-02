# Twitter/X Thread Draft — NeuroGen

---

**Tweet 1 (Hook)**

I ran 200+ experiments testing whether biological developmental principles can improve transformers.

Most failed. But one thing works substantially: forcing early layers to attend locally, then gradually opening to full attention (quartic growth).

+1.5% at 3.4M (p=0.001), +12.9% at 125M (gap still widening).

---

**Tweet 2 (The gradient finding)**

The mechanism isn't what you'd expect.

Gradient noise is CONSTANT across all window sizes (~0.0053). What changes is signal — it increases 18x with smaller windows.

Windows don't reduce noise. They make each gradient step point in a more coherent direction.

[attach: charts/attention_entropy_per_layer.png]

---

**Tweet 3 (Entropy evidence)**

Attention entropy confirms forced specialization.

Early layers: 57% lower entropy in windowed models (0.85 vs 1.99 nats). Attention is laser-focused on nearby tokens.

Final layer: slightly higher entropy — it compensates with broader context. Exactly the hierarchy you'd want.

---

**Tweet 4 (Curriculum effect)**

Most surprising: you can REMOVE the windows after 10k steps and the model keeps the benefit.

Full attention → 0.898 bpb
Quartic windows → 0.886 bpb
Quartic 10k, then full → 0.884 bpb (best!)

The early constraint creates permanent structure. Like critical periods in brain development.

---

**Tweet 5 (Practical + CTA)**

Practical upshot:
- Zero extra parameters
- Zero extra compute (actually slightly faster with Flash Attention)
- Works as a permanent constraint OR early-training curriculum
- Just add a quartic window schedule to your attention layers

Code, paper, and all data: github.com/nicoveraz/neurogen

Seven mechanism experiments, five validated seeds, two scales. All reproducible.

---

### Suggested images to attach:
- Tweet 1: charts/window_schedule.svg (the window growth visualization)
- Tweet 2: charts/attention_entropy_per_layer.png (entropy comparison)
- Tweet 3: charts/125m_gap_evolution.svg (scaling gap widening)
- Tweet 5: charts/final_performance.svg (3.4M bar chart)
