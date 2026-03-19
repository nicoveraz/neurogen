# Round 2 Results

## Track 1 — Horizon Validation

Not run.

## Track 2 — Blend & Pattern Optimization

Best blend ratio: xavier_ca5 (val_bpb: 1.2481)

Best CA pattern: xavier_block_ca (val_bpb: 1.2523)

Overall best init: xavier_ca5 (val_bpb: 1.2481, improvement over xavier: +0.6%)

### Blend Ratio Results

| method | mean val_bpb | vs xavier |
|--------|-------------|-----------|
| xavier_ca10 | 1.2603 | -0.4% |
| xavier_ca15 | 1.2554 | +0.0% |
| xavier_ca20 | 1.2513 | +0.3% |
| xavier_ca30 | 1.2630 | -0.6% |
| xavier_ca5 | 1.2481 | +0.6% |

### CA Pattern Results

| method | mean val_bpb | vs xavier |
|--------|-------------|-----------|
| xavier_block_ca | 1.2523 | +0.3% |
| xavier_grid_ca | 1.2667 | -0.9% |
| xavier_rd_spots | 1.3509 | -7.6% |
| xavier_rd_stripes | 1.3474 | -7.3% |
| xavier_spectral_ca | 1.2591 | -0.3% |

## Track 3 — Output Quality

Not run.

## Overall Conclusion

Best init is xavier_ca5 (val_bpb 1.2481).
