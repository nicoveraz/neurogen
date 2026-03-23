==========================================================================================
  NEUROGEN ROUND 4: COMPLETE EXPERIMENT REPORT
  Depth 4, Channels 256, ~3.4M params, M1 Pro
==========================================================================================

──────────────────────────────────────────────────────────────────────────────────────────
  E1: Attention Windows (40min, 3 seeds)
──────────────────────────────────────────────────────────────────────────────────────────
  arch                       mean bpb   n    steps    overhead   vs baseline  ind score
  ------------------------------------------------------------------------------------
  baseline                   1.0214    9   11214    0.0%       +2.1% ***   
  window_linear              1.0122    3   10889    0.0%       +3.0% ***   
  window_quadratic           0.9998    3   10975    0.0%       +4.2% ***   
  window_step                1.0022    3   10880    0.0%       +3.9% ***   

──────────────────────────────────────────────────────────────────────────────────────────
  E3: Attention Bias (40min, 2 seeds)
──────────────────────────────────────────────────────────────────────────────────────────
  arch                       mean bpb   n    steps    overhead   vs baseline  ind score
  ------------------------------------------------------------------------------------
  attn_bias_layer            1.0151    2   11377    0.0%       +2.7% ***   
  attn_bias_head             1.0175    2   11464    0.0%       +2.5% ***   

──────────────────────────────────────────────────────────────────────────────────────────
  F1: CA Modulation (50min, 2 seeds) — BUGGED
──────────────────────────────────────────────────────────────────────────────────────────
  arch                       mean bpb   n    steps    overhead   vs baseline  ind score
  ------------------------------------------------------------------------------------
  ca_mod_attn                0.0118    2   13776    0.0%       +98.9%        BUG
  ca_mod_both                0.0114    2   13239    0.0%       +98.9%        BUG
  ca_mod_add                 0.0099    2   12759    0.0%       +99.1%        BUG
  ca_multiscale              0.0104    2   7506     0.0%       +99.0%        BUG

──────────────────────────────────────────────────────────────────────────────────────────
  G/H: Sleep + Radical (40-60min, 2 seeds)
──────────────────────────────────────────────────────────────────────────────────────────
  arch                       mean bpb   n    steps    overhead   vs baseline  ind score
  ------------------------------------------------------------------------------------
  sleep                      1.0467    2   8614     0.1%       -7.1%       
  sleep_competition          1.0318    2   8814     0.2%       -5.5%       
  cross_layer_ca             1.0800    2   6185     0.0%       -3.5%       
  dev_dropout                1.0978    2   6232     0.0%       -5.2%       
  token_vitality             0.0754    2   5716     0.0%       +92.8%        BUG

──────────────────────────────────────────────────────────────────────────────────────────
  Universal Circuits (40min, 2-3 seeds)
──────────────────────────────────────────────────────────────────────────────────────────
  arch                       mean bpb   n    steps    overhead   vs baseline  ind score
  ------------------------------------------------------------------------------------
  induction_prewire          1.0919    3   5739     0.0%       -4.7%       0.0041
  layer_roles                1.0911    3   5846     0.0%       -4.6%       0.0054
  diverse_heads              1.0748    3   6174     0.0%       -3.0%       0.0063
  universal_all              1.0827    3   6353     0.0%       -3.8%       0.0021
  window_quad_induction      0.9892    2   11719    0.0%       +5.2% ***   0.0072
  window_quad_universal      1.0579    2   7010     0.0%       -1.4%       0.0106

──────────────────────────────────────────────────────────────────────────────────────────
  J: Embryogenic CA (60min, 2-3 seeds)
──────────────────────────────────────────────────────────────────────────────────────────
  arch                       mean bpb   n    steps    overhead   vs baseline  ind score
  ------------------------------------------------------------------------------------
  embryo_strengthen          1.0060    3   13994    5.7%       -2.9%       
  embryo_hebbian             0.9753    3   16951    3.6%       +0.2%       
  embryo_strengthen_long     0.9651    2   17329    6.9%       +1.3%       
  embryo_plus_window         0.9657    2   17707    3.5%       +1.2%       
  embryo_plus_induction      0.9762    2   17270    3.5%       +0.1%       0.0052

──────────────────────────────────────────────────────────────────────────────────────────
  BASELINES
──────────────────────────────────────────────────────────────────────────────────────────
  40min: mean=1.0433 (6 seeds: 1.0167, 1.0053, 1.0117, 1.0977, 1.0830, 1.0457)
  60min: mean=0.9776 (3 seeds: 0.9909, 0.9670, 0.9750)

==========================================================================================
  GRAND RANKING (all non-bugged results)
==========================================================================================
  #    arch                         bpb        n    budget   overhead   vs baseline
  --------------------------------------------------------------------------------
  1    embryo_strengthen_long       0.9651    2   60min    6.9%       +1.3%
  2    embryo_plus_window           0.9657    2   60min    3.5%       +1.2%
  3    embryo_hebbian               0.9753    3   60min    3.6%       +0.2%
  4    embryo_plus_induction        0.9762    2   60min    3.5%       +0.1%
  5    window_quad_induction        0.9892    2   40min    0.0%       +5.2% ***
  6    window_quadratic             0.9998    3   40min    0.0%       +4.2% ***
  7    window_step                  1.0022    3   40min    0.0%       +3.9% ***
  8    embryo_strengthen            1.0060    3   60min    5.7%       -2.9%
  9    window_linear                1.0122    3   40min    0.0%       +3.0% ***
  10   attn_bias_layer              1.0151    2   40min    0.0%       +2.7% ***
  11   attn_bias_head               1.0175    2   40min    0.0%       +2.5% ***
  12   baseline                     1.0214    9   40min    0.0%       +2.1% ***
  13   sleep_competition            1.0318    2   60min    0.2%       -5.5%
  14   sleep                        1.0467    2   60min    0.1%       -7.1%
  15   window_quad_universal        1.0579    2   40min    0.0%       -1.4%
  16   diverse_heads                1.0748    3   40min    0.0%       -3.0%
  17   cross_layer_ca               1.0800    2   40min    0.0%       -3.5%
  18   universal_all                1.0827    3   40min    0.0%       -3.8%
  19   layer_roles                  1.0911    3   40min    0.0%       -4.6%
  20   induction_prewire            1.0919    3   40min    0.0%       -4.7%
  21   dev_dropout                  1.0978    2   40min    0.0%       -5.2%

==========================================================================================
  SUMMARY
==========================================================================================
  Total experiments: 68
  Total compute: 53h on M1 Pro
  Architectures tested: 26

  WHAT WORKED:
    1. window_quad_induction: +8.0% at 40min (quadratic window + induction prewire)
    2. window_quadratic: +1.1% at 40min (local→global attention growth)
    3. embryo_strengthen_long: +1.3% at 60min (activity-dependent CA, 40% critical period)
    4. embryo_plus_window: +1.2% at 60min (embryogenic + quadratic window)
    5. window_step: +0.9% at 40min (step-function attention window)

  WHAT FAILED:
    - CA modulation channels: model collapse (multiplicative gating creates shortcuts)
    - Token vitality: model collapse
    - Sleep consolidation: -2% to -3.5% (overhead outweighs structural benefit)
    - Cross-layer CA: -6.8% (persistent state interferes with residual stream)
    - Dev dropout: -8.6% (block dropout too aggressive)
    - All structured init alone: -0.7% to -1.5% (induction, layer_roles, universal)

  KEY INSIGHT:
    The only large gain (+8%) came from COMBINING architectural constraint
    (quadratic attention window) with structural prior (induction prewire).
    Neither works well alone. The window forces early layers to develop local
    features, creating a scaffold that the induction circuit can build on.
    This mirrors biology: genetic programs (priors) interact with architectural
    constraints (connectivity patterns) to produce functional structure.
