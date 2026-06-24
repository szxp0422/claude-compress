# claude-compress evaluation

> Live run against the real API. Numbers are ground truth.

### Full pipeline (default config)

- tasks: 4  |  scored turns: 3  |  total turns: 9
- **input saved: 7.0%** (95% CI -0.9–17.2%)
- cost/turn: $0.00168 → $0.00125  (**25.4% cheaper**)
- quality (0–1): baseline 1.000, compressed 1.000
- mean quality loss +0.000 (95% CI +0.000–+0.000); margin 0.030 → **PASS ✅ non-inferior**
- win/tie/loss for compressed: 0/3/0
- savings by turn (watch for drift): t0:0%  t1:-1%  t2:26%  t3:41%

### Ablation: − checkpoint

- tasks: 4  |  scored turns: 3  |  total turns: 9
- **input saved: 1.5%** (95% CI -1.3–6.1%)
- cost/turn: $0.00151 → $0.00141  (**6.7% cheaper**)
- quality (0–1): baseline 1.000, compressed 1.000
- mean quality loss +0.000 (95% CI +0.000–+0.000); margin 0.030 → **PASS ✅ non-inferior**
- win/tie/loss for compressed: 0/3/0
- savings by turn (watch for drift): t0:0%  t1:5%  t2:1%  t3:-1%

### Ablation: − dedup

- tasks: 4  |  scored turns: 3  |  total turns: 9
- **input saved: -2.2%** (95% CI -6.1–1.8%)
- cost/turn: $0.00143 → $0.00148  (**-3.1% cheaper**)
- quality (0–1): baseline 1.000, compressed 1.000
- mean quality loss +0.000 (95% CI +0.000–+0.000); margin 0.030 → **PASS ✅ non-inferior**
- win/tie/loss for compressed: 0/3/0
- savings by turn (watch for drift): t0:0%  t1:2%  t2:-11%  t3:-13%

### Ablation: − delta (cache)

- tasks: 4  |  scored turns: 3  |  total turns: 9
- **input saved: -0.6%** (95% CI -7.6–7.1%)
- cost/turn: $0.00145 → $0.00142  (**1.8% cheaper**)
- quality (0–1): baseline 1.000, compressed 1.000
- mean quality loss +0.000 (95% CI +0.000–+0.000); margin 0.030 → **PASS ✅ non-inferior**
- win/tie/loss for compressed: 0/3/0
- savings by turn (watch for drift): t0:0%  t1:9%  t2:-12%  t3:-20%
