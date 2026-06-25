# claude-compress evaluation

> Live run against the real API. Numbers are ground truth.

### Full pipeline (default config)

- tasks: 7  |  scored turns: 10  |  total turns: 44
- **input saved: 17.3%** (95% CI 7.9–27.3%)
- cost/turn: $0.03574 → $0.02260  (**36.8% cheaper**)
- quality (0–1): baseline 0.400, compressed 0.600
- mean quality loss -0.200 (95% CI -0.700–+0.400); margin 0.030 → **FAIL ❌ quality regressed**
- win/tie/loss for compressed: 5/2/3
- savings by turn (watch for drift): t0:0%  t1:-6%  t2:-3%  t3:-0%  t4:-10%  t5:-12%  t6:-11%  t7:-8%  t8:-7%  t9:7%  t10:6%  t11:15%  t12:22%  t13:28%  t14:59%  t15:62%  t16:67%  t17:69%  t18:72%  t19:78%  t20:75%  t21:76%  t22:78%  t23:80%  t24:78%

### Ablation: − checkpoint

- tasks: 7  |  scored turns: 10  |  total turns: 44
- **input saved: 8.9%** (95% CI 5.6–12.5%)
- cost/turn: $0.03548 → $0.02008  (**43.4% cheaper**)
- quality (0–1): baseline 0.400, compressed 0.600
- mean quality loss -0.200 (95% CI -0.700–+0.300); margin 0.030 → **FAIL ❌ quality regressed**
- win/tie/loss for compressed: 5/2/3
- savings by turn (watch for drift): t0:0%  t1:1%  t2:1%  t3:3%  t4:4%  t5:2%  t6:2%  t7:2%  t8:2%  t9:13%  t10:12%  t11:21%  t12:14%  t13:12%  t14:18%  t15:17%  t16:19%  t17:23%  t18:27%  t19:31%  t20:29%  t21:32%  t22:31%  t23:32%  t24:31%

### Ablation: − dedup

- tasks: 7  |  scored turns: 10  |  total turns: 44
- **input saved: 16.9%** (95% CI 7.8–26.6%)
- cost/turn: $0.03612 → $0.02217  (**38.6% cheaper**)
- quality (0–1): baseline 0.450, compressed 0.550
- mean quality loss -0.100 (95% CI -0.500–+0.300); margin 0.030 → **FAIL ❌ quality regressed**
- win/tie/loss for compressed: 3/5/2
- savings by turn (watch for drift): t0:0%  t1:-2%  t2:-2%  t3:-5%  t4:-1%  t5:1%  t6:-1%  t7:-1%  t8:-0%  t9:-0%  t10:-2%  t11:-2%  t12:-2%  t13:-2%  t14:65%  t15:67%  t16:67%  t17:69%  t18:71%  t19:73%  t20:74%  t21:75%  t22:78%  t23:76%  t24:77%

### Ablation: − delta (cache)

- tasks: 7  |  scored turns: 10  |  total turns: 44
- **input saved: 23.9%** (95% CI 15.5–32.9%)
- cost/turn: $0.03643 → $0.02297  (**36.9% cheaper**)
- quality (0–1): baseline 0.500, compressed 0.500
- mean quality loss +0.000 (95% CI -0.500–+0.500); margin 0.030 → **FAIL ❌ quality regressed**
- win/tie/loss for compressed: 3/4/3
- savings by turn (watch for drift): t0:0%  t1:-1%  t2:-0%  t3:3%  t4:7%  t5:34%  t6:32%  t7:26%  t8:22%  t9:31%  t10:34%  t11:47%  t12:49%  t13:44%  t14:40%  t15:37%  t16:37%  t17:69%  t18:72%  t19:74%  t20:75%  t21:81%  t22:78%  t23:81%  t24:81%
