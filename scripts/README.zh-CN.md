# scripts/ — 算力、跑实验与效率（分工 C）

English: [README.md](README.md)

把这个仓库里的一份配置变成 GPU 机器上的一行结果，并在事后说清它花了多少算力，需要的东西都在这里。
这份分工的说明在 [work/C-compute-runs-efficiency.zh-CN.md](../work/C-compute-runs-efficiency.zh-CN.md)。

这里没有任何一处写出某个集群的机器名、账号或文件夹。机器的情况写在 `scripts/site.env` 里，git 会忽略它；
复制 `scripts/site.env.example` 再填好即可。

## 按顺序做

```bash
cp scripts/site.env.example scripts/site.env   # 然后填好
scripts/submit.sh setup                        # 1. 建环境               （CPU，需要外网）
scripts/submit.sh fetch --all                  # 2. 数据划分和模型权重   （CPU，需要外网）
scripts/submit.sh -r results/smoke grid configs/grid/smoke_gpu.txt    # 3. 每个模型家族一次很小的运行（GPU）
scripts/submit.sh -r results/jobs/t1a grid configs/grid/tier1_transformers.txt 1-5   # 4. 网格（GPU）
python scripts/runlog.py                       # 5. 提交了什么，各自怎样结束
python scripts/collect.py results/jobs/*       # 6. 把每个作业的结果行并进 results/
python -m levellens.results                    # 7. 重建排行榜
python scripts/efficiency.py                   # 8. 效率表，带 R6 和 R7 的判定
```

每个 GPU 作业用 `-r` 指定一个自己的结果库，这样两个作业不会同时往同一个 `runs.jsonl` 里追加；
跑完后用 `collect.py` 合并。

第 1 步和第 2 步需要外网；GPU 作业不需要，因为它们从第 2 步填好的缓存里读数据划分和权重。在没有调度器的
机器上，用 `bash` 直接运行作业脚本，不用提交：`bash scripts/job.sbatch data=arc model=distilbert head=ce`。

## 各个文件

| 文件 | 是什么 |
|---|---|
| `site.env.example` | 一台机器的设置：文件夹、模块、怎样激活环境、向调度器申请什么 |
| `job_setup.sh` | 每个作业的开头部分：找到仓库，读 `site.env`，安排几个大文件夹的位置，加载模块，激活环境 |
| `stage_cache` | 把数据划分和权重拷到计算节点的本地盘，运行时就不用经网络读取 |
| `setup.sbatch` | C1，CPU 作业：按 `requirements.txt` 建环境，跑测试，打印版本 |
| `fetch.sbatch` | C2，CPU 作业：先 `download_data.py`，再做离线检查 |
| `job.sbatch` | C3，GPU 作业：一次运行。脚本名后面的内容都传给 `levellens.run` |
| `grid.sbatch` | C3，GPU 作业：网格文件的一段，在一个作业里一次接一次地运行 |
| `tune.sbatch` | C4，GPU 作业：Optuna 搜索。一次只跑一个，因为 study 是单个文件 |
| `submit.sh` | C3 和 C4：提交四个作业脚本之一，并把这次提交记进 `logs/submissions.tsv` |
| `download_data.py` | C2：四个数据集、DeBERTa-v3、DistilBERT、GloVe、开源大模型，以及证明作业能离线运行的 `--check` |
| `env_report.py` | C1：`docs/compute.md` 里的硬件和软件表；`--require-gpu` 让没有 GPU 的作业直接失败 |
| `runlog.py` | C4：运行日志。读取提交记录，并向调度器查询每个作业怎样结束 |
| `collect.py` | C5：把各作业结果库里的结果行和预测并进 `results/` |
| `latency.py` | C6：一个保存好的模型在你指定的设备上每道题用多少秒 |
| `efficiency.py` | C6：效率表，以及 R6 和 R7 的判定 |

`env_report.py`、`runlog.py` 和 `efficiency.py` 只用标准库，所以在没建完的环境里、在任务 C7 的
干净环境检查里也能用。

## 东西写到哪里

| | |
|---|---|
| `$LEVELLENS_DATA` | 数据划分文件，运行期间只读 |
| `$HF_HOME` | 模型权重，运行期间只读 |
| `$LEVELLENS_CACHE` | 分词后的数据划分、保存的模型、Optuna 的 study。运行时会写这里 |
| `$LEVELLENS_RESULTS` | `runs.jsonl`、预测、表格。运行时会写这里 |
| `logs/` | 作业输出、`submissions.tsv`、版本报告。git 会忽略它 |

这四个变量的说明在 [docs/interfaces.zh-CN.md](../docs/interfaces.zh-CN.md)。`site.env` 可以逐个设置它们，
也可以只设 `LEVELLENS_PROJECT_DIR`，让 `job_setup.sh` 把四个都放到它下面。

我们自己这些运行的硬件、运行日志和可复现性说明在 [docs/compute.zh-CN.md](../docs/compute.zh-CN.md)。
