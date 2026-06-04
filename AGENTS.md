# CCD-MAVD Codex Rules

## Project Locations

- Server project directory: `/home/han/CCD-MAVD`
- Local publishing directory: `D:\桌面\CCD-MAVD`
- GitHub repository: `https://github.com/sqhhyyds1/CCD-MAVD.git`

The server project is the main place for experiments and GPU runs. The local
project is the publishing checkpoint used for GitHub pushes.

## File Safety

Do not bulk-delete files or directories.

Never use:

- `del /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

When deleting is truly needed, delete one explicit file path at a time. If a
batch cleanup is needed, stop and ask the user to delete the files manually.

## Remote Command Hygiene

The local shell is Windows PowerShell. The server shell is Linux shell. When
running commands through `ssh han@10.129.30.100`, avoid letting local PowerShell
interpret remote shell syntax first.

- Prefer sending complex remote scripts through standard input.
- Strip Windows CRLF before sending multiline scripts to Linux.
- Avoid fragile nested quoting, shell pipes, redirections, regex alternation,
  and here-docs inside one PowerShell string.
- Do not assume `rg` exists on the server. Check it first, otherwise use
  `grep`, `find`, or Python.
- Use remote Python for JSON parsing and table generation instead of complex
  shell pipelines.

## Server Runtime Defaults

- Connect to the server with `ssh han@10.129.30.100`.
- Run project code under the `hh` virtual environment by default.
- Use `/home/han/miniconda3/envs/hh/bin/python` explicitly when running Python
  scripts on the server.
- GPU is the default device for experiments.
- When starting training, explicitly set `CUDA_VISIBLE_DEVICES`, the Python
  path, and the output log path. After launch, verify the process with `pgrep`
  and check the GPU mapping with `nvidia-smi`.

## Server-To-Local-To-GitHub Sync Rule

After every small completed stage, pause further development and run the full
sync and publishing checkpoint:

1. On the server, confirm the current stage has passed the relevant basic
   verification, such as smoke test（冒烟测试）, unit test（单元测试）, data audit
   （数据核验）, or short training log check（短训练日志检查）.
2. Sync code from `/home/han/CCD-MAVD` back to `D:\桌面\CCD-MAVD`.
3. Sync only code, configs, docs, tests, and lightweight data lists. Do not sync
   or commit large data and run artifacts, including `data/raw`, `data/features`,
   `checkpoints`, `outputs`, `logs`, `experiments`, and `wandb`.
4. In the local project, run `git status -sb` and verify that no large data,
   checkpoints, logs, or unrelated files are staged.
5. Commit locally and push to GitHub.
6. Continue the next server-side stage only after the GitHub push succeeds.

If server-side stage changes have not been synced and pushed, do not continue
accumulating more experimental code on the server.

## Language Rule

When answering the user, English technical terms should be followed by Chinese
explanations in parentheses where useful, for example `remote`（远端仓库） and
`commit`（提交）.
