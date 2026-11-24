# Context engineering exercise

Exercise for lecture 7 of "Agentic AI and Foundation Models in Civil and Building Engineering" at TUM, on 26 November 2026.
Three agent strategies answer questions about IFC building models, and the class compares their accuracy, tokens, cost and time:

- The single agent (`--strategy single`) runs Python on the IFC files and keeps every result in its context.
- Smart compaction (`--strategy compaction`) is the single agent plus a warning.
  The warning comes once its context passes the compaction threshold.
  The agent then decides what to drop or summarise.
- The orchestrator (`--strategy orchestrator`) cannot run Python. It starts one ifc-worker subagent per IFC file, and each ifc-worker answers its questions in its own context.

Every agent, each ifc-worker included, has a context cap of 48000 tokens.

## Before the lecture

The setup takes about 15 minutes, most of it downloads, so do it the evening before.
It needs a GitHub account, about 1 GB of free disk space and the OpenRouter key handed out for the lecture.
The commands work in PowerShell on Windows and in the terminal on macOS and Linux.
On Windows, use PowerShell.
The Command Prompt (cmd) cannot run the key command of step 6.

1. Check that git is installed:

   ```
   git --version
   ```

   If the command is not found, install git: on Windows from https://git-scm.com/downloads/win with the default options; on macOS with `xcode-select --install`, or by accepting the dialog that `git --version` opens; on Linux with the package manager (for example `sudo apt install git`).
   Close the terminal and open a new one, then give git a name and an email for the commits, once per laptop:

   ```
   git config --global user.name "Firstname Lastname"
   git config --global user.email "firstname.lastname@tum.de"
   ```

2. Install uv, which then installs Python 3.12 and every package of the exercise by itself.
   On Windows, in PowerShell:

   ```
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   On macOS and Linux:

   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Close the terminal, open a new one, and check that this prints a version number:

   ```
   uv --version
   ```

3. Fork this repository: press Fork at the top right of this page, then Create fork.
   The fork is a copy of the repository under the student's own GitHub account.
   Runs are handed in from the fork by pull request.

4. Clone the fork and go into its folder, with the GitHub user name in place of `<github-user>`:

   ```
   git clone https://github.com/<github-user>/context-engineering-exercise.git
   cd context-engineering-exercise
   ```

5. Download the eleven IFC models (110 MB) into `models/`.
   The first `uv run` also installs Python and the packages, which takes a few minutes.

   ```
   uv run fetch_models.py
   ```

6. Set the OpenRouter key.
   On Windows, in PowerShell:

   ```
   $env:OPENROUTER_API_KEY = "sk-or-..."
   ```

   On macOS and Linux:

   ```
   export OPENROUTER_API_KEY="sk-or-..."
   ```

   The key holds only in this terminal window, so set it again in every new window.
   Never commit the key or paste it into an issue or a pull request.

7. Check the setup.
   The check makes one model call, which costs a fraction of a cent.

   ```
   uv run python -m harness.check
   ```

   The last line reads `all checks passed`.
   A line starting with `FAIL` names the problem; the section "When something fails" below lists the fixes.

## On the lecture day

1. Get the questions.
   They are added to `questions/` in this repository on the morning of the lecture.
   Open a terminal and go into the folder of the clone:

   ```
   cd context-engineering-exercise
   ```

   Then run:

   ```
   git pull https://github.com/sylvainHellin/context-engineering-exercise.git main
   ```

   Pressing Sync fork on the fork's GitHub page and then running `git pull` in the clone does the same.
   Until the questions are in, `harness.run` stops with `no task set`.

2. Set the OpenRouter key in the terminal window (step 6 above).

3. Run the strategy and the task set assigned to the group, with the group name from the lecture:

   ```
   uv run python -m harness.run --strategy single --set n05 --group g07 --submit-dir submissions
   ```

   Keep the command on one line: PowerShell does not accept the backslash line breaks of macOS and Linux.
   In the instructor's pilot a run took from a few seconds (`n01`) to about seven minutes (`n20` and `n40`).

4. Open the trace of the run (section "Reading the trace") and hand the run in (section "Handing in a run").

## Options of harness.run

- `--strategy`: `single`, `compaction` or `orchestrator`.
- `--set`: `n01`, `n05`, `n10`, `n20` or `n40`: 1 to 40 questions spread over the eleven IFC files.
  `single_4351_n05` and `single_4351_n10`: 5 or 10 questions on one file, `4351.ifc`.
- `--group`: the group name from the lecture, for example `g07`.
- `--submit-dir submissions`: also writes the hand-in folder `submissions/<group>/<run-id>/`.
- `--threshold`: the compaction threshold in tokens, 30000 by default, used by smart compaction only.
  At 1.5 times the threshold (45000 tokens) the harness drops old tool results itself, whatever the agent did with the warning.
- `--context-cap`: the context cap of each agent in tokens, 48000 by default.
- `--no-context-cap`: gives every agent the model's full context window instead.
- `--model`: an OpenRouter model id; the default is `deepseek/deepseek-v4.1-flash-1017`.

This command lists every option:

```
uv run python -m harness.run --help
```

## Reading the trace

A run writes its files to `runs/<run-id>/`, which git ignores.
The run id has the form `<group>-<strategy>-<set>-<n>`, for example `g07-single-n05-1`, and the run prints it at the end.
This command opens the trace in the browser:

```
uv run python -m harness.view runs/g07-single-n05-1
```

The trace lists every model call and tool call in order, with its tokens, cost and time.
In an orchestrator run, the calls of each ifc-worker appear indented under the `task` call that started it.

## Handing in a run

The answer key is not in this repository: GitHub scores the runs after the lecturer merges them.

1. Check the hand-in folders:

   ```
   uv run python -m harness.submission
   ```

   The last line reads `0 problem(s)`.

2. Commit the group's folder and push it to the fork, with the group name in place of `g07`:

   ```
   git add submissions/g07
   git commit -m "g07: single on n05"
   git push
   ```

3. Open a pull request: on the fork's GitHub page, press Contribute, then Open pull request, then Create pull request.
   Its check passes (green tick) when the pull request only adds hand-in folders and each folder is complete.

4. The lecturer merges the pull requests during the lecture.
   After each merge, GitHub scores every hand-in into `results/runs.csv` and `results/plot.png` on the `main` branch of this repository.

One pull request can carry several runs.
Push more commits to it until it is merged, then open a new pull request for later runs.

## How a run ends

At the end the run prints its status, the number of answered questions, the tokens, the cost and the time.
Every run writes its answers and, with `--submit-dir submissions`, its hand-in folder, whatever its status:

- `ok`: the agent answered every question.
- `incomplete_answers`: the agent finished but left some questions unanswered.
- `context_overflow`: an agent reached the context cap.
  A single agent or a smart-compaction agent ends the run there.
  An ifc-worker that reaches the cap fails only its own task, and the orchestrator carries on.
- `step_limit`: the run stopped after 500 steps (about 150 model calls), a guard against runaway cost.
- `timeout` or `error`: a model call failed, and the `error:` line below the status names the cause.
- `interrupted`: the run was stopped with Ctrl-C.

A `context_overflow` is a result of the strategy, not a bug to fix: hand the run in like any other.
The same holds for every other status, since the results table shows each run's status next to its score.

## When something fails

- `uv` is not found after the install: close the terminal and open a new one.
- `harness.check` prints `FAIL  API key`: the key is not set in this terminal window; set it (step 6 of the setup).
- `harness.check` prints `FAIL  model call: UnauthorizedResponseError: User not found.`: the key is wrong or incomplete; copy it again and set it between the quotes.
- `harness.check` prints `FAIL  IFC models`: run `uv run fetch_models.py` again; it downloads only the missing or damaged files.
- `harness.run` stops with `no task set`: the `--set` name is misspelt, or the questions are not in the clone yet.
  Get them with step 1 of the lecture day.
- A rate limit (`TooManyRequestsResponseError` or code 429): all groups share one key, so OpenRouter may refuse calls for a moment.
  The harness waits and retries by itself, up to five times over about a minute.
  If the run still ends with status `error` and the error names a rate limit or 429, wait a minute and start the same command again.
  The new run gets the next run id; hand in both.
- `git push` asks for a password: GitHub does not accept the account password there.
  Git for Windows opens a browser window to sign in.
  On macOS and Linux, create a token on GitHub: Settings, Developer settings, Personal access tokens, Tokens (classic), Generate new token (classic), scope `repo`.
  Give the GitHub user name as the user name and paste the token as the password.
  The terminal shows nothing while the token is pasted.
- The check of the pull request fails: open Details next to the check on the pull request page.
  Each `FAIL` line names the folder and the problem.
  A pull request may only add files in `submissions/<group>/`; remove anything else, commit and push again.

## Rules

- The agent answers from the IFC files with its Python tool only, without web access.
- A pull request changes nothing outside `submissions/<group>/`; the lecturer closes any other.
