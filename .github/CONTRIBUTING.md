# Contributing to [this project]

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components

## Workflow

1. fork and pull the latest mmsegmentation
2. checkout a new branch (do not use master branch for PRs)
3. commit your changes
4. create a PR

Note

- If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.
- If you are the author of some papers and would like to include your method to mmsegmentation,
  please contact Yakhyokhuja Valikhujaev (yakhyo9696\[at\]gmail\[dot\]com). We will much appreciate your contribution
  

## Code style

### Python

We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:

- [flake8](https://github.com/PyCQA/flake8): static code analyser
- [ufmt](https://github.com/omnilib/ufmt): formatter
- [pydocstyle](https://github.com/PyCQA/pydocstyle): sort imports


We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `trailing whitespaces`, 
fixes `end-of-files` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](../.pre-commit-config.yaml).

After you clone the repository, you will need to install initialize pre-commit hook.

```shell
pip install -U pre-commit
```

From the repository folder

```shell
pre-commit install
```

After this on every commit check code linters and formatter will be enforced.

## Commit Message Convention

### Commit Message Format

```
<Type>: Short description

Longer description here if necessary

BREAKING CHANGE: only contain breaking change
```
- Any line of the commit message cannot be longer 100 characters!

### Revert
```
revert: commit <short-hash>

This reverts commit <full-hash>
More description if needed
```

### Type
| Syntax      | Description                 | Detailed Description     |
| :---        | :-----                      | :----           |
| `feat`      | Features                    | A new feature   |
| `fix`       | Bug Fixes                   | A bug fix       |
| `docs`      | Documentation               | Documentation only changes      |
| `style`     | Styles                      | Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)      |
| `refactor`  | Code Refactoring            | A code change that neither fixes a bug nor adds a feature      |
| `perf`      | Performance Improvements    | A code change that improves performance      |
| `test`      | Tests        | Adding missing tests or correcting existing tests     |
| `build`     | Builds        | Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)    |
| `ci`        | Continuous Integrations     | Changes to our CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)      |
| `chore`     | Chores                      | Other changes that don't modify src or test files      |
| `revert`    | Reverts                     | Reverts a previous commit      |

### Subject
- use the imperative, __present__ tense: "change" not "changed" nor "changes"
- do capitalize the first letter
- no dot (.) at the end

### Body

- use the imperative, __present__ tense: "change" not "changed" nor "changes".
- the motivation for the change and contrast this with previous behavior.

### BREAKING CHANGE
- This commit contains breaking change(s).
- start with the word BREAKING CHANGE: with a space or two newlines. The rest of the commit message is then used for this.
