---
name: Bug Report
about: Report a bug to help us improve daos-fsspec
title: ''
labels: bug
assignees: ''
---

## Environment

- **daos-fsspec version**: <!-- e.g. 1.4.1 -->
- **Python version**: <!-- e.g. 3.11.4 -->
- **DAOS version**: <!-- e.g. 2.6.4 -->
- **OS**: <!-- e.g. Ubuntu 22.04 -->
- **Installation method**: <!-- pip install, from source, etc. -->

## Description

<!-- A clear and concise description of what the bug is. -->

## Steps to Reproduce

```python
# Minimal code that reproduces the issue
import fsspec

fs = fsspec.filesystem("dfs", pool="mypool", container="mycont")
# ...
```

## Expected Behavior

<!-- What you expected to happen. -->

## Actual Behavior

<!-- What actually happened. Include full tracebacks if applicable. -->

```
Paste traceback here
```

## Additional Context

<!-- Any other context: DAOS pool/container config, network setup, whether the issue is intermittent, etc. -->
