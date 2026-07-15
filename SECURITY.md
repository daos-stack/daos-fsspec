# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.2.x   | :white_check_mark: |
| 1.1.x   | Security fixes only |
| 1.0.x   | Security fixes only |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in daos-fsspec, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email **oss@enakta.com** with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation plan within 7 days.

## Scope

daos-fsspec is a pure-Python ctypes wrapper around the DAOS DFS C library. Security concerns may include:

- **Handle lifecycle bugs** — use-after-free, double-close, or handle leaks that could cause crashes or memory corruption
- **Path traversal** — insufficient validation of user-supplied paths that could escape the DFS container root
- **Credential exposure** — accidental logging or caching of DAOS pool/container credentials
- **Denial of service** — unbounded resource allocation (threads, memory, file handles)

## Out of Scope

- Vulnerabilities in the DAOS C library itself (report to [daos-stack/daos](https://github.com/daos-stack/daos))
- Vulnerabilities in fsspec (report to [fsspec/filesystem_spec](https://github.com/fsspec/filesystem_spec))
- Issues requiring root access to the DAOS cluster or host system

## Path Handling Notes

`daos-fsspec` normalizes `dfs://pool/container/path` URIs to paths inside the
selected DFS container, rejects NUL bytes, and enforces DAOS path/component
length limits before calling libdfs. DAOS DFS itself enforces the container
namespace boundary; there is no host-filesystem path escape from paths such as
`..`. Callers should still treat paths as untrusted input for authorization
decisions in their own applications.
