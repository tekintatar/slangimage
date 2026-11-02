# Security Policy

SlangImage is pre-alpha software. It compiles and runs GPU shader code; treat
`.slang` kernels from untrusted sources (including directories registered via
`ModuleCache.add_ops_dir`) with the same caution as any other code you execute.

## Reporting a vulnerability

Please report security issues **privately** rather than opening a public issue:

- Use GitHub's **[Report a vulnerability](https://github.com/tekintatar/slangimage/security/advisories/new)**
  (Security → Advisories) to open a private advisory.

We aim to acknowledge reports within a few days. Since the project is pre-1.0,
fixes ship on a best-effort basis in the next release.

## Supported versions

Only the latest release / `main` is supported while the project is pre-1.0.
