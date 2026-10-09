# Security policy

## Supported versions

Security fixes are made on the latest 0.5.x release. Older 0.x releases do not
receive backports.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's "Report a
vulnerability" form on the [ILuce/deqio](https://github.com/ILuce/deqio)
Security tab. Do not open a public issue with exploit details. Include the
Deqio version (`deqio --version`), the operating system, the active model
profile and the smallest request or command that reproduces the problem.

## Threat model

Deqio is a local-first decision runtime. `deqio serve` binds to `127.0.0.1`
by default and has no authentication of its own: anyone who can reach the
port can call every `/v1/*` route, switch the active installed profile and
read Watch. Watch (`/ui/watch`, `/v1/watch`, `.deqio/watch/`) stores the full
request and response payloads of the current session, so binding to another
interface (for example `--host 0.0.0.0`) without an authenticated reverse
proxy in front exposes that data. The benchmark control channel is
authenticated with a per-server token kept in the private workspace file
`.deqio/server-control.json`.

Decision engines are upstream code that Deqio installs into isolated
environments under `.model-runtimes/` and runs as localhost-only sidecar
processes. Deqio does not sandbox them: installing a profile means trusting
its upstream package and model artifacts. Hugging Face artifacts record the
resolved revision in the local registry and in response provenance, but in
0.5.x not every catalog package and artifact is pinned to an immutable
revision, so two installations made at different times can differ. Catalog
pinning is planned for 0.6.

Response provenance and attestation are local and unsigned: the attestation
digest binds a result to the runtime identity and score provenance inside one
Deqio process, but it is not a cryptographic proof for a remote party. Model
outputs are probabilities, not safety guarantees; applications must keep the
decision to act (and any side effects) in their own deterministic policy or
with a human.
