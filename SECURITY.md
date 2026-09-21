# Security Policy

## Reporting

Please report security issues privately to the maintainers instead of opening a public issue with exploit details.

## Document handling

The default local deployment does not intentionally send documents to third-party services.

Public deployments should add:

- authentication as appropriate
- rate limits
- request/body size limits at the reverse proxy
- isolated worker execution
- temporary-storage quotas
- malware scanning where required
- automatic deletion policies

Uploaded filenames are not trusted as filesystem paths.
