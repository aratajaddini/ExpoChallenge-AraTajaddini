# Security and API Key Handling

## API key authentication
Access is controlled with an API key sent in the X-API-Key request header. There
is no session or cookie. A request without a valid key is rejected with HTTP 401
before it reaches the model or the database, so an unauthenticated caller cannot
consume CPU inference time.

## Keys are never stored in plaintext
Issued keys are stored as hashes, not as readable strings. The full key value is
shown once at creation time and cannot be recovered afterwards. If a key is lost,
the correct action is to issue a new one and revoke the old, not to try to read
the stored value.

## Revoking a key
Keys are managed through the admin key endpoints, which are themselves protected.
Revocation takes effect on the next request, so a leaked key can be disabled
without redeploying or restarting the service.

## CORS allow-list
Cross-origin access is restricted to an explicit allow-list of origins rather
than a wildcard. A wildcard would be unsafe here because the browser sends the
API key on every request. When the frontend is served by the same application,
no cross-origin configuration is needed at all; the allow-list only matters when
the interface is hosted on a different port or host.

## Startup validation of secrets
The service verifies its required configuration at startup, including key
configuration and the presence of model weights. Failing fast at startup is
preferred over discovering a missing secret in the middle of a live
demonstration.
