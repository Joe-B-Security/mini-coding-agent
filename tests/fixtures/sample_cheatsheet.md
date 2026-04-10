# Sample Cheat Sheet

## Introduction

This is an overview of the topic.

## Password Storage

### Bad Example

Never do this:

```python
import hashlib
hashlib.md5(password.encode()).hexdigest()
```

### Good Example

Always use a slow KDF like bcrypt:

```python
import bcrypt
bcrypt.hashpw(password.encode(), bcrypt.gensalt())
```

See [Authentication](Authentication_Cheat_Sheet.md#passwords) for more.

## Testing

### How to Verify

Run the tests to check your implementation is correct.

## References

- [ASVS](https://owasp.org/asvs)
