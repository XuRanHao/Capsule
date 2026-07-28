# Schema revisions

Generate the first explicit schema revision after PostgreSQL is running:

```bash
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

Keeping the first revision generated from the reviewed model definitions avoids
an opaque metadata-driven bootstrap migration.
