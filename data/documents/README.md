# Knowledge Base Documents (Synthetic)

**Every document in this folder is synthetic training data written for an educational
portfolio project.** None of them is an official policy, donor record, or report of
UNICEF or any other real organization. They contain no real donor personal information.

## Format

One JSON file per document:

```json
{
  "id": "DOC001",
  "title": "Donor Retention",
  "category": "fundraising",
  "created_at": "2026-01-15",
  "source_type": "synthetic_training_document",
  "disclaimer": "Synthetic training document for an educational project. ...",
  "content": "..."
}
```

The disclaimer is kept in its own field, not inside `content`, so it is not
indexed by the retriever (otherwise every document would match questions about
"official policy").

A file may also contain a JSON array of document objects.

| Field         | Required | Notes                                          |
|---------------|----------|------------------------------------------------|
| `id`          | yes      | Unique across the whole knowledge base         |
| `title`       | yes      | Shown to users and cited as a source           |
| `category`    | yes      | Free text, e.g. `fundraising`, `analytics`     |
| `content`     | yes      | Plain text; long content is split into chunks  |
| `created_at`  | yes      | ISO date string                                |
| `source_type` | yes      | Use `synthetic_training_document` for samples  |
| `disclaimer`  | no       | Ignored by the loader; for human readers       |

## Adding a document

1. Create `DOC007_your_topic.json` with the fields above.
2. Restart `python main.py`. The TF-IDF index is rebuilt at startup.
