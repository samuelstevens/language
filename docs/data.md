# Data 

## Wrong SQL

* In scholar.json, the `WHERE ... LIKE ...` pattern is wrong: instead of `where year like '199'`, it should be `... like '199%'` (wildcard is needed).

```json
{
    "text": "What paper did authorname0 wrote in the 90s ?",
    "question-split": "train",
    "variables": {
        "authorname0": "Michael Armstrong",
        "year0": "199"
    }
}
```
