LOCUS human labels

Annotations were collected iteratively: internal -> Prolific -> Snorkel.

The Snorkel data had the most hours invested and is recommended for evaluation;
other annotations are included in all (LOCUS_all_labels.csv). The codebook
used for the Snorkel data was condensed into our final prompt.

Data post-processing was improved further for v1.1 after these insights.

LOCUS_snorkel_labels.csv: 250 excerpts, three annotations each,
individual ratings and cleaned notes.
LOCUS_all_labels.csv: internal 100×1, Prolific 250×5, Snorkel 250×3
(600 rows, 500 unique excerpts).

Snorkel is already included in the combined file. Internal's 100 excerpts also
repeat in Snorkel; keep shared audit_id values together when splitting data.

Gold fields: function, topic, confidence. Majority is
2-of-3 for snorkel and 3-of-5 for prolific. Score only *_eligible=1; joint scores
require all relevant flags. An eligible blank topic is valid null; other blanks
are unscored. Prolific topics were not collected. *_votes contains JSON counts:
"null" means a null-topic vote; {} means uncollected.

Read as UTF-8 CSV with keep_default_na=False; fields contain line breaks.
Internal topics are mapped to five categories from an initial seven. Source truncation markers are retained.
