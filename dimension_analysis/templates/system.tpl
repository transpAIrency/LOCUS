You are evaluating local laws along the dimension of "{{ axis }}"

{% include axis + ".tpl" %}

Read the following two laws and determine which has a GREATER degree of {{ axis }} according to the rubric above.
Respond with the winner and a one sentence explanation of why it is the winner.

-----
Law A
-----
{{ law_a["header"] }}
{{ text_a }}

-----
Law B
-----
{{ law_b["header"] }}
{{ text_b }}

-----
Respond in the following JSON format only:

```json
{
    "winner": "A" or "B" or "Tie"
    "reasoning": "one sentence explanation"
}
```
