# README Update Summary

## New Features Added to Notebook (Not Yet in README)

### 1. **Claim Type Classification**
Backstory claims are automatically categorized into 5 types:
- **EVENT**: Time, place, concrete past actions
- **BELIEF**: Internal states, assumptions, fears
- **TRAIT**: Persistent personality or habits
- **WORLD_RULE**: Assumptions about how the world works
- **RELATIONSHIP**: Ties to specific characters

### 2. **Contradiction Severity Classification**
The system distinguishes between different types of contradictions:
- **HARD_CONTRADICTION**: Violates explicit narrative constraints (e.g., EVENT contradictions)
- **SOFT_TENSION**: Creates narrative strain but allows character development (e.g., BELIEF contradictions)
- **UNCONSTRAINED**: No direct evidence found in the novel

### 3. **Character Absolute Constraints**
Pre-defined narrative constraints for known characters:
- E.g., Edmond Dantès: Cannot have traveled during imprisonment at Château d'If
- Constraint-based retrieval for EVENT-type claims
- Automatic hard contradiction detection

### 4. **Enhanced Evidence Output**
Each prediction now includes:
- **Claim type** classification (EVENT/BELIEF/TRAIT/WORLD_RULE/RELATIONSHIP)
- **Claim status** (HARD_CONTRADICTION/SOFT_TENSION/UNCONSTRAINED)
- **Hard contradiction count**
- **Soft tension count**
- **Entailment count**

### 5. **File Naming Update**
- Output file changed from `predictions.csv` to `results.csv`

---

## Implementation Details

### Cell 13: Claim Classification Logic
```python
CLAIM_TYPES = {
    "EVENT",      # Time, place, concrete past actions
    "BELIEF",     # Internal states, assumptions
    "TRAIT",      # Persistent personality
    "WORLD_RULE", # How the world works
    "RELATIONSHIP" # Ties to others
}

def classify_claim_type(claim_text: str) -> str:
    # Pattern matching for claim categorization
    # EVENT: time markers, locations
    # BELIEF: internal state verbs
    # etc.
```

### Cell 14: Contradiction Severity
```python
def classify_contradiction_severity(nli_label: str, claim_type: str, claim_text: str) -> str:
    if 'CONTRADICTION' not in nli_label.upper():
        return "UNCONSTRAINED"
    
    # Events and world rules are hard constraints
    if claim_type in {"EVENT", "WORLD_RULE"}:
        return "HARD_CONTRADICTION"
    
    # Beliefs, traits, relationships allow narrative tension
    if claim_type in {"BELIEF", "TRAIT", "RELATIONSHIP"}:
        return "SOFT_TENSION"
```

### Cell 14: Character Constraints
```python
CHARACTER_ABSOLUTE_CONSTRAINTS = {
    ("The Count of Monte Cristo", "Edmond Dantès"): [
        "Château d'If",
        "imprisoned",
        "fourteen years",
        "cut off from the world"
    ]
}
```

---

## Updated Evidence Example

**Old Format:**
```
Evidence:
Claim: [claim text]
Passage: [novel text]
NLI: contradiction (score: 0.852)
```

**New Format:**
```
Evidence 1:
Claim (EVENT): During his imprisonment at the Château d'If, Edmond Dantès traveled to Paris...
Claim Status: HARD_CONTRADICTION
Passage (Lines 37239-37258): [novel text showing imprisonment]
NLI: CONTRADICTION (score: 0.952)
```

---

## README Sections That Need Updates

1. **Features Section** ✅ (Already updated)
2. **Pipeline Workflow - Stage 8** ✅ (Already updated)
3. **Output Format - test_predictions_with_evidence.csv** ✅ (Already updated)
4. **Project Structure** ✅ (Already updated - predictions.csv → results.csv)
5. **Usage - Outputs generated** ✅ (Already updated)
6. **Pipeline Workflow - Stage 9** ⏳ (Needs update)
7. **Configuration & Hyperparameters** ⏳ (Needs new section)
8. **References** ⏳ (Needs novel techniques section)

---

## Key Improvements

1. **More Granular Analysis**: Claims are now categorized by type
2. **Smarter Contradiction Detection**: Distinguishes between hard violations and narrative tension
3. **Character-Specific Logic**: Known constraints automatically checked
4. **Better Evidence**: Each piece of evidence includes claim type and severity status
5. **Clearer Output**: Separate counts for hard contradictions vs soft tensions
