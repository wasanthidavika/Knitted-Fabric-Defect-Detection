# Q4 - Web application (report skeleton, ~2 pages + screenshots)

## 1. Purpose, user, decision
The user is a knitted-fabric quality inspector. Given the system's output the inspector decides whether to pass the fabric or send it for manual re-inspection, and where to look. (Use the Q1 wording.)

## 2. Architecture
Diagram: browser (HTML/JS) -> FastAPI (`/api/predict`, `/api/explain`, `/api/info`, `/health`) -> `AppState` -> `YoloDetector` (the trained ANN). Preprocessing identical to training: EXIF handling per `exif_mode`, resize to 640 px on the long side. One model call at a time (lock). All settings in `app/app_config.yaml`.

## 3. Prediction at request time
Describe the code path and point to the video moment where two different images give different outputs. Mention that nothing is cached or precomputed.

## 4. Score, threshold and uncertainty
- Score = detector confidence (not a probability).
- Validation reliability table (`calibration_check.py`): insert `reliability_<model>.png`; say whether scores are over- or under-confident, with the numbers.
- Decision threshold chosen on validation (`select_threshold.py`): value, target recall, achieved recall and false-alarm rate. State that these may not hold on new data.
- UI wording: show the exact sentences used.

## 5. Limitations shown to the user
Quote the list from `app/inference.py` (edit it to match your findings: small defects, grey vs dyed, trained on one dataset, no guarantee when nothing is found).

## 6. Testing
| Category | Input | Expected | Observed (real model) | Change made |
|---|---|---|---|---|
| Normal | defect-free fabric photo | PASS or few weak boxes | | |
| Normal | fabric with each defect type (7) | REFER, box on defect | | |
| Invalid | empty file / text file renamed .jpg / truncated JPEG | clear error (400 / 415) | automated: see tests/test_app.py | |
| Invalid | tiny image / very long thin image / oversize upload / huge dimensions | 422 / 413 | automated | |
| Out of distribution | photo of a person, a screen, woven fabric, a blank wall, an out-of-focus fabric, a very dark fabric | REFER (unusual input) or a documented failure | | |
| Missing-field / malformed API call | no file, bad box, unknown class | 422 | automated | |
Say which rows are automated (model-independent validation) and which you ran by hand with the real model. Report every failure honestly (the OOD check is a heuristic on 8 global statistics: it cannot catch subtle shifts such as a new camera or yarn).

## 7. Performance and robustness
Latency of `/api/predict` measured on your laptop (state the CPU); upload limits; no stack traces to users; limits of the single-process design.

## 8. Screenshots for the report/video
Verdict REFER with boxes; PASS; invalid-file error; unusual-input warning; explanation panel.
